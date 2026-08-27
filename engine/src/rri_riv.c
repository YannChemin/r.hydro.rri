/**
 * @file rri_riv.c
 * @brief River channel routing: per-reach discharge and the RK45
 * derivative source term for the river's state variable (stored
 * volume, `vr_idx`).
 *
 * This is the "how much water moves down each river reach this instant"
 * half of the coupled solver (rri_slope.c is the hillslope half); called
 * six times per accepted RK45 step by the river integration loop in
 * main.c, once per Cash-Karp stage.
 *
 * Fortran reference: RRI_Riv.f90 (`funcr`, `qr_calc`, `hq_riv` -- the
 * rectangular-channel `hq_riv` body itself lives in kernels.h as
 * `rri_k_hq_riv`, shared with the future OpenCL backend).
 *
 * NOT implemented here: dam operation, diversion, and river water-level/
 * discharge boundary conditions -- all off in the validated solo30s
 * config (see README.md), and RRI_Riv.f90's Fortran `funcr` folds them
 * in as extra branches this port omits entirely rather than stubs out.
 *
 * @note Sign convention: a positive `qr_idx[k]` means water flowing OUT
 * of cell k (downstream, toward `rc->down[k]`); reverse (upstream) flow
 * is represented as a NEGATIVE qr_idx[k], not by swapping which cell is
 * "source" -- see rri_qr_calc's `dh < 0.0` branch, which computes the
 * hydraulically correct discharge using the neighbor's depth/width but
 * still stores it at index k with a negated sign. This convention is
 * what lets rri_funcr's flux-scatter step (below) treat every cell
 * uniformly: `qr_sum[k] += qr_idx[k]` (outflow from k) and
 * `qr_sum[down[k]] -= qr_idx[k]` (that same flux arriving as inflow at
 * the downstream cell) are correct regardless of the physical flow
 * direction, because the sign already encodes it.
 */
#include "rri/rri.h"
#include "rri/kernels.h"
#include <math.h>

/**
 * @brief Per-reach river discharge, independent across cells given the
 * previous step's `hr_idx` -- the OpenMP/OpenCL parallel target
 * described in rri.h's file-level comment (PLAN.md section 2).
 *
 * For each river cell k, computes the head gradient to its single
 * downstream neighbor `rc->down[k]` and, depending on which way it
 * points, calls kernels.h's rectangular-channel Manning formula
 * (rri_k_hq_riv) using either k's or its neighbor's depth/width -- see
 * the file-level comment above for why the result's sign, not which
 * cell it's "attached to", is what encodes flow direction.
 *
 * @param rc        River cellset (topology + geometry), from rri_riv_idx_setting.
 * @param hr_idx    Water depth per river cell [m], length rc->count.
 * @param ns_river  Manning's roughness for the channel [s/m^(1/3)].
 * @param[out] qr_idx  Discharge per river cell [m^3/s], length rc->count;
 *                     positive = flowing toward rc->down[k] (see file-level
 *                     comment). Outlet cells (rc->domain[k]==2) always get
 *                     exactly 0 here -- an outlet's actual water loss is
 *                     accounted for separately in main.c's end-of-timestep
 *                     domain==2 drain step, not through this discharge.
 */
void rri_qr_calc(const rri_riv_cellset *rc, const double *hr_idx, double ns_river, double *qr_idx)
{
#pragma omp parallel for
    for (int k = 0; k < rc->count; k++) {
        if (rc->domain[k] == 2) { qr_idx[k] = 0.0; continue; }
        double zb_p = rc->zb[k], hr_p = hr_idx[k];
        double distance = rc->dis[k];
        int kk = rc->down[k];
        double zb_n = rc->zb[kk], hr_n = hr_idx[kk];

        /* Diffusive-wave head gradient: water-surface elevation
         * difference over distance. When the downstream neighbor is
         * itself the outlet, drop its (undefined/irrelevant) water depth
         * from the gradient -- water surface at an outlet is whatever
         * leaves the domain, not a depth this cell should react to. */
        double dh = ((zb_p + hr_p) - (zb_n + hr_n)) / distance;
        if (rc->domain[kk] == 2) dh = (zb_p + hr_p - zb_n) / distance;

        double q;
        if (dh >= 0.0) {
            /* Flow k -> downstream: use k's own depth, but if k's bed is
             * actually higher than the neighbor's, the driving depth is
             * only the amount by which k's water surface exceeds the
             * neighbor's BED (not k's own bed) -- a cell can still push
             * water downhill over a bed step even with a shallow local
             * depth, as long as its water surface clears the drop. */
            double hw = hr_p;
            if (zb_p < zb_n) { double t = zb_p + hr_p - zb_n; hw = t > 0.0 ? t : 0.0; }
            q = rri_k_hq_riv(hw, dh, rc->width[k], ns_river);
        } else {
            /* Reverse flow: same logic, mirrored onto the neighbor's
             * depth/width, result stored at k with a negated sign (see
             * file-level comment). */
            double hw = hr_n;
            if (zb_n < zb_p) { double t = zb_n + hr_n - zb_p; hw = t > 0.0 ? t : 0.0; }
            q = -rri_k_hq_riv(hw, fabs(dh), rc->width[kk], ns_river);
        }
        qr_idx[k] = q;
    }
}

/**
 * @brief One RK45 stage's derivative evaluation for the river's state
 * variable: converts the trial stored-volume `vr_idx` to depth, computes
 * discharge, and sums it into the outflow-minus-inflow rate `fr_idx`
 * each cell's volume ODE integrates.
 *
 * Fortran reference: RRI_Riv.f90, `funcr`.
 *
 * @param rc        River cellset.
 * @param vr_idx    Trial stored volume per cell [m^3] (the RK45 stage's
 *                  current estimate, not necessarily the accepted state).
 * @param ns_river  See rri_qr_calc.
 * @param area      Grid cell area [m^2] (rri_grid::area), for the
 *                  volume<->depth conversion (rri_vr2hr).
 * @param[out] hr_idx   Depth corresponding to vr_idx [m], length rc->count
 *                      -- also an OUTPUT here (recomputed from vr_idx at
 *                      the top of every call), not just an input, since
 *                      each RK stage advances a new trial volume.
 * @param[out] fr_idx   Volume derivative per cell [m^3/s] (`-qr_sum`,
 *                      i.e. net inflow rate), length rc->count -- this is
 *                      the actual RK45 "f(t, y)" the integrator combines
 *                      across stages.
 * @param[out] qr_idx   Per-reach discharge, see rri_qr_calc; also
 *                      returned here for main.c's flow-weighted time
 *                      averaging (qr_ave) across the accepted sub-step.
 * @param qr_sum_scratch  Caller-owned scratch buffer, length rc->count,
 *                        reused every call to avoid an allocation per RK
 *                        stage (there are 6-7 calls per accepted step).
 */
void rri_funcr(const rri_riv_cellset *rc, const double *vr_idx, double ns_river, double area,
                double *hr_idx, double *fr_idx, double *qr_idx, double *qr_sum_scratch)
{
#pragma omp parallel for
    for (int k = 0; k < rc->count; k++) hr_idx[k] = rri_vr2hr(vr_idx[k], area, rc->area_ratio[k]);

    rri_qr_calc(rc, hr_idx, ns_river, qr_idx);

    /* Flux scatter: each cell's outflow (qr_idx[k], signed per the
     * file-level comment) is subtracted from ITS downstream neighbor's
     * balance as inflow. Multiple upstream cells can share one
     * downstream neighbor, so this write pattern has a shared
     * destination across loop iterations -- kept serial rather than
     * folded into rri_qr_calc's parallel loop above (see rri.h's
     * file-level comment on why this part of the solver stays
     * host-side). */
    for (int k = 0; k < rc->count; k++) qr_sum_scratch[k] = 0.0;
    for (int k = 0; k < rc->count; k++) {
        qr_sum_scratch[k] += qr_idx[k];
        int kk = rc->down[k];
        if (rc->domain[kk] != 0) qr_sum_scratch[kk] -= qr_idx[k];
    }
    for (int k = 0; k < rc->count; k++) fr_idx[k] = -qr_sum_scratch[k];
}
