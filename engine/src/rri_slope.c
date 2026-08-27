/**
 * @file rri_slope.c
 * @brief Hillslope routing: per-direction overland+subsurface discharge
 * and the RK45 derivative source term for the slope's state variable
 * (water depth, `hs_idx`), including direct rainfall input.
 *
 * The hillslope counterpart of rri_riv.c -- called six times per
 * accepted RK45 step by the slope integration loop in main.c. Unlike
 * the river network (a tree, one downstream neighbor per cell),
 * hillslope cells can exchange water with up to `sc->lmax` neighbors
 * (rri_slo_cellset), so this file's inner loop has an extra dimension
 * (`l`, the direction slot) that rri_riv.c's does not.
 *
 * Fortran reference: RRI_Slope.f90 (`funcs`, `qs_calc` -- the discharge
 * formula and water-level correction themselves live in kernels.h as
 * `rri_k_hq_slope`/`rri_k_h2lev`, shared with the future OpenCL backend).
 *
 * NOT implemented here: slope discharge boundary conditions -- off in
 * the validated solo30s config (see README.md).
 *
 * @note Sign convention: same as rri_riv.c -- `qs_idx[l][k] > 0` means
 * outflow from cell k in direction l; reverse flow is a negative value
 * at the same [l][k] slot (see RRI_LMAX8's doc in rri.h for why the two
 * opposite directions across one edge share a slot). Kinematic-routing
 * cells (`dif==0`) use only slot 0 in effect, always via `down_1d`
 * regardless of the geometric direction that actually points to their
 * D8 downstream neighbor -- see the `dif_p == 0` branches below.
 */
#include "rri/rri.h"
#include "rri/kernels.h"
#include <math.h>
#include <string.h>

/**
 * @brief Per-direction hillslope discharge, independent across (l, k)
 * given the previous step's `hs_idx` -- the OpenMP/OpenCL parallel
 * target described in rri.h's file-level comment.
 *
 * @param sc      Hillslope cellset, from rri_slo_idx_setting.
 * @param hs_idx  Water depth per cell [m], length sc->count.
 * @param area    Grid cell area [m^2] (rri_grid::area).
 * @param[out] qs_idx  Discharge per unit area [m/s], RRI_LMAX8 arrays
 *                     each of length sc->count, indexed [l][k]; slots
 *                     beyond sc->lmax (and the kinematic-routing slots 1-3
 *                     for cells with dif==0) are left at 0.
 */
void rri_qs_calc(const rri_slo_cellset *sc, const double *hs_idx, double area, double *qs_idx[RRI_LMAX8])
{
#pragma omp parallel for
    for (int k = 0; k < sc->count; k++) {
        double zb_p = sc->zb[k], hs_p = hs_idx[k];
        double ns_p = sc->ns_slope[k], ka_p = sc->ka[k], da_p = sc->da[k], dm_p = sc->dm[k], b_p = sc->beta[k];
        int dif_p = sc->dif[k];
        int lmax = sc->lmax;

        for (int l = 0; l < RRI_LMAX8; l++) qs_idx[l][k] = 0.0;

        for (int l = 0; l < lmax; l++) {
            if (dif_p == 0 && l == 1) break; /* kinematic cells only ever use down_1d (see file-level comment); nothing beyond slot 0 applies */
            int kk = (dif_p == 0) ? sc->down_1d[k] : sc->down[l][k];
            if (kk == -1) continue;
            double distance = (dif_p == 0) ? sc->dis_1d[k] : sc->dis[l][k];
            double length = (dif_p == 0) ? sc->len_1d[k] : sc->len[l][k];

            double zb_n = sc->zb[kk], hs_n = hs_idx[kk];
            /* Head gradient uses the porosity-corrected water LEVEL
             * (rri_k_h2lev), not raw depth -- see that function's doc in
             * kernels.h for why a saturated-vs-unsaturated soil column
             * changes how much "level" a given depth represents. */
            double lev_p = rri_k_h2lev(hs_p, sc->soildepth[k], sc->gammaa[k]);
            double lev_n = rri_k_h2lev(hs_n, sc->soildepth[kk], sc->gammaa[kk]);

            double dh = ((zb_p + lev_p) - (zb_n + lev_n)) / distance;
            /* Kinematic routing floors the gradient at a small positive
             * constant instead of using the true (possibly near-zero or
             * negative) water-level gradient: kinematic cells route
             * purely by bed slope, always downhill, never reversing --
             * the 0.001 floor keeps the discharge formula well-defined
             * (and nonzero) even on a locally flat bed. */
            if (dif_p == 0) { double t = (zb_p - zb_n) / distance; dh = t > 0.001 ? t : 0.001; }

            double q;
            if (dh >= 0.0) {
                double hw = hs_p;
                if (zb_p < zb_n) { double t = zb_p + hs_p - zb_n; hw = t > 0.0 ? t : 0.0; }
                q = rri_k_hq_slope(ns_p, ka_p, da_p, dm_p, b_p, hw, dh, length, area);
            } else {
                double ns_n = sc->ns_slope[kk], ka_n = sc->ka[kk], da_n = sc->da[kk], dm_n = sc->dm[kk], b_n = sc->beta[kk];
                double hw = hs_n;
                dh = fabs(dh);
                if (zb_n < zb_p) { double t = zb_n + hs_n - zb_p; hw = t > 0.0 ? t : 0.0; }
                q = -rri_k_hq_slope(ns_n, ka_n, da_n, dm_n, b_n, hw, dh, length, area);
            }
            qs_idx[l][k] = q;
        }
    }
}

/**
 * @brief One RK45 stage's derivative evaluation for the slope's state
 * variable: rainfall input minus net lateral outflow.
 *
 * Fortran reference: RRI_Slope.f90, `funcs`.
 *
 * @param sc         Hillslope cellset.
 * @param hs_idx     Trial water depth per cell [m] (this RK stage's estimate).
 * @param qp_t_idx   Rainfall intensity per cell [m/s] (the currently-active
 *                   rain time-block, resolved and gathered onto the
 *                   cellset in main.c before each RK stage).
 * @param area       See rri_qs_calc.
 * @param[out] fs_idx   Depth derivative per cell [m/s]: rainfall in,
 *                      minus this cell's own outflow in all directions,
 *                      plus inflow scattered in from upstream neighbors
 *                      (below) -- the RK45 "f(t, y)" the integrator combines.
 * @param[out] qs_idx   Per-direction discharge, see rri_qs_calc; also
 *                      returned for main.c's flow-weighted time averaging
 *                      (qs_ave) across the accepted sub-step. Caller-owned
 *                      scratch, RRI_LMAX8 arrays of length sc->count.
 */
void rri_funcs(const rri_slo_cellset *sc, const double *hs_idx, const double *qp_t_idx,
                double area, double *fs_idx, double *qs_idx[RRI_LMAX8])
{
    rri_qs_calc(sc, hs_idx, area, qs_idx);

    for (int k = 0; k < sc->count; k++) {
        double outflow = 0.0;
        for (int l = 0; l < RRI_LMAX8; l++) outflow += qs_idx[l][k];
        fs_idx[k] = qp_t_idx[k] - outflow;
    }

    /* Flux scatter: see rri_riv.c: rri_funcr's matching comment -- same
     * shared-destination-write reasoning applies here per direction slot,
     * kept serial rather than folded into rri_qs_calc's parallel loop. */
    for (int k = 0; k < sc->count; k++) {
        int lmax = sc->lmax;
        int dif_p = sc->dif[k];
        for (int l = 0; l < lmax; l++) {
            if (dif_p == 0 && l == 1) break;
            int kk = (dif_p == 0) ? sc->down_1d[k] : sc->down[l][k];
            if (kk == -1) continue;
            fs_idx[kk] += qs_idx[l][k];
        }
    }
}
