/**
 * @file rri_gw.c
 * @brief Shallow bedrock groundwater: lateral flow between cells
 * (rri_qg_calc/rri_funcg, the RK45-integrated half) plus three
 * non-RK, once-per-outer-timestep vertical exchange processes between
 * the aquifer and the surface/soil column: recharge (surface water
 * draining down into the aquifer), a constant background loss, and
 * exfiltration (aquifer water rising back to the surface once the
 * deficit goes negative, i.e. the water table reaches the surface).
 *
 * `hg` is a DEFICIT below the ground surface [m], not a depth above it
 * like `hs`/`hr` -- larger hg means a lower water table / less stored
 * groundwater. See kernels.h: rri_k_hg_calc's doc, and
 * rri_setup.c: rri_storage_calc's `sg -= hg * gammag * area` for where
 * that sign convention gets converted into an actual storage value.
 *
 * Only exercised when rri_config::gw_switch is set (any landuse has
 * ksg>0) -- main.c skips the entire groundwater RK45 sub-loop and the
 * recharge/lose/exfilt calls otherwise. The validated solo30s config
 * DOES enable groundwater (gammag=0.4, kg0=0.0005, fpg=0.03), so this
 * file is exercised by, not incidental to, the full-run Fortran
 * validation described in README.md.
 *
 * Fortran reference: RRI_GW.f90 (`funcg`, `qg_calc`, `hg_calc`,
 * `gw_recharge`, `gw_lose`, `gw_exfilt` -- `hg_calc`'s formula itself
 * lives in kernels.h as `rri_k_hg_calc`, shared with the future OpenCL
 * backend).
 */
#include "rri/rri.h"
#include "rri/kernels.h"
#include <math.h>

/**
 * @brief Per-direction lateral groundwater discharge, independent
 * across (l, k) given the previous step's `hg_idx` -- OpenMP/OpenCL
 * parallel target, same structure as rri_slope.c: rri_qs_calc.
 *
 * A cell (or its neighbor) with ksg<=0 -- no bedrock aquifer for that
 * landuse -- contributes/receives exactly 0 lateral flow; groundwater
 * is only meaningful where the landuse config actually enables it.
 *
 * @param sc      Hillslope cellset (groundwater shares the same
 *                topology/neighbor structure as hillslope routing).
 * @param hg_idx  Groundwater deficit per cell [m] (see file-level comment
 *                on sign convention), length sc->count.
 * @param area    Grid cell area [m^2].
 * @param[out] qg_idx  Discharge per unit area [m/s], same [l][k] layout
 *                     and sign convention as rri_slope.c's qs_idx.
 */
void rri_qg_calc(const rri_slo_cellset *sc, const double *hg_idx, double area, double *qg_idx[RRI_LMAX8])
{
#pragma omp parallel for
    for (int k = 0; k < sc->count; k++) {
        for (int l = 0; l < RRI_LMAX8; l++) qg_idx[l][k] = 0.0;
        if (sc->ksg[k] <= 0.0) continue;

        double zb_p = sc->zb[k], hg_p = hg_idx[k];
        double gammag_p = sc->gammag[k], kg0_p = sc->kg0[k], fpg_p = sc->fpg[k];
        int dif_p = sc->dif[k];
        int lmax = sc->lmax;

        for (int l = 0; l < lmax; l++) {
            if (dif_p == 0 && l == 1) break;
            int kk = (dif_p == 0) ? sc->down_1d[k] : sc->down[l][k];
            if (kk == -1) continue;
            if (sc->ksg[kk] <= 0.0) continue;
            double distance = (dif_p == 0) ? sc->dis_1d[k] : sc->dis[l][k];
            double length = (dif_p == 0) ? sc->len_1d[k] : sc->len[l][k];

            double zb_n = sc->zb[kk], hg_n = hg_idx[kk];
            double dh = ((zb_p - hg_p) - (zb_n - hg_n)) / distance;
            if (dif_p == 0) { double t = (zb_p - zb_n) / distance; dh = t > 0.001 ? t : 0.001; }

            if (dh >= 0.0) {
                qg_idx[l][k] = rri_k_hg_calc(gammag_p, kg0_p, fpg_p, hg_p, dh, length, area);
            } else {
                double gammag_n = sc->gammag[kk], kg0_n = sc->kg0[kk], fpg_n = sc->fpg[kk];
                qg_idx[l][k] = -rri_k_hg_calc(gammag_n, kg0_n, fpg_n, hg_n, fabs(dh), length, area);
            }
        }
    }
}

/**
 * @brief One RK45 stage's derivative evaluation for the groundwater
 * deficit state variable: net lateral flow converted from a volumetric
 * rate to a deficit-depth rate via `gammag` (specific yield).
 *
 * Fortran reference: RRI_GW.f90, `funcg`.
 * @param[out] fg_idx  Deficit derivative per cell [m/s]. Both a cell's own
 *                     outflow AND the inflow it receives from a neighbor
 *                     are divided by the SOURCE cell's `gammag` (see the
 *                     scatter loop below: `fg_idx[kk] -= qg_idx[l][k] /
 *                     sc->gammag[k]`, using `gammag[k]` not `gammag[kk]`)
 *                     -- confirmed against RRI_GW.f90 `funcg`
 *                     (`fg_idx(kk) = fg_idx(kk) - qg_idx(l,k) /
 *                     gammag_idx(k)`): this asymmetry is verbatim
 *                     Fortran behavior, not a bug introduced in this port.
 */
void rri_funcg(const rri_slo_cellset *sc, const double *hg_idx, double area,
                double *fg_idx, double *qg_idx[RRI_LMAX8])
{
    rri_qg_calc(sc, hg_idx, area, qg_idx);

    for (int k = 0; k < sc->count; k++) {
        fg_idx[k] = 0.0;
        if (sc->gammag[k] > 0.0) {
            double s = 0.0;
            for (int l = 0; l < RRI_LMAX8; l++) s += qg_idx[l][k];
            fg_idx[k] = s / sc->gammag[k];
        }
    }
    for (int k = 0; k < sc->count; k++) {
        if (sc->gammag[k] <= 0.0) continue;
        int lmax = sc->lmax;
        int dif_p = sc->dif[k];
        for (int l = 0; l < lmax; l++) {
            if (dif_p == 0 && l == 1) break;
            int kk = (dif_p == 0) ? sc->down_1d[k] : sc->down[l][k];
            if (kk == -1) continue;
            fg_idx[kk] -= qg_idx[l][k] / sc->gammag[k];
        }
    }
}

/**
 * @brief Vertical recharge: surface water (and, once Green-Ampt
 * infiltration storage `gampt_ff` is exhausted, the infiltration
 * front itself) drains down into the aquifer at rate `ksg`, reducing
 * the groundwater deficit `hg`. Called once per outer timestep (not
 * inside the RK45 sub-loop) in main.c, before the groundwater RK45
 * integration -- recharge changes the STARTING deficit for that
 * timestep's lateral-flow integration, it isn't itself integrated.
 *
 * Fortran reference: RRI_GW.f90, `gw_recharge`.
 * @param dt  Outer timestep [s] (rri_config::dt), not an RK sub-step.
 * @param[in,out] hs_idx        Slope water depth [m]; drawn down by
 *                              whatever recharge isn't satisfied from
 *                              gampt_ff_idx.
 * @param[in,out] gampt_ff_idx  Cumulative Green-Ampt infiltration [m];
 *                              drawn down first, before hs_idx.
 * @param[in,out] hg_idx        Groundwater deficit [m]; decreases
 *                              (recharge fills the aquifer) unless
 *                              already at or below 0 (aquifer already
 *                              full/overflowing -- see rri_gw_exfilt).
 */
void rri_gw_recharge(const rri_slo_cellset *sc, double dt, double *hs_idx,
                      double *gampt_ff_idx, double *hg_idx)
{
    for (int k = 0; k < sc->count; k++) {
        if (hg_idx[k] < 0.0) continue;
        double rech = sc->ksg[k] * dt;
        if (hg_idx[k] * sc->gammag[k] < rech) rech = hg_idx[k] * sc->gammag[k];

        double rech_ga = 0.0, rech_hs = 0.0;
        if (rech < gampt_ff_idx[k] && sc->ksv[k] > 0.0) {
            rech_ga = rech;
        } else {
            if (sc->ksv[k] > 0.0) rech_ga = gampt_ff_idx[k];
            rech_hs = (rech - rech_ga < hs_idx[k]) ? (rech - rech_ga) : hs_idx[k];
        }
        hs_idx[k] -= rech_hs;
        if (sc->ksv[k] > 0.0) gampt_ff_idx[k] -= rech_ga;
        double rech_tot = rech_ga + rech_hs;
        if (sc->gammag[k] > 0.0) hg_idx[k] -= rech_tot / sc->gammag[k];
    }
}

/** @brief Constant background groundwater loss (e.g. deep percolation
 * out of the modeled aquifer entirely) at per-landuse rate `rgl`,
 * increasing the deficit `hg`. Called alongside rri_gw_recharge, once
 * per outer timestep. Fortran reference: RRI_GW.f90, `gw_lose`. */
void rri_gw_lose(const rri_slo_cellset *sc, double dt, double *hg_idx)
{
    for (int k = 0; k < sc->count; k++)
        if (sc->rgl[k] > 0.0) hg_idx[k] += sc->rgl[k] / sc->gammag[k] * dt;
}

/**
 * @brief Exfiltration: once a cell's groundwater deficit goes negative
 * (the water table has risen above the ground surface -- i.e. the
 * aquifer is over-full, `hg_idx[k] < 0`), that excess water returns to
 * the surface, first refilling Green-Ampt infiltration storage up to its
 * cap and only then adding to the surface water depth. Called once per
 * outer timestep in main.c, AFTER the groundwater RK45 integration
 * (the lateral-flow sub-loop can itself drive a cell's deficit negative;
 * this is the cleanup step that pushes that excess back to the surface
 * before the next outer timestep starts).
 *
 * Fortran reference: RRI_GW.f90, `gw_exfilt`.
 * @param[in,out] hg_idx  Reset to exactly 0 for every cell this touches
 *                        (the exfiltrated water is fully accounted for
 *                        elsewhere, so the deficit doesn't stay negative).
 */
void rri_gw_exfilt(const rri_slo_cellset *sc, double dt, double *hs_idx,
                    double *gampt_ff_idx, double *hg_idx)
{
    for (int k = 0; k < sc->count; k++) {
        if (hg_idx[k] >= 0.0) continue;
        double exfilt = -hg_idx[k] * sc->gammag[k];
        double exfilt_ga = 0.0, exfilt_hs;
        if (sc->infilt_limit[k] > gampt_ff_idx[k] &&
            sc->infilt_limit[k] - gampt_ff_idx[k] >= exfilt &&
            sc->ksv[k] > 0.0 && sc->infilt_limit[k] > 0.0) {
            exfilt_ga = exfilt;
            exfilt_hs = 0.0;
        } else {
            if (sc->ksv[k] > 0.0 && sc->infilt_limit[k] > 0.0) exfilt_ga = sc->infilt_limit[k] - gampt_ff_idx[k];
            exfilt_hs = exfilt - exfilt_ga;
        }
        hg_idx[k] = 0.0;
        if (sc->ksg[k] > 0.0 && sc->infilt_limit[k] > 0.0) gampt_ff_idx[k] += exfilt_ga;
        hs_idx[k] += exfilt_hs;
    }
}
