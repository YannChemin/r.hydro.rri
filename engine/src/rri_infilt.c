/**
 * @file rri_infilt.c
 * @brief Green-Ampt infiltration: surface water infiltrating into the
 * soil column at a rate that decays as the cumulative infiltrated depth
 * `gampt_ff` grows (the classic Green-Ampt time-decaying infiltration
 * capacity curve, driven by the wetting-front suction head `faif`).
 *
 * Called once per outer timestep in main.c (not inside any RK45
 * sub-loop -- infiltration is treated as an instantaneous per-timestep
 * adjustment to `hs`, applied after river<->slope exchange), and this
 * is a per-cell-independent computation given the previous step's state,
 * so it's OpenMP-parallel like the RK45 kernels even though it isn't one.
 *
 * Fortran reference: RRI_Infilt.f90, `infilt`.
 */
#include "rri/rri.h"

/**
 * @param sc     Hillslope cellset.
 * @param dt     Outer timestep [s].
 * @param[in,out] hs_idx        Slope water depth [m]; drawn down by the
 *                              infiltrated amount, floored at 0.
 * @param[in,out] gampt_ff_idx  Cumulative infiltrated depth [m] (the
 *                              Green-Ampt state variable driving the
 *                              capacity curve below); floored at 0.01 m
 *                              ONLY for the purposes of the capacity
 *                              formula's denominator (avoiding a
 *                              division blowing up at the start of a
 *                              dry cell's infiltration) -- the actual
 *                              stored value is not floored, only the
 *                              local `gampt_ff_temp` used in the rate
 *                              calculation.
 * @param[out] gampt_f_idx      Instantaneous infiltration RATE [m/s]
 *                              this step (distinct from gampt_ff, the
 *                              cumulative DEPTH) -- exposed separately
 *                              since some output/diagnostic paths in the
 *                              Fortran reference want the rate, not the
 *                              running total; unused elsewhere in this
 *                              port but kept for API parity.
 */
void rri_infilt(const rri_slo_cellset *sc, double dt, double *hs_idx,
                 double *gampt_ff_idx, double *gampt_f_idx)
{
#pragma omp parallel for
    for (int k = 0; k < sc->count; k++) {
        double gampt_ff_temp = gampt_ff_idx[k];
        if (gampt_ff_temp <= 0.01) gampt_ff_temp = 0.01;

        /* Green-Ampt capacity: ksv * (1 + faif*gammaa / cumulative_depth) --
         * capacity is highest right at the start (small cumulative depth,
         * large suction-head term) and decays toward ksv as the wetting
         * front advances. */
        double f = sc->ksv[k] * (1.0 + sc->faif[k] * sc->gammaa[k] / gampt_ff_temp);
        if (f >= hs_idx[k] / dt) f = hs_idx[k] / dt; /* can't infiltrate more than the standing water available */
        if (sc->infilt_limit[k] >= 0.0 && gampt_ff_idx[k] >= sc->infilt_limit[k]) f = 0.0; /* soil column saturated: no more capacity */

        gampt_f_idx[k] = f;
        gampt_ff_idx[k] += f * dt;
        hs_idx[k] -= f * dt;
        if (hs_idx[k] <= 0.0) hs_idx[k] = 0.0;
    }
}
