/**
 * @file rri_rivslo.c
 * @brief River<->slope water exchange: at every river cell, moves water
 * between the collocated hillslope depth `hs` and river depth `hr` --
 * water falling off the slope into an under-full channel (a small
 * waterfall/step, when the river is below bank height), or overtopping
 * in either direction once one side's water surface rises above the
 * levee crest (`height`).
 *
 * Called once per outer timestep in main.c, between the slope/
 * groundwater updates and Green-Ampt infiltration -- unlike the RK45
 * kernels (rri_riv.c/rri_slope.c/rri_gw.c), this is NOT part of any
 * adaptive sub-stepping; it's a direct, instantaneous per-timestep
 * adjustment applied once, using a small fixed-iteration-count
 * (10-pass) local relaxation to settle `hs`/`hr` toward equal water
 * surface elevation when the weir-flow estimate alone would overshoot
 * (see the `for (int c = 0; c < 10; c++)` loops below).
 *
 * Walks the FULL (ny, nx) grid (not a cellset) because it needs BOTH a
 * cell's slope depth AND its collocated river depth together at every
 * river cell -- the natural representation for that is the raster form,
 * not either compressed cellset alone.
 *
 * Rectangular channel only: `hr_update_rect` below uses the closed-form
 * rectangular hr<->vr relations (rri_setup.c: rri_hr2vr/rri_vr2hr)
 * instead of the general cross-section lookup table, matching
 * RRI_Section.f90's `hr_update`/`sec_h2b` restricted to their
 * `sec_map_idx(k)<=0` branch -- custom cross-sections (sec_map>0) aren't
 * implemented in this port (see rri.h).
 *
 * Fortran reference: RRI_RivSlo.f90, `funcrs`. The four cases (a)-(d)
 * below are named and ordered exactly as in that source's comments, so
 * the two can be compared branch-by-branch.
 */
#include "rri/rri.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** @brief Apply a volume increment @p vr_inc [m^3] to a river cell
 * currently at depth @p hr_org, returning the resulting depth --
 * i.e. hr2vr, add, vr2hr, done as one step since every call site here
 * needs exactly that composition. Rectangular-channel only, see file doc. */
static double hr_update_rect(double hr_org, double vr_inc, double area, double area_ratio)
{
    double vr_org = rri_hr2vr(hr_org, area, area_ratio);
    return rri_vr2hr(vr_org + vr_inc, area, area_ratio);
}

/**
 * @brief Exchange water between slope and river depth at every river
 * cell for one outer timestep.
 * @param g   Grid (for domain/riv/depth/height/area).
 * @param rc  River cellset (for width/len_riv/area_ratio at each river cell).
 * @param dt  Outer timestep [s].
 * @param[in,out] hr  River water depth [m], full (ny,nx) grid.
 * @param[in,out] hs  Slope water depth [m], full (ny,nx) grid.
 */
void rri_funcrs(const rri_grid *g, const rri_riv_cellset *rc, double dt, double *hr, double *hs)
{
    /* Weir/orifice discharge coefficients: mu1 for the step-fall case
     * (a) (a submerged-weir-like formula with a fixed (2/3)^(3/2)
     * coefficient), mu2/mu3 for the two overtopping regimes (c)/(d) --
     * mu2 for free (unsubmerged) weir flow, mu3 for submerged weir flow,
     * selected by the h2/h1 <= 2/3 downstream-submergence ratio test in
     * each of those branches below. Standard empirical weir-flow
     * coefficients from open-channel hydraulics, not derived here. */
    const double mu1 = pow(2.0 / 3.0, 3.0 / 2.0);
    const double mu2 = 0.350, mu3 = 0.910;
    int ny = g->ny, nx = g->nx;

    int *riv_ij2idx = (int *)calloc((size_t)ny * nx, sizeof(int));
    for (int k = 0; k < rc->count; k++) riv_ij2idx[rc->idx2i[k] * nx + rc->idx2j[k]] = k;

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            size_t p = (size_t)i * nx + j;
            if (g->domain[p] == 0 || g->riv[p] == 0) continue;

            double hs_top = hs[p];
            double hr_top = hr[p] - g->depth[p];
            int k = riv_ij2idx[p];
            double length = rc->len_riv[k];
            double area_ratio = rc->area_ratio[k];
            double height = g->height[p];

            double hrs = 0.0;

            if ((height == 0.0 && hr_top < 0.0) || (height > 0.0 && hr_top < 0.0 && hs_top <= height)) {
                /* (a) slope -> river: step fall */
                hrs = mu1 * hs_top * sqrt(9.810 * hs_top) * dt * length * 2.0 / g->area;
                if (hrs > hs[p]) hrs = hs[p];
                hs[p] -= hrs;
                hr[p] = hr_update_rect(hr[p], hrs * g->area, g->area, area_ratio);

                hs_top = hs[p]; hr_top = hr[p] - g->depth[p];
                if (hr_top >= -0.000010 && hr_top > hs_top) {
                    for (int c = 0; c < 10; c++) {
                        double b = rc->width[k];
                        double ar = length * b / g->area;
                        double d = (hs_top - hr_top) / (1.0 + 1.0 / ar);
                        hs[p] -= d;
                        hr[p] = hr_update_rect(hr[p], d * g->area, g->area, area_ratio);
                        hrs += d;
                        if (fabs(hs[p] - (hr[p] - g->depth[p])) < 0.000010) break;
                        hs_top = hs[p]; hr_top = hr[p] - g->depth[p];
                    }
                    hr[p] = hs[p] + g->depth[p];
                }
            } else if (height > 0.0 && hs_top <= height && hr_top <= height && hr_top >= 0.0) {
                /* (b) no exchange */
            } else if (hs_top <= hr_top && hr_top >= height) {
                /* (c) river -> slope: overtopping */
                double h1 = hr_top - height, h2 = hs_top - height;
                hrs = (h2 / h1 <= 2.0 / 3.0)
                    ? -mu2 * h1 * sqrt(2.0 * 9.810 * h1) * dt * length * 2.0 / g->area
                    : -mu3 * h2 * sqrt(2.0 * 9.810 * (h1 - h2)) * dt * length * 2.0 / g->area;
                double b = rc->width[k];
                double ar = length * b / g->area;
                if (fabs(hrs / ar) > (hr_top - height)) hrs = -(hr_top - height) * ar;
                hs[p] -= hrs;
                hr[p] = hr_update_rect(hr[p], hrs * g->area, g->area, area_ratio);

                hs_top = hs[p]; hr_top = hr[p] - g->depth[p];
                if (hs_top > hr_top) {
                    for (int c = 0; c < 10; c++) {
                        b = rc->width[k];
                        ar = length * b / g->area;
                        double d = (hs_top - hr_top) / (1.0 + 1.0 / ar);
                        hs[p] -= d;
                        hr[p] = hr_update_rect(hr[p], d * g->area, g->area, area_ratio);
                        hrs += d;
                        if (fabs(hs[p] - (hr[p] - g->depth[p])) < 0.000010) break;
                        hs_top = hs[p]; hr_top = hr[p] - g->depth[p];
                    }
                    hr[p] = hs[p] + g->depth[p];
                }
            } else if (hs_top >= hr_top && hs_top >= height) {
                /* (d) slope -> river: overtopping */
                double h1 = hs_top - height, h2 = hr_top - height;
                hrs = (h2 / h1 <= 2.0 / 3.0)
                    ? mu2 * h1 * sqrt(2.0 * 9.810 * h1) * dt * length * 2.0 / g->area
                    : mu3 * h2 * sqrt(2.0 * 9.810 * (h1 - h2)) * dt * length * 2.0 / g->area;
                if (hrs > (hs_top - height)) hrs = hs[p] - height;
                hs[p] -= hrs;
                hr[p] = hr_update_rect(hr[p], hrs * g->area, g->area, area_ratio);

                hs_top = hs[p]; hr_top = hr[p] - g->depth[p];
                if (hr_top >= -0.000010 && hr_top > hs_top) {
                    for (int c = 0; c < 10; c++) {
                        double b = rc->width[k];
                        double ar = length * b / g->area;
                        double d = (hs_top - hr_top) / (1.0 + 1.0 / ar);
                        hs[p] -= d;
                        hr[p] = hr_update_rect(hr[p], d * g->area, g->area, area_ratio);
                        hrs += d;
                        if (fabs(hs[p] - (hr[p] - g->depth[p])) < 0.000010) break;
                        hs_top = hs[p]; hr_top = hr[p] - g->depth[p];
                    }
                    hr[p] = hs[p] + g->depth[p];
                }
            } else {
                fprintf(stderr, "rri_funcrs: unhandled case at (%d,%d)\n", i, j);
                exit(1);
            }
        }
    }
    free(riv_ij2idx);
}
