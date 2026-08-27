/**
 * @file rri_setup.c
 * @brief One-time domain setup: builds the sparse river/hillslope
 * cellsets from the full-raster grid, converts between the two
 * representations, and computes total water storage for the
 * mass-balance check.
 *
 * This is where the "why sparse index arrays" design described in
 * rri.h's file-level comment is actually constructed: each of
 * rri_riv_idx_setting/rri_slo_idx_setting walks the full (ny,nx) grid
 * ONCE, in row-major order, and for every active cell precomputes its
 * neighbor lookups (down/dis/len) so the time-stepping loop (main.c) and
 * the physics kernels (rri_riv.c/rri_slope.c/rri_gw.c) never need to
 * re-derive "what's adjacent to cell k" from (i,j) arithmetic again.
 * None of this runs per-timestep -- it's pure setup cost, paid once.
 *
 * Fortran reference: RRI_Sub.f90 (`riv_idx_setting`, `slo_idx_setting`,
 * `sub_riv_ij2idx`/`idx2ij`, `sub_slo_ij2idx`/`idx2ij`, `storage_calc`).
 */
#include "rri/rri.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static double *dalloc(size_t n) { return (double *)calloc(n ? n : 1, sizeof(double)); }
static int *ialloc(size_t n) { return (int *)calloc(n ? n : 1, sizeof(int)); }

int rri_riv_idx_setting(rri_grid *g, rri_riv_cellset *rc)
{
    memset(rc, 0, sizeof(*rc));
    int ny = g->ny, nx = g->nx;

    int count = 0;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            if (g->domain[i * nx + j] != 0 && g->riv[i * nx + j] == 1) count++;
    rc->count = count;
    if (count == 0) return 0;

    rc->idx2i = ialloc(count); rc->idx2j = ialloc(count);
    rc->down = ialloc(count);
    rc->dis = dalloc(count);
    rc->zb = dalloc(count);
    rc->domain = ialloc(count);
    rc->width = dalloc(count); rc->depth = dalloc(count); rc->height = dalloc(count);
    rc->area_ratio = dalloc(count); rc->len_riv = dalloc(count);
    rc->dif = dalloc(count);

    int *riv_ij2idx = ialloc((size_t)ny * nx);

    int k = 0;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if (g->domain[i * nx + j] == 0 || g->riv[i * nx + j] != 1) continue;
            rc->idx2i[k] = i; rc->idx2j[k] = j;
            rc->domain[k] = g->domain[i * nx + j]; /* captured before the neighbor-search pass below may mutate domain[] to 2 -- matches RRI_Sub.f90/RRIpy exactly (domain_riv_idx can go stale relative to grid.domain for a cell discovered as an outlet only during that pass) */
            rc->width[k] = g->width[i * nx + j];
            rc->depth[k] = g->depth[i * nx + j];
            rc->height[k] = g->height[i * nx + j];
            rc->area_ratio[k] = g->area_ratio[i * nx + j];
            rc->zb[k] = g->zb_riv[i * nx + j]; /* river bed elevation, NOT the slope's zb -- see rri.h */
            rc->len_riv[k] = g->len_riv[i * nx + j];
            riv_ij2idx[i * nx + j] = k;
            k++;
        }
    }

    /* Second pass: resolve each river cell's single downstream neighbor
     * from its D8 flow direction code, discovering outlet cells (where
     * the resolved neighbor is out of bounds or itself domain==0) along
     * the way. This MUST be a separate pass from the one above -- it
     * needs riv_ij2idx fully populated first to translate a resolved
     * (ii,jj) neighbor location into a cellset index. */
    k = 0;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if (g->domain[i * nx + j] == 0 || g->riv[i * nx + j] != 1) continue;
            int ii = i, jj = j;
            double distance = 0.0;
            int d = g->dir[i * nx + j];
            switch (d) { /* D8 direction codes -- see rri.h: rri_grid::dir's doc */
                case 1:   ii = i;     jj = j + 1; distance = g->dx; break;
                case 2:   ii = i + 1; jj = j + 1; distance = sqrt(g->dx * g->dx + g->dy * g->dy); break;
                case 4:   ii = i + 1; jj = j;     distance = g->dy; break;
                case 8:   ii = i + 1; jj = j - 1; distance = sqrt(g->dx * g->dx + g->dy * g->dy); break;
                case 16:  ii = i;     jj = j - 1; distance = g->dx; break;
                case 32:  ii = i - 1; jj = j - 1; distance = sqrt(g->dx * g->dx + g->dy * g->dy); break;
                case 64:  ii = i - 1; jj = j;     distance = g->dy; break;
                case 128: ii = i - 1; jj = j + 1; distance = sqrt(g->dx * g->dx + g->dy * g->dy); break;
                case 0: case -1: ii = i; jj = j; break;
                default:
                    fprintf(stderr, "rri_riv_idx_setting: dir[%d,%d]=%d error\n", i, j, d);
                    free(riv_ij2idx);
                    return -1;
            }
            /* Neighbor off the raster, or in a nodata cell: this cell is
             * actually an outlet, even though its own dir code didn't say
             * so. Redirect it to itself (self-loop, like a real outlet)
             * and correct grid::domain/dir in place so later readers see
             * it as domain==2, not the stale domain==1 it had before this
             * was discovered. */
            if (ii < 0 || ii > ny - 1 || jj < 0 || jj > nx - 1) {
                g->domain[i * nx + j] = 2; g->dir[i * nx + j] = 0; ii = i; jj = j;
            }
            if (g->domain[ii * nx + jj] == 0) {
                g->domain[i * nx + j] = 2; g->dir[i * nx + j] = 0; ii = i; jj = j;
            }
            if (g->riv[ii * nx + jj] == 0) {
                fprintf(stderr, "rri_riv_idx_setting: riv[%d,%d] should be 1 (from [%d,%d])\n", ii, jj, i, j);
                free(riv_ij2idx);
                return -1;
            }
            rc->dis[k] = distance;
            rc->down[k] = riv_ij2idx[ii * nx + jj];
            k++;
        }
    }

    free(riv_ij2idx);
    return 0;
}

int rri_slo_idx_setting(rri_grid *g, const rri_landuse *lu, int eight_dir, rri_slo_cellset *sc)
{
    memset(sc, 0, sizeof(*sc));
    int ny = g->ny, nx = g->nx;

    int count = 0;
    for (int i = 0; i < ny; i++)
        for (int j = 0; j < nx; j++)
            if (g->domain[i * nx + j] != 0) count++;
    sc->count = count;
    if (count == 0) return 0;

    sc->idx2i = ialloc(count); sc->idx2j = ialloc(count);
    sc->domain = ialloc(count);
    sc->zb = dalloc(count);
    for (int l = 0; l < RRI_LMAX8; l++) {
        sc->down[l] = ialloc(count);
        sc->dis[l] = dalloc(count);
        sc->len[l] = dalloc(count);
    }
    sc->down_1d = ialloc(count);
    sc->dis_1d = dalloc(count);
    sc->len_1d = dalloc(count);

    sc->ns_slope = dalloc(count); sc->soildepth = dalloc(count); sc->gammaa = dalloc(count);
    sc->ksv = dalloc(count); sc->faif = dalloc(count); sc->infilt_limit = dalloc(count);
    sc->ka = dalloc(count); sc->gammam = dalloc(count); sc->beta = dalloc(count);
    sc->da = dalloc(count); sc->dm = dalloc(count);
    sc->ksg = dalloc(count); sc->gammag = dalloc(count); sc->kg0 = dalloc(count);
    sc->fpg = dalloc(count); sc->rgl = dalloc(count);
    sc->dif = ialloc(count);

    int *slo_ij2idx = ialloc((size_t)ny * nx);

    int k = 0;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if (g->domain[i * nx + j] == 0) continue;
            sc->idx2i[k] = i; sc->idx2j[k] = j;
            sc->domain[k] = g->domain[i * nx + j];
            sc->zb[k] = g->zb[i * nx + j];
            slo_ij2idx[i * nx + j] = k;
            int land = g->land[i * nx + j];
            int li = land - 1; /* land is 1-based */
            if (li < 0) li = 0;
            if (li >= lu->n) li = lu->n - 1;
            sc->dif[k] = lu->dif[li];
            sc->ns_slope[k] = lu->ns_slope[li];
            sc->soildepth[k] = lu->soildepth[li];
            sc->gammaa[k] = lu->gammaa[li];
            sc->ksv[k] = lu->ksv[li];
            sc->faif[k] = lu->faif[li];
            sc->infilt_limit[k] = lu->infilt_limit[li];
            sc->ka[k] = lu->ka[li];
            sc->gammam[k] = lu->gammam[li];
            sc->beta[k] = lu->beta[li];
            sc->da[k] = lu->da[li];
            sc->dm[k] = lu->dm[li];
            sc->ksg[k] = lu->ksg[li];
            sc->gammag[k] = lu->gammag[li];
            sc->kg0[k] = lu->kg0[li];
            sc->fpg[k] = lu->fpg[li];
            sc->rgl[k] = lu->rgl[li];
            k++;
        }
    }

    double l1, l2, l3;
    if (eight_dir == 1) {
        sc->lmax = 4;
        l1 = g->dy / 2.0; l2 = g->dx / 2.0; l3 = sqrt(g->dx * g->dx + g->dy * g->dy) / 4.0;
    } else if (eight_dir == 0) {
        sc->lmax = 2;
        l1 = g->dy; l2 = g->dx; l3 = 0.0;
    } else {
        fprintf(stderr, "rri_slo_idx_setting: eight_dir must be 0 or 1\n");
        free(slo_ij2idx);
        return -1;
    }

    for (int l = 0; l < RRI_LMAX8; l++)
        for (int kk = 0; kk < count; kk++) sc->down[l][kk] = -1;

    k = 0;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if (g->domain[i * nx + j] == 0) continue;
            for (int l = 0; l < sc->lmax; l++) {
                int ii, jj; double distance, length;
                switch (l) {
                    case 0: ii = i;     jj = j + 1; distance = g->dx; length = l1; break;
                    case 1: ii = i + 1; jj = j;     distance = g->dy; length = l2; break;
                    case 2: ii = i + 1; jj = j + 1; distance = sqrt(g->dx * g->dx + g->dy * g->dy); length = l3; break;
                    default: ii = i + 1; jj = j - 1; distance = sqrt(g->dx * g->dx + g->dy * g->dy); length = l3; break;
                }
                if (ii >= ny || jj >= nx || ii < 0 || jj < 0) continue;
                if (g->domain[ii * nx + jj] == 0) continue;
                sc->down[l][k] = slo_ij2idx[ii * nx + jj];
                sc->dis[l][k] = distance;
                sc->len[l][k] = length;
            }
            k++;
        }
    }

    /* Every cell also gets a SEPARATE single-direction (kinematic)
     * neighbor, following dir[] exactly like the river network does,
     * independent of the 8-direction lookup above -- a cell using
     * diffusive routing (dif==1) never reads down_1d, and a cell using
     * kinematic routing (dif==0) never reads down[]/dis[]/len[]; see
     * rri_qs_calc/rri_qg_calc's `dif_p == 0` branches. Both are always
     * populated regardless of any given cell's own dif flag, since
     * dif is a per-landuse choice that can vary cell-to-cell within one
     * domain. */
    double l1_kin = g->dy, l2_kin = g->dx;
    double l3_kin = g->dx * g->dy / sqrt(g->dx * g->dx + g->dy * g->dy);
    for (int kk = 0; kk < count; kk++) { sc->down_1d[kk] = -1; sc->dis_1d[kk] = l1; sc->len_1d[kk] = g->dx; }

    k = 0;
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if (g->domain[i * nx + j] == 0) continue;
            int ii = i, jj = j; double distance = g->dx, length = l1_kin;
            int d = g->dir[i * nx + j];
            switch (d) {
                case 1:   ii = i;     jj = j + 1; distance = g->dx; length = l1_kin; break;
                case 2:   ii = i + 1; jj = j + 1; distance = sqrt(g->dx * g->dx + g->dy * g->dy); length = l3_kin; break;
                case 4:   ii = i + 1; jj = j;     distance = g->dy; length = l2_kin; break;
                case 8:   ii = i + 1; jj = j - 1; distance = sqrt(g->dx * g->dx + g->dy * g->dy); length = l3_kin; break;
                case 16:  ii = i;     jj = j - 1; distance = g->dx; length = l1_kin; break;
                case 32:  ii = i - 1; jj = j - 1; distance = sqrt(g->dx * g->dx + g->dy * g->dy); length = l3_kin; break;
                case 64:  ii = i - 1; jj = j;     distance = g->dy; length = l2_kin; break;
                case 128: ii = i - 1; jj = j + 1; distance = sqrt(g->dx * g->dx + g->dy * g->dy); length = l3_kin; break;
                case 0: case -1: ii = i; jj = j; distance = g->dx; length = l1_kin; break;
                default:
                    fprintf(stderr, "rri_slo_idx_setting: dir[%d,%d]=%d error\n", i, j, d);
                    free(slo_ij2idx);
                    return -1;
            }
            if (ii >= ny || jj >= nx || ii < 0 || jj < 0 || g->domain[ii * nx + jj] == 0) { k++; continue; }
            sc->down_1d[k] = slo_ij2idx[ii * nx + jj];
            sc->dis_1d[k] = distance;
            sc->len_1d[k] = length;
            k++;
        }
    }

    free(slo_ij2idx);
    return 0;
}

void rri_riv_cellset_free(rri_riv_cellset *rc)
{
    free(rc->idx2i); free(rc->idx2j); free(rc->down); free(rc->dis); free(rc->zb);
    free(rc->domain); free(rc->width); free(rc->depth); free(rc->height);
    free(rc->area_ratio); free(rc->len_riv); free(rc->dif);
    memset(rc, 0, sizeof(*rc));
}

void rri_slo_cellset_free(rri_slo_cellset *sc)
{
    free(sc->idx2i); free(sc->idx2j); free(sc->domain); free(sc->zb);
    for (int l = 0; l < RRI_LMAX8; l++) { free(sc->down[l]); free(sc->dis[l]); free(sc->len[l]); }
    free(sc->down_1d); free(sc->dis_1d); free(sc->len_1d);
    free(sc->ns_slope); free(sc->soildepth); free(sc->gammaa);
    free(sc->ksv); free(sc->faif); free(sc->infilt_limit);
    free(sc->ka); free(sc->gammam); free(sc->beta); free(sc->da); free(sc->dm);
    free(sc->ksg); free(sc->gammag); free(sc->kg0); free(sc->fpg); free(sc->rgl);
    free(sc->dif);
    memset(sc, 0, sizeof(*sc));
}

void rri_riv_ij2idx(const rri_riv_cellset *rc, const double *grid, int nx, double *idx_out)
{
    for (int k = 0; k < rc->count; k++) idx_out[k] = grid[(size_t)rc->idx2i[k] * nx + rc->idx2j[k]];
}

void rri_riv_idx2ij(const rri_riv_cellset *rc, const double *idx, int ny, int nx, double *grid_out)
{
    memset(grid_out, 0, (size_t)ny * nx * sizeof(double));
    for (int k = 0; k < rc->count; k++) grid_out[(size_t)rc->idx2i[k] * nx + rc->idx2j[k]] = idx[k];
}

void rri_slo_ij2idx(const rri_slo_cellset *sc, const double *grid, int nx, double *idx_out)
{
    for (int k = 0; k < sc->count; k++) idx_out[k] = grid[(size_t)sc->idx2i[k] * nx + sc->idx2j[k]];
}

void rri_slo_idx2ij(const rri_slo_cellset *sc, const double *idx, int ny, int nx, double *grid_out)
{
    memset(grid_out, 0, (size_t)ny * nx * sizeof(double));
    for (int k = 0; k < sc->count; k++) grid_out[(size_t)sc->idx2i[k] * nx + sc->idx2j[k]] = idx[k];
}

/* Rectangular-channel cross section only (sec_map_idx(k)<=0 branch of
 * RRI_Section.f90: hr2vr/vr2hr -- custom cross-sections not implemented). */
double rri_hr2vr(double hr, double area, double area_ratio) { return hr * area * area_ratio; }
double rri_vr2hr(double vr, double area, double area_ratio) { return vr / (area * area_ratio); }

rri_storage rri_storage_calc(const rri_grid *g, const double *hs, const double *hr,
                              const double *hg, const double *gampt_ff,
                              const rri_slo_cellset *sc, const rri_riv_cellset *rc,
                              int riv_thresh)
{
    rri_storage s = {0.0, 0.0, 0.0, 0.0};
    int ny = g->ny, nx = g->nx;
    int *slo_ij2idx = ialloc((size_t)ny * nx);
    for (int k = 0; k < sc->count; k++) slo_ij2idx[sc->idx2i[k] * nx + sc->idx2j[k]] = k;
    int *riv_ij2idx = NULL;
    if (rc->count > 0) {
        riv_ij2idx = ialloc((size_t)ny * nx);
        for (int k = 0; k < rc->count; k++) riv_ij2idx[rc->idx2i[k] * nx + rc->idx2j[k]] = k;
    }

    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) {
            if (g->domain[i * nx + j] == 0) continue;
            s.ss += hs[i * nx + j] * g->area;
            if (riv_thresh >= 0 && g->riv[i * nx + j] == 1) {
                int k = riv_ij2idx[i * nx + j];
                s.sr += rri_hr2vr(hr[i * nx + j], g->area, rc->area_ratio[k]);
            }
            s.si += gampt_ff[i * nx + j] * g->area;
            int sk = slo_ij2idx[i * nx + j];
            s.sg -= hg[i * nx + j] * sc->gammag[sk] * g->area;
        }
    }
    free(slo_ij2idx);
    free(riv_ij2idx);
    return s;
}
