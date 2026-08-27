/****************************************************************************
 *
 * MODULE:       r.hydro.rri
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Native GRASS driver for the RRI (Rainfall-Runoff-
 *               Inundation) distributed hydrology model. This first
 *               increment (per NATIVE_GRASS_PLAN.md, section 5's
 *               validation order) covers ONLY static-input reading and
 *               compressed-cellset index setting -- reading elevation,
 *               flow direction, and flow accumulation directly from
 *               GRASS rasters (no ASCII intermediate files anywhere),
 *               deriving the same parametric river geometry
 *               (width/depth/zb_riv) the vendored engine's own
 *               ASCII-driven main.c computes, and calling straight into
 *               the unchanged rri_riv_idx_setting/rri_slo_idx_setting
 *               physics-core functions. Forcing (rain/PET), the RK45
 *               time loop, and output are NOT YET wired up here -- see
 *               NATIVE_GRASS_PLAN.md for the remaining stages, added in
 *               that order, each validated before the next begins.
 *
 * COPYRIGHT:    (C) 2026 by Yann Chemin
 *
 *               This is free and unencumbered software released into the
 *               public domain. See the file LICENSE (Unlicense) that
 *               comes with this source distribution for details.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "rri/rri.h"

/* Single-landuse-class defaults, matching the values the vendored
 * engine's own ASCII-driven main.c/RRI_Input.py used for the validated
 * solo30s config -- exposed here as G_parser options rather than a
 * generated RRI_Input.txt. Multi-landuse (per-category) parameterization
 * is NOT implemented in this increment -- see README "Known gaps"; land[]
 * is hardcoded to 1 everywhere, same simplification the vendored engine's
 * own main.c already had. */
static void set_single_landuse_defaults(rri_landuse *lu, double ns_slope,
                                         double soildepth, double gammaa,
                                         double ksv, double faif, double ka,
                                         double gammam, double beta)
{
    lu->n = 1;
    lu->dif = G_malloc(sizeof(int));
    lu->ns_slope = G_malloc(sizeof(double));
    lu->soildepth = G_malloc(sizeof(double));
    lu->gammaa = G_malloc(sizeof(double));
    lu->ksv = G_malloc(sizeof(double));
    lu->faif = G_malloc(sizeof(double));
    lu->ka = G_malloc(sizeof(double));
    lu->gammam = G_malloc(sizeof(double));
    lu->beta = G_malloc(sizeof(double));
    lu->ksg = G_malloc(sizeof(double));
    lu->gammag = G_malloc(sizeof(double));
    lu->kg0 = G_malloc(sizeof(double));
    lu->fpg = G_malloc(sizeof(double));
    lu->rgl = G_malloc(sizeof(double));
    lu->da = G_malloc(sizeof(double));
    lu->dm = G_malloc(sizeof(double));
    lu->infilt_limit = G_malloc(sizeof(double));

    lu->dif[0] = 1;
    lu->ns_slope[0] = ns_slope;
    lu->soildepth[0] = soildepth;
    lu->gammaa[0] = gammaa;
    lu->ksv[0] = ksv;
    lu->faif[0] = faif;
    lu->ka[0] = ka;
    lu->gammam[0] = gammam;
    lu->beta[0] = beta;
    lu->ksg[0] = 0.0;
    lu->gammag[0] = 0.0;
    lu->kg0[0] = 0.0;
    lu->fpg[0] = 0.0;
    lu->rgl[0] = 0.0;
    lu->da[0] = (soildepth > 0.0 && ka > 0.0) ? soildepth * gammaa : 0.0;
    lu->dm[0] = (soildepth > 0.0 && ka > 0.0 && gammam > 0.0) ? soildepth * gammam : 0.0;
    lu->infilt_limit[0] = (soildepth > 0.0 && ksv > 0.0) ? soildepth * gammaa : 0.0;
}

/* r.watershed's (and r.watershed.opencl's, for drop-in compatibility)
 * 'drainage' output numbers 8 directions counter-clockwise from
 * 1=north-east (r.watershed.md); RRI's own grid::dir instead uses an
 * 8-bit D8 bitmask, E=1 doubling clockwise (RRI_Break.f90's comment
 * block; see include/rri/rri.h's grid::dir doc). A cell whose flow
 * leaves the region gets a NEGATIVE drainage value from r.watershed --
 * such cells become RRI outlets (dir=0). This table and the outlet
 * handling are ported as-is from the (now superseded) Python driver's
 * reclass_direction_to_rri, which was smoke-tested but never
 * cross-validated against a real watershed with independently-known
 * flow paths -- same caveat applies here. */
static int drainage_to_rri_dir(double drainage)
{
    if (Rast_is_d_null_value(&drainage) || drainage < 0) return 0;
    switch ((int)drainage) {
        case 1: return 128; /* NE */
        case 2: return 64;  /* N */
        case 3: return 32;  /* NW */
        case 4: return 16;  /* W */
        case 5: return 8;   /* SW */
        case 6: return 4;   /* S */
        case 7: return 2;   /* SE */
        case 8: return 1;   /* E */
        default: return 0;
    }
}

int main(int argc, char *argv[])
{
    struct GModule *module;
    struct {
        struct Option *elevation, *drainage, *accumulation, *rain;
        struct Option *riv_thresh, *width_c, *width_s, *depth_c, *depth_s,
            *height_param, *height_limit;
        struct Option *ns_river, *ns_slope, *soildepth, *gammaa, *ksv,
            *faif, *ka, *gammam, *beta;
        struct Option *utm;
    } opt;
    struct Flag *eight_dir_flag;

    G_gisinit(argv[0]);

    module = G_define_module();
    G_add_keyword(_("raster"));
    G_add_keyword(_("hydrology"));
    G_add_keyword(_("temporal"));
    G_add_keyword(_("RRI"));
    module->description =
        _("Native GRASS driver for the RRI distributed rainfall-runoff-"
          "inundation model (static input + index setting only in this "
          "increment -- see NATIVE_GRASS_PLAN.md).");

    opt.elevation = G_define_standard_option(G_OPT_R_ELEV);

    opt.drainage = G_define_standard_option(G_OPT_R_INPUT);
    opt.drainage->key = "drainage";
    opt.drainage->required = YES;
    opt.drainage->description =
        _("D8 flow direction, r.watershed/r.watershed.opencl 'drainage' "
          "convention (counter-clockwise from north-east=1, negative at "
          "domain edges). Not auto-derived in this increment -- run "
          "r.watershed.opencl (or r.watershed) first.");

    opt.accumulation = G_define_standard_option(G_OPT_R_INPUT);
    opt.accumulation->key = "accumulation";
    opt.accumulation->required = YES;
    opt.accumulation->description =
        _("Flow accumulation (cell count), r.watershed/r.watershed.opencl "
          "convention. Magnitude only is used (sign, where present, is "
          "r.watershed's own per-cell reliability flag, not part of RRI's "
          "model -- see drainage_to_rri_dir's neighboring comment for the "
          "equivalent caveat on 'drainage').");

    opt.riv_thresh = G_define_option();
    opt.riv_thresh->key = "riv_thresh";
    opt.riv_thresh->type = TYPE_DOUBLE;
    opt.riv_thresh->answer = "100";
    opt.riv_thresh->description =
        _("Minimum flow accumulation (cell count) for a cell to be "
          "treated as a river channel");

    opt.width_c = G_define_option();
    opt.width_c->key = "width_param_c";
    opt.width_c->type = TYPE_DOUBLE;
    opt.width_c->answer = "5.0";
    opt.width_c->description = _("River width power-law coefficient c: width = c * (drainage area km^2)^s");

    opt.width_s = G_define_option();
    opt.width_s->key = "width_param_s";
    opt.width_s->type = TYPE_DOUBLE;
    opt.width_s->answer = "0.35";
    opt.width_s->description = _("River width power-law exponent s");

    opt.depth_c = G_define_option();
    opt.depth_c->key = "depth_param_c";
    opt.depth_c->type = TYPE_DOUBLE;
    opt.depth_c->answer = "0.95";
    opt.depth_c->description = _("River depth power-law coefficient c");

    opt.depth_s = G_define_option();
    opt.depth_s->key = "depth_param_s";
    opt.depth_s->type = TYPE_DOUBLE;
    opt.depth_s->answer = "0.2";
    opt.depth_s->description = _("River depth power-law exponent s");

    opt.height_param = G_define_option();
    opt.height_param->key = "height_param";
    opt.height_param->type = TYPE_DOUBLE;
    opt.height_param->answer = "0.0";
    opt.height_param->description = _("Levee height above bank [m] for river cells above height_limit");

    opt.height_limit = G_define_option();
    opt.height_limit->key = "height_limit_param";
    opt.height_limit->type = TYPE_DOUBLE;
    opt.height_limit->answer = "20";
    opt.height_limit->description = _("Minimum accumulation for height_param to apply");

    opt.ns_river = G_define_option();
    opt.ns_river->key = "ns_river";
    opt.ns_river->type = TYPE_DOUBLE;
    opt.ns_river->answer = "0.03";
    opt.ns_river->description = _("River Manning's roughness coefficient");

    opt.ns_slope = G_define_option();
    opt.ns_slope->key = "ns_slope";
    opt.ns_slope->type = TYPE_DOUBLE;
    opt.ns_slope->answer = "0.4";
    opt.ns_slope->description = _("Hillslope Manning's roughness coefficient");

    opt.soildepth = G_define_option();
    opt.soildepth->key = "soildepth";
    opt.soildepth->type = TYPE_DOUBLE;
    opt.soildepth->answer = "1.0";
    opt.soildepth->description = _("Soil depth [m]");

    opt.gammaa = G_define_option();
    opt.gammaa->key = "gammaa";
    opt.gammaa->type = TYPE_DOUBLE;
    opt.gammaa->answer = "0.475";
    opt.gammaa->description = _("Soil porosity [-]");

    opt.ksv = G_define_option();
    opt.ksv->key = "ksv";
    opt.ksv->type = TYPE_DOUBLE;
    opt.ksv->answer = "0.0";
    opt.ksv->description = _("Green-Ampt vertical saturated hydraulic conductivity [m/s] (0 disables)");

    opt.faif = G_define_option();
    opt.faif->key = "faif";
    opt.faif->type = TYPE_DOUBLE;
    opt.faif->answer = "0.316";
    opt.faif->description = _("Green-Ampt wetting front suction head [m]");

    opt.ka = G_define_option();
    opt.ka->key = "ka";
    opt.ka->type = TYPE_DOUBLE;
    opt.ka->answer = "0.0";
    opt.ka->description = _("Lateral subsurface (Darcy) conductivity [m/s] (0 disables; mutually exclusive with ksv)");

    opt.gammam = G_define_option();
    opt.gammam->key = "gammam";
    opt.gammam->type = TYPE_DOUBLE;
    opt.gammam->answer = "0.0";
    opt.gammam->description = _("Matrix-flow porosity [-]");

    opt.beta = G_define_option();
    opt.beta->key = "beta";
    opt.beta->type = TYPE_DOUBLE;
    opt.beta->answer = "8.0";
    opt.beta->description = _("Hillslope subsurface flow power-law exponent");

    opt.utm = G_define_option();
    opt.utm->key = "utm";
    opt.utm->type = TYPE_INTEGER;
    opt.utm->options = "0,1";
    opt.utm->answer = "0";
    opt.utm->description = _("0 = lat/lon (geodesic dx/dy via Hubeny's formula), 1 = projected/UTM");

    opt.rain = G_define_standard_option(G_OPT_R_INPUT);
    opt.rain->key = "rain";
    opt.rain->required = NO;
    opt.rain->description =
        _("Precipitation intensity [mm/h], a SINGLE static raster applied "
          "as constant forcing for the whole run -- this increment does "
          "not yet iterate a rain_strds= time series (see "
          "NATIVE_GRASS_PLAN.md 'Progress'); reads/converts/indexes this "
          "raster into slope-idx space and reports a diagnostic, but does "
          "NOT yet drive the RK45 time loop (not wired up yet either).");

    eight_dir_flag = G_define_flag();
    eight_dir_flag->key = 'e';
    eight_dir_flag->description = _("8-direction hillslope routing (default: 4-direction)");

    if (G_parser(argc, argv)) return EXIT_FAILURE;

    if (atof(opt.ksv->answer) > 0.0 && atof(opt.ka->answer) > 0.0)
        G_fatal_error(_("ksv and ka cannot both be nonzero (RRI's own parameter-check rule)"));

    struct Cell_head window;
    G_get_window(&window);
    int ny = window.rows, nx = window.cols;
    size_t ncell = (size_t)ny * nx;

    const char *mapset;

    mapset = G_find_raster2(opt.elevation->answer, "");
    if (!mapset) G_fatal_error(_("Raster map <%s> not found"), opt.elevation->answer);
    int fd_elev = Rast_open_old(opt.elevation->answer, mapset);
    DCELL *zs_row = G_malloc(nx * sizeof(DCELL));
    double *zs = G_malloc(ncell * sizeof(double));
    for (int row = 0; row < ny; row++) {
        Rast_get_d_row(fd_elev, zs_row, row);
        for (int col = 0; col < nx; col++)
            zs[(size_t)row * nx + col] = Rast_is_d_null_value(&zs_row[col]) ? -9999.0 : zs_row[col];
    }
    Rast_close(fd_elev);
    G_free(zs_row);

    mapset = G_find_raster2(opt.drainage->answer, "");
    if (!mapset) G_fatal_error(_("Raster map <%s> not found"), opt.drainage->answer);
    int fd_dir = Rast_open_old(opt.drainage->answer, mapset);
    DCELL *dir_row = G_malloc(nx * sizeof(DCELL));
    int *dir_rri = G_malloc(ncell * sizeof(int));
    for (int row = 0; row < ny; row++) {
        Rast_get_d_row(fd_dir, dir_row, row);
        for (int col = 0; col < nx; col++)
            dir_rri[(size_t)row * nx + col] = drainage_to_rri_dir(dir_row[col]);
    }
    Rast_close(fd_dir);
    G_free(dir_row);

    mapset = G_find_raster2(opt.accumulation->answer, "");
    if (!mapset) G_fatal_error(_("Raster map <%s> not found"), opt.accumulation->answer);
    int fd_acc = Rast_open_old(opt.accumulation->answer, mapset);
    DCELL *acc_row = G_malloc(nx * sizeof(DCELL));
    double *acc = G_malloc(ncell * sizeof(double));
    for (int row = 0; row < ny; row++) {
        Rast_get_d_row(fd_acc, acc_row, row);
        for (int col = 0; col < nx; col++) {
            double v = Rast_is_d_null_value(&acc_row[col]) ? 0.0 : acc_row[col];
            acc[(size_t)row * nx + col] = fabs(v); /* sign is r.watershed's reliability flag, not RRI's model -- see opt.accumulation->description */
        }
    }
    Rast_close(fd_acc);
    G_free(acc_row);

    rri_grid g;
    memset(&g, 0, sizeof(g));
    g.ny = ny; g.nx = nx;
    g.xllcorner = window.west; g.yllcorner = window.south;
    g.cellsize = window.ew_res; /* assumed square cells, matching the vendored engine's own assumption */

    g.zb = G_malloc(ncell * sizeof(double));
    g.zb_riv = G_malloc(ncell * sizeof(double));
    g.acc = acc;
    g.dir = dir_rri;
    g.domain = G_calloc(ncell, sizeof(int));
    g.riv = G_calloc(ncell, sizeof(int));
    g.land = G_malloc(ncell * sizeof(int));
    g.width = G_calloc(ncell, sizeof(double));
    g.depth = G_calloc(ncell, sizeof(double));
    g.height = G_calloc(ncell, sizeof(double));
    g.len_riv = G_calloc(ncell, sizeof(double));
    g.area_ratio = G_calloc(ncell, sizeof(double));

    double soildepth = atof(opt.soildepth->answer);

    for (size_t p = 0; p < ncell; p++) {
        g.land[p] = 1;
        if (zs[p] > -100.0)
            g.domain[p] = (g.dir[p] == 0) ? 2 : 1;
        g.zb[p] = zs[p] - soildepth;
        g.zb_riv[p] = zs[p];
    }

    /* dx, dy (matches vendored engine main.c's STEP 2, utm==0 path) */
    if (atoi(opt.utm->answer) == 0) {
        double d1 = rri_hubeny_sub(g.xllcorner, g.yllcorner, g.xllcorner + nx * g.cellsize, g.yllcorner);
        double d2 = rri_hubeny_sub(g.xllcorner, g.yllcorner + ny * g.cellsize, g.xllcorner + nx * g.cellsize, g.yllcorner + ny * g.cellsize);
        double d3 = rri_hubeny_sub(g.xllcorner, g.yllcorner, g.xllcorner, g.yllcorner + ny * g.cellsize);
        double d4 = rri_hubeny_sub(g.xllcorner + nx * g.cellsize, g.yllcorner, g.xllcorner + nx * g.cellsize, g.yllcorner + ny * g.cellsize);
        g.dx = (d1 + d2) / 2.0 / nx;
        g.dy = (d3 + d4) / 2.0 / ny;
    } else {
        g.dx = g.cellsize; g.dy = g.cellsize;
    }
    g.area = g.dx * g.dy;
    g.length = sqrt(g.dx * g.dy);
    G_message(_("dx=%.3f dy=%.3f area=%.3f"), g.dx, g.dy, g.area);

    double riv_thresh = atof(opt.riv_thresh->answer);
    double width_c = atof(opt.width_c->answer), width_s = atof(opt.width_s->answer);
    double depth_c = atof(opt.depth_c->answer), depth_s = atof(opt.depth_s->answer);
    double height_param = atof(opt.height_param->answer);
    double height_limit = atof(opt.height_limit->answer);

    if (riv_thresh > 0)
        for (size_t p = 0; p < ncell; p++)
            if (g.acc[p] > riv_thresh) g.riv[p] = 1;

    for (size_t p = 0; p < ncell; p++) {
        if (g.riv[p] != 1) continue;
        double km2 = g.acc[p] * g.dx * g.dy * 1e-6;
        g.width[p] = width_c * pow(km2, width_s);
        g.depth[p] = depth_c * pow(km2, depth_s);
        if (g.acc[p] > height_limit) g.height[p] = height_param;
        g.len_riv[p] = g.length;
        g.area_ratio[p] = g.width[p] * g.len_riv[p] / g.area;
        g.zb_riv[p] = zs[p] - g.depth[p];
    }
    G_free(zs);

    rri_landuse lu;
    set_single_landuse_defaults(&lu, atof(opt.ns_slope->answer), soildepth,
                                 atof(opt.gammaa->answer), atof(opt.ksv->answer),
                                 atof(opt.faif->answer), atof(opt.ka->answer),
                                 atof(opt.gammam->answer), atof(opt.beta->answer));

    rri_riv_cellset rc;
    rri_slo_cellset sc;
    memset(&rc, 0, sizeof(rc));
    memset(&sc, 0, sizeof(sc));

    if (rri_riv_idx_setting(&g, &rc) != 0)
        G_fatal_error(_("rri_riv_idx_setting failed -- see stderr diagnostic above"));
    if (rri_slo_idx_setting(&g, &lu, eight_dir_flag->answer ? 1 : 0, &sc) != 0)
        G_fatal_error(_("rri_slo_idx_setting failed -- see stderr diagnostic above"));

    G_message(_("riv_count=%d slo_count=%d"), rc.count, sc.count);

    /* Forcing input, increment 2 (NATIVE_GRASS_PLAN.md "Progress"):
     * read a single static rain raster and index it into slope-idx
     * space, matching what RRI.f90's `qp_t_idx` is at any one instant
     * inside the time loop -- but the time loop itself (and therefore a
     * rain_strds= time series, which needs a *sequence* of these) is
     * NOT wired up yet. This only proves the read+convert+index path is
     * correct in isolation, via a checkable diagnostic below. */
    if (opt.rain->answer) {
        mapset = G_find_raster2(opt.rain->answer, "");
        if (!mapset) G_fatal_error(_("Raster map <%s> not found"), opt.rain->answer);
        int fd_rain = Rast_open_old(opt.rain->answer, mapset);
        DCELL *rain_row = G_malloc(nx * sizeof(DCELL));
        double *qp_t = G_calloc(ncell, sizeof(double));
        for (int row = 0; row < ny; row++) {
            Rast_get_d_row(fd_rain, rain_row, row);
            for (int col = 0; col < nx; col++)
                qp_t[(size_t)row * nx + col] =
                    Rast_is_d_null_value(&rain_row[col]) ? 0.0 : rain_row[col];
        }
        Rast_close(fd_rain);
        G_free(rain_row);

        double *qp_t_idx = G_malloc(sc.count * sizeof(double));
        rri_slo_ij2idx(&sc, qp_t, nx, qp_t_idx);

        /* Diagnostic: mean of qp_t_idx over active slope cells should
         * equal the raster's own mean over in-domain cells (a uniform
         * input raster makes this trivially checkable by eye -- see
         * tests/native_io_test.py's rain cross-check for the automated
         * version of this same idea). */
        double sum = 0.0;
        for (int k = 0; k < sc.count; k++) sum += qp_t_idx[k];
        G_message(_("rain: mean qp_t_idx over %d active slope cells = %.6f mm/h"),
                   sc.count, sc.count ? sum / sc.count : 0.0);
        G_free(qp_t);
        G_free(qp_t_idx);
    }

    rri_riv_cellset_free(&rc);
    rri_slo_cellset_free(&sc);

    return EXIT_SUCCESS;
}
