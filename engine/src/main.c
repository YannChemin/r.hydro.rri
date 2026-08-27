/**
 * @file main.c
 * @brief Driver: reads a config + grids, builds the sparse cellsets, then
 * runs the main coupled time loop -- for each outer timestep `t` (of
 * length `dt`), advance the river depth to time `t*dt` via an adaptive
 * RK45 sub-loop, then the slope depth the same way, then (if enabled)
 * groundwater, then apply river<->slope exchange and infiltration once,
 * then write that timestep's mass-balance and hydrograph output. This
 * is a direct translation of RRI.f90's STEP 0-2 setup and its single
 * large main `do t = 1, maxt` loop -- see the "----" section-divider
 * comments below, each one corresponds to one of that loop's `!*******`
 * comment banners in the Fortran source.
 *
 * @par Coupling order (matches RRI.f90 exactly; do not reorder without
 * re-deriving against the Fortran source -- the order matters because
 * each stage's output is the next stage's input for the SAME outer
 * timestep, not last timestep's):
 * river RK45 -> slope RK45 (consumes this timestep's rainfall) ->
 * groundwater RK45 (if gw_switch) -> river<->slope exchange ->
 * Green-Ampt infiltration -> outlet drain -> output.
 *
 * @par Adaptive RK45 structure (repeated three times below: river,
 * slope, groundwater -- each with its own independent `ddt` and
 * accept/reject sequence within the shared outer `dt`):
 * an inner `for (;;)` loop evaluates all 6 Cash-Karp stages
 * (rri_rk_coeffs, src/rri_rk.c) at the current trial step size `ddt`,
 * computes the 4th-vs-5th-order error estimate, and either accepts the
 * step (breaks out) or shrinks `ddt` and retries. The error norm
 * (`errmax`) MUST be the plain signed maximum across cells, not
 * `fabs()`-then-max -- see the inline comment at the first `errmax =
 * -DBL_MAX` below for why: using `fabs()` there was a real bug (large
 * negative per-cell errors incorrectly forced step shrinks that
 * Fortran's `maxval()` never would have), found by diffing this port's
 * accepted-`ddt` sequence against the compiled Fortran binary's after a
 * full 360-hour run showed unexplained divergence -- see README.md's
 * "Root-caused bug" section for the full writeup, including the OTHER
 * (larger-impact) bug found in the same pass: `zb` vs `zb_riv` bed
 * elevation being conflated during grid setup, below.
 *
 * Scope (see README.md for the full list and why each omission is safe
 * against the validated solo30s config): rectangular-channel diffusive-
 * wave river+slope routing, river<->slope exchange, groundwater,
 * Green-Ampt infiltration. NOT implemented: dam, diversion, boundary
 * conditions, custom cross-sections (sec_map), evapotranspiration,
 * initial-condition files, rivfile_switch>=1 (river geometry from
 * files), periodic full-grid output (hs_/hr_/qr_/... grids) -- only
 * hydro.txt and storage.dat are written, which is exactly what's needed
 * to validate against the Fortran reference (see README.md's
 * "Validation" section for the comparison procedure).
 *
 * Fortran reference: RRI.f90 (the whole file).
 */
#include "rri/rri.h"
#include "rri/opencl.h"

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/** @brief Join a dataset directory with an RRI_Input.txt-relative path
 * (which conventionally starts with "./", e.g. "./rain/rain.dat" --
 * stripped here so the join doesn't produce a doubled path separator). */
static char *joinpath(const char *dir, const char *rel, char *buf, size_t n)
{
    if (rel[0] == '.' && rel[1] == '/') rel += 2;
    snprintf(buf, n, "%s%s", dir, rel);
    return buf;
}

/** @brief Everything needed to run one simulation: parsed config, grid,
 * both cellsets, rainfall forcing, and the parsed hydro-output point
 * list -- bundled so main() can pass "the whole setup" around without a
 * long parameter list; not part of the public API (rri.h), local to
 * this driver only. */
typedef struct {
    rri_config cfg;
    rri_grid grid;
    rri_riv_cellset rc;
    rri_slo_cellset sc;
    rri_rain rain;
    int maxhydro;
    int *hydro_i, *hydro_j;
} rri_model;

/**
 * @brief Load a rainfall forcing file into @p rain (see rri.h:
 * rri_rain's doc for the file format). Two passes: first counts
 * non-blank lines to size the output arrays (the file's total time-block
 * count isn't known up front without either a second header field or a
 * full scan), second actually reads the values.
 * @return 0 on success, -1 on a missing file or malformed header.
 */
static int load_rain(const char *path, rri_rain *rain)
{
    FILE *f = fopen(path, "r");
    if (!f) { fprintf(stderr, "cannot open rain file %s\n", path); return -1; }

    /* First pass: count non-blank lines and read header for nx_rain/ny_rain. */
    char line[1 << 16];
    long nlines = 0;
    int nx_rain = 0, ny_rain = 0;
    double t0 = 0.0;
    int have_header = 0;
    while (fgets(line, sizeof(line), f)) {
        char *p = line; while (*p == ' ' || *p == '\t') p++;
        if (*p == '\n' || *p == '\0' || *p == '\r') continue;
        nlines++;
        if (!have_header) {
            sscanf(p, "%lf %d %d", &t0, &nx_rain, &ny_rain);
            have_header = 1;
        }
    }
    rewind(f);
    if (nx_rain <= 0 || ny_rain <= 0) { fclose(f); fprintf(stderr, "bad rain header in %s\n", path); return -1; }

    int nt = (int)(nlines / (1 + ny_rain));
    rain->nt = nt; rain->nx_rain = nx_rain; rain->ny_rain = ny_rain;
    rain->t = (double *)calloc((size_t)nt, sizeof(double));
    rain->qp = (double *)calloc((size_t)nt * ny_rain * nx_rain, sizeof(double));

    for (int tt = 0; tt < nt; tt++) {
        do { if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; } } while (line[0] == '\n' || line[0] == '\r');
        double t; int a, b;
        sscanf(line, "%lf %d %d", &t, &a, &b);
        rain->t[tt] = t;
        for (int i = 0; i < ny_rain; i++) {
            do { if (!fgets(line, sizeof(line), f)) { fclose(f); return -1; } } while (line[0] == '\n' || line[0] == '\r');
            char *tok = strtok(line, " \t,");
            for (int j = 0; j < nx_rain; j++) {
                double v = tok ? atof(tok) : 0.0;
                rain->qp[((size_t)tt * ny_rain + i) * nx_rain + j] = v / 3600.0 / 1000.0; /* mm/h -> m/s */
                if (tok) tok = strtok(NULL, " \t,");
            }
        }
    }
    fclose(f);
    return 0;
}

/**
 * @brief Entry point: parse config, build the domain, run the coupled
 * time loop (see file-level comment for the coupling order and adaptive
 * RK45 structure), write hydro.txt/hydro_hr.txt/storage.dat.
 * @param argv[1]  Optional literal "--gpu": dispatch the four hot
 *                 per-cell kernels (river/slope/groundwater discharge,
 *                 infiltration) through the OpenCL backend
 *                 (rri_cl_funcr/funcs/funcg/rri_cl_infilt, src/rri_opencl.c)
 *                 instead of OpenMP -- see the CALL_FUNCR/CALL_FUNCS/
 *                 CALL_FUNCG/CALL_INFILT macros just below main()'s
 *                 local variables for how the SAME RK45 time-loop code
 *                 below dispatches to either backend. Without this flag
 *                 the run is CPU/OpenMP-only and does not touch OpenCL
 *                 at all. Exits with an error if no cl_khr_fp64-capable
 *                 OpenCL device is found.
 * @param argv[1 or 2]  Dataset directory containing RRI_Input.txt and the
 *                 relative paths it references (first positional arg
 *                 after --gpu is consumed, if present).
 * @param argv[2 or 3]  Optional: override rri_config::lasth (hours) without
 *                 editing RRI_Input.txt -- used for fast partial-length
 *                 comparison runs against the Fortran reference (see
 *                 README.md's validation procedure).
 * @return 0 on success; 1 on any setup/config error, an unimplemented
 *         config option being requested (see the switch checks just
 *         below), or a "stepsize underflow" (the adaptive integrator
 *         hit its minimum step size and still failed the error test --
 *         matches RRI.f90's `stop 'stepsize underflow'`).
 */
int main(int argc, char **argv)
{
    int use_gpu = 0;
    if (argc >= 2 && strcmp(argv[1], "--gpu") == 0) { use_gpu = 1; argc--; argv++; }
    if (argc < 2) {
        fprintf(stderr, "usage: %s [--gpu] <datadir> [lasth_override_hours]\n", argv[0]);
        return 1;
    }
    char datadir[512];
    snprintf(datadir, sizeof(datadir), "%s", argv[1]);
    size_t dl = strlen(datadir);
    if (dl == 0 || datadir[dl - 1] != '/') strncat(datadir, "/", sizeof(datadir) - dl - 1);

    char path[1024], inputtxt[1024];
    snprintf(inputtxt, sizeof(inputtxt), "%sRRI_Input.txt", datadir);

    rri_model m;
    memset(&m, 0, sizeof(m));
    if (rri_config_read(inputtxt, &m.cfg) != 0) return 1;
    if (argc >= 3) m.cfg.lasth = atoi(argv[2]);
    if (m.cfg.rivfile_switch != 0) {
        fprintf(stderr, "main: rivfile_switch=%d not supported (only 0: parametric river geometry) -- see README\n", m.cfg.rivfile_switch);
        return 1;
    }
    if (m.cfg.sec_switch != 0 || m.cfg.dam_switch != 0 || m.cfg.div_switch != 0 ||
        m.cfg.bound_slo_wlev_switch != 0 || m.cfg.bound_riv_wlev_switch != 0 ||
        m.cfg.bound_slo_disc_switch != 0 || m.cfg.bound_riv_disc_switch != 0 ||
        m.cfg.evp_switch != 0) {
        fprintf(stderr, "main: sec/dam/div/boundary/evp switches are not implemented in this port -- see README\n");
        return 1;
    }

    rri_grid *g = &m.grid;
    joinpath(datadir, m.cfg.demfile, path, sizeof(path));
    if (rri_read_gis_header(path, &g->ny, &g->nx, &g->xllcorner, &g->yllcorner, &g->cellsize) != 0) return 1;
    int ny = g->ny, nx = g->nx;
    size_t ncell = (size_t)ny * nx;

    double *zs = calloc(ncell, sizeof(double)); /* raw DEM -- zb/zb_riv are both DERIVED from this, see rri.h */
    g->zb = calloc(ncell, sizeof(double));
    g->zb_riv = calloc(ncell, sizeof(double));
    g->acc = calloc(ncell, sizeof(double));
    g->dir = calloc(ncell, sizeof(int));
    g->domain = calloc(ncell, sizeof(int));
    g->riv = calloc(ncell, sizeof(int));
    g->land = calloc(ncell, sizeof(int));
    g->width = calloc(ncell, sizeof(double));
    g->depth = calloc(ncell, sizeof(double));
    g->height = calloc(ncell, sizeof(double));
    g->len_riv = calloc(ncell, sizeof(double));
    g->area_ratio = calloc(ncell, sizeof(double));

    if (rri_read_gis_real(path, ny, nx, g->xllcorner, g->yllcorner, g->cellsize, zs) != 0) return 1;
    joinpath(datadir, m.cfg.accfile, path, sizeof(path));
    if (rri_read_gis_real(path, ny, nx, g->xllcorner, g->yllcorner, g->cellsize, g->acc) != 0) return 1;
    joinpath(datadir, m.cfg.dirfile, path, sizeof(path));
    { double *dtmp = calloc(ncell, sizeof(double));
      if (rri_read_gis_real(path, ny, nx, g->xllcorner, g->yllcorner, g->cellsize, dtmp) != 0) return 1;
      for (size_t p = 0; p < ncell; p++) g->dir[p] = (int)dtmp[p];
      free(dtmp);
    }
    for (size_t p = 0; p < ncell; p++) g->land[p] = 1; /* land_switch not implemented: uniform landuse 1 */

    for (size_t p = 0; p < ncell; p++) {
        if (zs[p] > -100.0) {
            g->domain[p] = (g->dir[p] == 0 || g->dir[p] == -1) ? 2 : 1;
        }
    }

    /* Slope bed elevation: dem - soildepth[land] for EVERY cell (RRI.f90
     * ~line 223). zb_riv (river bed, dem - channel depth) is filled in
     * below once river geometry is known -- defaults to the raw dem
     * elsewhere, matching RRI.py's `zb_riv = zs` initialization. */
    for (size_t p = 0; p < ncell; p++) {
        int li = g->land[p] - 1;
        if (li < 0) li = 0;
        if (li >= m.cfg.lu.n) li = m.cfg.lu.n - 1;
        g->zb[p] = zs[p] - m.cfg.lu.soildepth[li];
        g->zb_riv[p] = zs[p];
    }

    /* dx, dy (RRI.f90 STEP 2, utm==0: geodesic via hubeny_sub) */
    if (m.cfg.utm == 0) {
        double d1 = rri_hubeny_sub(g->xllcorner, g->yllcorner, g->xllcorner + nx * g->cellsize, g->yllcorner);
        double d2 = rri_hubeny_sub(g->xllcorner, g->yllcorner + ny * g->cellsize, g->xllcorner + nx * g->cellsize, g->yllcorner + ny * g->cellsize);
        double d3 = rri_hubeny_sub(g->xllcorner, g->yllcorner, g->xllcorner, g->yllcorner + ny * g->cellsize);
        double d4 = rri_hubeny_sub(g->xllcorner + nx * g->cellsize, g->yllcorner, g->xllcorner + nx * g->cellsize, g->yllcorner + ny * g->cellsize);
        g->dx = (d1 + d2) / 2.0 / nx;
        g->dy = (d3 + d4) / 2.0 / ny;
    } else {
        g->dx = g->cellsize; g->dy = g->cellsize;
    }
    g->area = g->dx * g->dy;
    g->length = sqrt(g->dx * g->dy);
    fprintf(stderr, "dx=%.3f dy=%.3f area=%.3f\n", g->dx, g->dy, g->area);

    /* river mask + parametric geometry (rivfile_switch==0 path only) */
    if (m.cfg.riv_thresh > 0) {
        for (size_t p = 0; p < ncell; p++) if (g->acc[p] > m.cfg.riv_thresh) g->riv[p] = 1;
    }
    for (size_t p = 0; p < ncell; p++) {
        if (g->riv[p] != 1) continue;
        double acc_pos = g->acc[p] > 0 ? g->acc[p] : 0.0;
        double km2 = acc_pos * g->dx * g->dy * 1e-6;
        g->width[p] = m.cfg.width_param_c * pow(km2, m.cfg.width_param_s);
        g->depth[p] = m.cfg.depth_param_c * pow(km2, m.cfg.depth_param_s);
        if (g->acc[p] > m.cfg.height_limit_param) g->height[p] = m.cfg.height_param;
        g->len_riv[p] = g->length;
        g->area_ratio[p] = g->width[p] * g->len_riv[p] / g->area;
        g->zb_riv[p] = zs[p] - g->depth[p]; /* channel cut into the dem -- was missing entirely before this fix */
    }
    /* NOT implemented: RRI.py raises zs by `height` (levee height) before
     * deriving zb/zb_riv when height>0, i.e. a levee also raises the
     * ground surface itself, not just the exchange threshold in funcrs.
     * height_param=0 in the validated solo30s config makes this a no-op
     * there, so it doesn't affect the comparison below, but any dataset
     * with height_param>0 needs this added first. */
    free(zs);

    if (rri_riv_idx_setting(g, &m.rc) != 0) return 1;
    if (rri_slo_idx_setting(g, &m.cfg.lu, m.cfg.eight_dir, &m.sc) != 0) return 1;
    fprintf(stderr, "riv_count=%d slo_count=%d\n", m.rc.count, m.sc.count);

    joinpath(datadir, m.cfg.rainfile, path, sizeof(path));
    if (load_rain(path, &m.rain) != 0) return 1;
    m.rain.xllcorner = m.cfg.xllcorner_rain; m.rain.yllcorner = m.cfg.yllcorner_rain;
    m.rain.cellsize_x = m.cfg.cellsize_rain_x; m.rain.cellsize_y = m.cfg.cellsize_rain_y;

    int *rain_i = calloc((size_t)ny, sizeof(int)), *rain_j = calloc((size_t)nx, sizeof(int));
    for (int j = 0; j < nx; j++)
        rain_j[j] = (int)((g->xllcorner + (j + 0.5) * g->cellsize - m.rain.xllcorner) / m.rain.cellsize_x);
    for (int i = 0; i < ny; i++)
        rain_i[i] = m.rain.ny_rain - 1 - (int)((g->yllcorner + (ny - i - 0.5) * g->cellsize - m.rain.yllcorner) / m.rain.cellsize_y);

    if (m.cfg.hydro_switch) {
        joinpath(datadir, m.cfg.location_file, path, sizeof(path));
        FILE *floc = fopen(path, "r");
        if (!floc) { fprintf(stderr, "cannot open %s\n", path); return 1; }
        char name[128]; int a, b;
        int cap = 16, n = 0;
        m.hydro_i = malloc(sizeof(int) * cap); m.hydro_j = malloc(sizeof(int) * cap);
        while (fscanf(floc, "%127s %d %d", name, &a, &b) == 3) {
            if (n == cap) { cap *= 2; m.hydro_i = realloc(m.hydro_i, sizeof(int) * cap); m.hydro_j = realloc(m.hydro_j, sizeof(int) * cap); }
            m.hydro_i[n] = a - 1; m.hydro_j[n] = b - 1; n++;
        }
        fclose(floc);
        m.maxhydro = n;
    }

    /* ---- state: full-grid depths (hs/hr/hg/gampt_ff) plus every RK45
     * stage's scratch buffer on both the river and slope/groundwater
     * compressed idx representations. Allocated once here, reused every
     * outer timestep -- none of this is per-timestep allocation. ---- */
    double *hs = calloc(ncell, sizeof(double));
    double *hr = calloc(ncell, sizeof(double));
    double *hg = calloc(ncell, sizeof(double));
    double *gampt_ff = calloc(ncell, sizeof(double));
    double *gampt_f = calloc(ncell, sizeof(double));
    double *qr_ave_grid = calloc(ncell, sizeof(double));

    int rc_count = m.rc.count, sc_count = m.sc.count;
    double *hr_idx = calloc(rc_count, sizeof(double));
    double *vr_idx = calloc(rc_count, sizeof(double));
    double *qr_idx = calloc(rc_count, sizeof(double));
    double *qr_ave_idx = calloc(rc_count, sizeof(double));
    double *qr_sum_scratch = calloc(rc_count, sizeof(double));
    double *fr = calloc(rc_count, sizeof(double)), *kr2 = calloc(rc_count, sizeof(double)),
           *kr3 = calloc(rc_count, sizeof(double)), *kr4 = calloc(rc_count, sizeof(double)),
           *kr5 = calloc(rc_count, sizeof(double)), *kr6 = calloc(rc_count, sizeof(double)),
           *vr_temp = calloc(rc_count, sizeof(double)), *vr_err = calloc(rc_count, sizeof(double));

    double *hs_idx = calloc(sc_count, sizeof(double)), *hg_idx = calloc(sc_count, sizeof(double));
    double *gampt_ff_idx = calloc(sc_count, sizeof(double)), *gampt_f_idx = calloc(sc_count, sizeof(double));
    double *qp_t_idx = calloc(sc_count, sizeof(double));
    double *fs = calloc(sc_count, sizeof(double)), *ks2 = calloc(sc_count, sizeof(double)),
           *ks3 = calloc(sc_count, sizeof(double)), *ks4 = calloc(sc_count, sizeof(double)),
           *ks5 = calloc(sc_count, sizeof(double)), *ks6 = calloc(sc_count, sizeof(double)),
           *hs_temp = calloc(sc_count, sizeof(double)), *hs_err = calloc(sc_count, sizeof(double));
    double *qs_buf[RRI_LMAX8]; for (int l = 0; l < RRI_LMAX8; l++) qs_buf[l] = calloc(sc_count, sizeof(double));

    double *fg = calloc(sc_count, sizeof(double)), *kg2 = calloc(sc_count, sizeof(double)),
           *kg3 = calloc(sc_count, sizeof(double)), *kg4 = calloc(sc_count, sizeof(double)),
           *kg5 = calloc(sc_count, sizeof(double)), *kg6 = calloc(sc_count, sizeof(double)),
           *hg_temp = calloc(sc_count, sizeof(double)), *hg_err = calloc(sc_count, sizeof(double));
    double *qg_buf[RRI_LMAX8]; for (int l = 0; l < RRI_LMAX8; l++) qg_buf[l] = calloc(sc_count, sizeof(double));

    double *qp_t_grid = calloc(ncell, sizeof(double));

    rri_cl_backend *cl = NULL;
    if (use_gpu) {
        cl = rri_cl_backend_init(/*prefer_gpu=*/1);
        if (!cl) { fprintf(stderr, "main: --gpu requested but no usable OpenCL device found\n"); return 1; }
        fprintf(stderr, "GPU backend: %s\n", rri_cl_backend_device_name(cl));
    }

    /* Backend dispatch macros: identical call-site shape to the plain
     * CPU calls (rri_funcr/rri_funcs/rri_funcg), just routed through the
     * GPU-backed drivers (rri_cl_funcr/funcs/funcg, src/rri_opencl.c)
     * when `cl` is non-NULL. This keeps the RK45 control-flow structure
     * below IDENTICAL between the CPU-only and --gpu invocations of this
     * binary -- only the per-cell discharge kernel's backend changes,
     * which is exactly what PLAN.md milestone 8's cross-backend
     * validation is meant to isolate. See README.md's OpenCL section for
     * how this was validated (PoCL locally, then the actual AMD
     * Polaris10 GPU on the remote host) against the same solo30s
     * comparison used to validate the CPU path. */
#define CALL_FUNCR(V, HRI, FR, QR) \
    (cl ? rri_cl_funcr(cl, &m.rc, (V), m.cfg.ns_river, g->area, (HRI), (FR), (QR), qr_sum_scratch) \
        : rri_funcr(&m.rc, (V), m.cfg.ns_river, g->area, (HRI), (FR), (QR), qr_sum_scratch))
#define CALL_FUNCS(HSI, QPT, FS, QS) \
    (cl ? rri_cl_funcs(cl, &m.sc, (HSI), (QPT), g->area, (FS), (QS)) \
        : rri_funcs(&m.sc, (HSI), (QPT), g->area, (FS), (QS)))
#define CALL_FUNCG(HGI, FG, QG) \
    (cl ? rri_cl_funcg(cl, &m.sc, (HGI), g->area, (FG), (QG)) \
        : rri_funcg(&m.sc, (HGI), g->area, (FG), (QG)))
#define CALL_INFILT(HSI, GFF, GF) \
    (cl ? rri_cl_infilt(cl, &m.sc, dt, (HSI), (GFF), (GF)) \
        : rri_infilt(&m.sc, dt, (HSI), (GFF), (GF)))

    rri_rk_coeffs rk; rri_rk_coeffs_init(&rk);

    double dt = m.cfg.dt, dt_riv = m.cfg.dt_riv;
    int maxt = (int)(m.cfg.lasth * 3600.0 / dt);
    double rain_sum = 0.0, sout = 0.0;

    joinpath(datadir, m.cfg.outfile_storage, path, sizeof(path));
    FILE *fstorage = fopen(path, "w");
    char hydro_path[1024], hydrohr_path[1024];
    FILE *fhydro = NULL, *fhydrohr = NULL;
    if (m.cfg.hydro_switch) {
        snprintf(hydro_path, sizeof(hydro_path), "%shydro.txt", datadir);
        snprintf(hydrohr_path, sizeof(hydrohr_path), "%shydro_hr.txt", datadir);
        fhydro = fopen(hydro_path, "w");
        fhydrohr = fopen(hydrohr_path, "w");
    }

    for (int t = 1; t <= maxt; t++) {
        fprintf(stderr, "%d / %d\n", t, maxt);

        /* ---- river RK45 (riv_thresh >= 0 always true: riv_thresh is an
         * unsigned config field clamped >=0 by the parser) ------------- */
        double time = (t - 1) * dt;
        double ddt = dt_riv;
        for (int k = 0; k < rc_count; k++) hr_idx[k] = hr[m.rc.idx2i[k] * nx + m.rc.idx2j[k]];
        for (int k = 0; k < rc_count; k++) vr_idx[k] = rri_hr2vr(hr_idx[k], g->area, m.rc.area_ratio[k]);
        for (int k = 0; k < rc_count; k++) qr_ave_idx[k] = 0.0;

        while (time < t * dt) {
            if (time + ddt > t * dt) ddt = t * dt - time;
            double errmax;
            double *qr_ave_temp = calloc(rc_count, sizeof(double));
            for (;;) {
                for (int k = 0; k < rc_count; k++) qr_ave_temp[k] = 0.0;

                CALL_FUNCR(vr_idx, hr_idx, fr, qr_idx);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + rk.b21 * ddt * fr[k]; vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                CALL_FUNCR(vr_temp, hr_idx, kr2, qr_idx);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + ddt * (rk.b31 * fr[k] + rk.b32 * kr2[k]); vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                CALL_FUNCR(vr_temp, hr_idx, kr3, qr_idx);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + ddt * (rk.b41 * fr[k] + rk.b42 * kr2[k] + rk.b43 * kr3[k]); vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                CALL_FUNCR(vr_temp, hr_idx, kr4, qr_idx);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + ddt * (rk.b51 * fr[k] + rk.b52 * kr2[k] + rk.b53 * kr3[k] + rk.b54 * kr4[k]); vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                CALL_FUNCR(vr_temp, hr_idx, kr5, qr_idx);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + ddt * (rk.b61 * fr[k] + rk.b62 * kr2[k] + rk.b63 * kr3[k] + rk.b64 * kr4[k] + rk.b65 * kr5[k]); vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                CALL_FUNCR(vr_temp, hr_idx, kr6, qr_idx);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + ddt * (rk.c1 * fr[k] + rk.c3 * kr3[k] + rk.c4 * kr4[k] + rk.c6 * kr6[k]); vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                /* Fortran: errmax = maxval(hr_err)/eps -- the SIGNED max, not
                 * maxval(abs(hr_err)). A large negative error does NOT trigger
                 * a shrink; only ruled out here after diffing accepted-ddt
                 * sequences against the Fortran binary and finding this exact
                 * mismatch (see README "known gaps" / root-cause note). */
                errmax = -DBL_MAX;
                for (int k = 0; k < rc_count; k++) {
                    vr_err[k] = ddt * (rk.dc1 * fr[k] + rk.dc3 * kr3[k] + rk.dc4 * kr4[k] + rk.dc5 * kr5[k] + rk.dc6 * kr6[k]);
                    double he = (m.rc.domain[k] == 0) ? 0.0 : (vr_err[k] / (g->area * m.rc.area_ratio[k]));
                    if (he > errmax) errmax = he;
                }
                errmax /= rk.eps;

                if (!(errmax > 1.0 && ddt > rk.ddt_min_riv)) break;
                double s1 = rk.safety * ddt * pow(errmax, rk.pshrnk), s2 = 0.5 * ddt;
                ddt = s1 > s2 ? s1 : s2;
                if (ddt < rk.ddt_min_riv) ddt = rk.ddt_min_riv;
                if (ddt == 0) { fprintf(stderr, "stepsize underflow (riv)\n"); return 1; }
            }
            if (ddt == rk.ddt_min_riv) {
                CALL_FUNCR(vr_temp, hr_idx, kr6, qr_idx);
                for (int k = 0; k < rc_count; k++) qr_ave_temp[k] = qr_idx[k] * ddt * 6.0;
            }
            if (time + ddt > t * dt) ddt = t * dt - time;
            time += ddt;
            memcpy(vr_idx, vr_temp, sizeof(double) * rc_count);
            for (int k = 0; k < rc_count; k++) qr_ave_idx[k] += qr_ave_temp[k];
            free(qr_ave_temp);
        }
        for (int k = 0; k < rc_count; k++) qr_ave_idx[k] /= (dt * 6.0);
        for (int k = 0; k < rc_count; k++) hr_idx[k] = rri_vr2hr(vr_idx[k], g->area, m.rc.area_ratio[k]);
        rri_riv_idx2ij(&m.rc, hr_idx, ny, nx, hr);
        rri_riv_idx2ij(&m.rc, qr_ave_idx, ny, nx, qr_ave_grid);

        /* ---- SLOPE: adaptive RK45 to advance hs (slope water depth)
         * from (t-1)*dt to t*dt, same accept/reject structure as the
         * river loop above but re-resolving the active rainfall block
         * (qp_t_grid) at the start of every sub-step, since ddt can span
         * a rainfall time-block boundary. ---- */
        time = (t - 1) * dt; ddt = dt;
        rri_slo_ij2idx(&m.sc, hs, nx, hs_idx);
        rri_slo_ij2idx(&m.sc, gampt_ff, nx, gampt_ff_idx);
        double *qs_ave_idx = calloc(sc_count, sizeof(double));

        while (time < t * dt) {
            if (time + ddt > t * dt) ddt = t * dt - time;

            int itemp = 0;
            for (int jt = 1; jt < m.rain.nt; jt++)
                if (m.rain.t[jt - 1] < (time + ddt) && (time + ddt) <= m.rain.t[jt]) itemp = jt;
            for (int i = 0; i < ny; i++) {
                if (rain_i[i] < 0 || rain_i[i] >= m.rain.ny_rain) continue;
                for (int j = 0; j < nx; j++) {
                    if (rain_j[j] < 0 || rain_j[j] >= m.rain.nx_rain) continue;
                    qp_t_grid[i * nx + j] = m.rain.qp[((size_t)itemp * m.rain.ny_rain + rain_i[i]) * m.rain.nx_rain + rain_j[j]];
                }
            }
            rri_slo_ij2idx(&m.sc, qp_t_grid, nx, qp_t_idx);

            double *qs_ave_temp = calloc(sc_count, sizeof(double));
            double errmax;
            for (;;) {
                for (int k = 0; k < sc_count; k++) qs_ave_temp[k] = 0.0;

                CALL_FUNCS(hs_idx, qp_t_idx, fs, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + rk.b21 * ddt * fs[k]; hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = calloc(sc_count, sizeof(double));
                  for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k];
                  for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt;
                  free(qsum);
                }

                CALL_FUNCS(hs_temp, qp_t_idx, ks2, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + ddt * (rk.b31 * fs[k] + rk.b32 * ks2[k]); hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = calloc(sc_count, sizeof(double)); for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k]; for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt; free(qsum); }

                CALL_FUNCS(hs_temp, qp_t_idx, ks3, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + ddt * (rk.b41 * fs[k] + rk.b42 * ks2[k] + rk.b43 * ks3[k]); hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = calloc(sc_count, sizeof(double)); for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k]; for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt; free(qsum); }

                CALL_FUNCS(hs_temp, qp_t_idx, ks4, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + ddt * (rk.b51 * fs[k] + rk.b52 * ks2[k] + rk.b53 * ks3[k] + rk.b54 * ks4[k]); hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = calloc(sc_count, sizeof(double)); for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k]; for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt; free(qsum); }

                CALL_FUNCS(hs_temp, qp_t_idx, ks5, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + ddt * (rk.b61 * fs[k] + rk.b62 * ks2[k] + rk.b63 * ks3[k] + rk.b64 * ks4[k] + rk.b65 * ks5[k]); hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = calloc(sc_count, sizeof(double)); for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k]; for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt; free(qsum); }

                CALL_FUNCS(hs_temp, qp_t_idx, ks6, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + ddt * (rk.c1 * fs[k] + rk.c3 * ks3[k] + rk.c4 * ks4[k] + rk.c6 * ks6[k]); hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = calloc(sc_count, sizeof(double)); for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k]; for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt; free(qsum); }

                /* signed maxval, not abs -- see the river-loop comment above */
                errmax = -DBL_MAX;
                for (int k = 0; k < sc_count; k++) {
                    hs_err[k] = ddt * (rk.dc1 * fs[k] + rk.dc3 * ks3[k] + rk.dc4 * ks4[k] + rk.dc5 * ks5[k] + rk.dc6 * ks6[k]);
                    double he = (m.sc.domain[k] == 0) ? 0.0 : hs_err[k];
                    if (he > errmax) errmax = he;
                }
                errmax /= rk.eps;

                if (!(errmax > 1.0 && ddt > rk.ddt_min_slo)) break;
                double s1 = rk.safety * ddt * pow(errmax, rk.pshrnk), s2 = 0.5 * ddt;
                ddt = s1 > s2 ? s1 : s2;
                if (ddt < rk.ddt_min_slo) ddt = rk.ddt_min_slo;
                if (ddt == 0) { fprintf(stderr, "stepsize underflow (slo)\n"); return 1; }
            }
            if (time + ddt > t * dt) ddt = t * dt - time;
            time += ddt;
            memcpy(hs_idx, hs_temp, sizeof(double) * sc_count);
            for (int k = 0; k < sc_count; k++) qs_ave_idx[k] += qs_ave_temp[k];
            free(qs_ave_temp);

            for (int i = 0; i < ny; i++) for (int j = 0; j < nx; j++)
                if (g->domain[i * nx + j] != 0) rain_sum += qp_t_grid[i * nx + j] * g->area * ddt;
        }
        for (int k = 0; k < sc_count; k++) qs_ave_idx[k] /= (dt * 6.0);
        free(qs_ave_idx);

        /* ---- GROUNDWATER: only when rri_config::gw_switch is set (any
         * landuse has ksg>0). Vertical recharge/loss happen once before
         * this sub-loop, exfiltration once after -- see rri_gw.c's
         * file-level comment for why those three are NOT part of the
         * RK45 integration itself. ---- */
        if (m.cfg.gw_switch) {
            time = (t - 1) * dt; ddt = dt;
            rri_slo_ij2idx(&m.sc, hg, nx, hg_idx);
            rri_gw_recharge(&m.sc, dt, hs_idx, gampt_ff_idx, hg_idx);
            rri_gw_lose(&m.sc, dt, hg_idx);

            while (time < t * dt) {
                if (time + ddt > t * dt) ddt = t * dt - time;
                double errmax;
                for (;;) {
                    CALL_FUNCG(hg_idx, fg, qg_buf);
                    for (int k = 0; k < sc_count; k++) hg_temp[k] = hg_idx[k] + rk.b21 * ddt * fg[k];

                    CALL_FUNCG(hg_temp, kg2, qg_buf);
                    for (int k = 0; k < sc_count; k++) hg_temp[k] = hg_idx[k] + ddt * (rk.b31 * fg[k] + rk.b32 * kg2[k]);

                    CALL_FUNCG(hg_temp, kg3, qg_buf);
                    for (int k = 0; k < sc_count; k++) hg_temp[k] = hg_idx[k] + ddt * (rk.b41 * fg[k] + rk.b42 * kg2[k] + rk.b43 * kg3[k]);

                    CALL_FUNCG(hg_temp, kg4, qg_buf);
                    for (int k = 0; k < sc_count; k++) hg_temp[k] = hg_idx[k] + ddt * (rk.b51 * fg[k] + rk.b52 * kg2[k] + rk.b53 * kg3[k] + rk.b54 * kg4[k]);

                    CALL_FUNCG(hg_temp, kg5, qg_buf);
                    for (int k = 0; k < sc_count; k++) hg_temp[k] = hg_idx[k] + ddt * (rk.b61 * fg[k] + rk.b62 * kg2[k] + rk.b63 * kg3[k] + rk.b64 * kg4[k] + rk.b65 * kg5[k]);

                    CALL_FUNCG(hg_temp, kg6, qg_buf);
                    for (int k = 0; k < sc_count; k++) hg_temp[k] = hg_idx[k] + ddt * (rk.c1 * fg[k] + rk.c3 * kg3[k] + rk.c4 * kg4[k] + rk.c6 * kg6[k]);

                    /* signed maxval, not abs -- see the river-loop comment above */
                    errmax = -DBL_MAX;
                    for (int k = 0; k < sc_count; k++) {
                        hg_err[k] = ddt * (rk.dc1 * fg[k] + rk.dc3 * kg3[k] + rk.dc4 * kg4[k] + rk.dc5 * kg5[k] + rk.dc6 * kg6[k]);
                        double he = (m.sc.domain[k] == 0) ? 0.0 : hg_err[k];
                        if (he > errmax) errmax = he;
                    }
                    errmax /= rk.eps;
                    if (!(errmax > 1.0 && ddt > rk.ddt_min_slo)) break;
                    double s1 = rk.safety * ddt * pow(errmax, rk.pshrnk), s2 = 0.5 * ddt;
                    ddt = s1 > s2 ? s1 : s2;
                    if (ddt < rk.ddt_min_slo) ddt = rk.ddt_min_slo;
                    if (ddt == 0) { fprintf(stderr, "stepsize underflow (gw)\n"); return 1; }
                }
                if (time + ddt > t * dt) ddt = t * dt - time;
                time += ddt;
                memcpy(hg_idx, hg_temp, sizeof(double) * sc_count);
            }
            time = t * dt;
            rri_gw_exfilt(&m.sc, dt, hs_idx, gampt_ff_idx, hg_idx);
        }

        rri_slo_idx2ij(&m.sc, hs_idx, ny, nx, hs);
        rri_slo_idx2ij(&m.sc, hg_idx, ny, nx, hg);
        rri_slo_idx2ij(&m.sc, gampt_ff_idx, ny, nx, gampt_ff);

        /* ---- RIVER<->SLOPE EXCHANGE: one direct (non-RK45) adjustment
         * per outer timestep, using this timestep's just-updated hs/hr,
         * then re-gathered back onto both idx representations since the
         * exchange (rri_funcrs) operates on the full grid, not either
         * cellset -- see that function's file-level comment for why. ---- */
        rri_funcrs(g, &m.rc, dt, hr, hs);
        rri_riv_ij2idx(&m.rc, hr, nx, hr_idx);
        rri_slo_ij2idx(&m.sc, hs, nx, hs_idx);

        /* ---- INFILTRATION: Green-Ampt, once per outer timestep, after
         * river<->slope exchange has settled this timestep's hs. ---- */
        CALL_INFILT(hs_idx, gampt_ff_idx, gampt_f_idx);
        rri_slo_idx2ij(&m.sc, hs_idx, ny, nx, hs);
        rri_slo_idx2ij(&m.sc, gampt_ff_idx, ny, nx, gampt_ff);

        /* ---- OUTLET DRAIN: any water remaining in a domain==2 (outlet)
         * cell after all of the above has left the modeled domain --
         * zeroed here and tallied into the cumulative `sout` used by the
         * storage.dat mass-balance check below. The linear scan for a
         * river cell's cellset index (`for kk in [0, rc_count)`) is O(n)
         * per outlet cell rather than using a precomputed riv_ij2idx
         * lookup -- fine in practice since outlet cells are a small
         * fraction of the domain, but a candidate for tightening if this
         * ever shows up in a profile. ---- */
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                size_t p = (size_t)i * nx + j;
                if (g->domain[p] != 2) continue;
                sout += hs[p] * g->area;
                hs[p] = 0.0;
                if (g->riv[p] == 1) {
                    int k = -1;
                    for (int kk = 0; kk < rc_count; kk++) if (m.rc.idx2i[kk] == i && m.rc.idx2j[kk] == j) { k = kk; break; }
                    if (k >= 0) sout += rri_hr2vr(hr[p], g->area, m.rc.area_ratio[k]);
                    hr[p] = 0.0;
                }
            }
        }

        if (m.cfg.hydro_switch && ((long)(time) % 3600 == 0)) {
            fprintf(fhydro, "%.2f", time);
            fprintf(fhydrohr, "%.2f", time);
            for (int k = 0; k < m.maxhydro; k++) {
                fprintf(fhydro, ",%.5f", qr_ave_grid[m.hydro_i[k] * nx + m.hydro_j[k]]);
                fprintf(fhydrohr, ",%.5f", hr[m.hydro_i[k] * nx + m.hydro_j[k]]);
            }
            fprintf(fhydro, "\n"); fprintf(fhydrohr, "\n");
            fflush(fhydro); fflush(fhydrohr);
        }

        rri_storage s = rri_storage_calc(g, hs, hr, hg, gampt_ff, &m.sc, &m.rc, m.cfg.riv_thresh);
        double storage = s.ss + s.sr + s.si + s.sg;
        double balance = rain_sum - 0.0 /* aevp_sum (evp not implemented) */ - sout - storage;
        fprintf(fstorage, "%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e,%.7e\n",
                rain_sum, 0.0, 0.0, sout, storage, balance, s.ss, s.sr, s.si, s.sg);
        fflush(fstorage);
    }

    fclose(fstorage);
    if (fhydro) fclose(fhydro);
    if (fhydrohr) fclose(fhydrohr);
    if (cl) rri_cl_backend_free(cl);
    fprintf(stderr, "done.\n");
    return 0;
}
