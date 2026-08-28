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

/* popen/pclose, strptime, timegm are POSIX/BSD extensions, not strict
 * ISO C11 -- needed for resolve_strds_steps' t.rast.list shell-out and
 * UTC date parsing. Must be defined before any system header is
 * included (glibc feature-test-macro convention). */
#define _XOPEN_SOURCE 700
#define _DEFAULT_SOURCE

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include <float.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#include "rri/rri.h"

/* One resolved timestep of a forcing STRDS: the raster map to read and
 * its elapsed seconds since the series' own first map's start -- same
 * convention as RRI.f90's t_rain (a record's timestamp is the END of
 * the interval it applies to; see rri_slo_ij2idx_rain's caller for how
 * this is used). */
typedef struct {
    char name[GNAME_MAX];
    double elapsed_s;
} rri_forcing_step;

/* Runs `t.rast.list input=<strds> columns=name,start_time,end_time
 * separator=|` via popen and parses its output into a chronologically
 * sorted array of rri_forcing_step (elapsed seconds relative to the
 * first entry's own start time). Shelling out to t.rast.list rather
 * than linking GRASS's temporal C library directly is the (a)/(b)
 * choice NATIVE_GRASS_PLAN.md section 3 left undecided -- resolved here
 * as (b) for now: linking libtgis directly from a Module.make build was
 * not attempted/verified this pass, and t.rast.list is a small, one-time
 * (not per-timestep) subprocess call, not a per-value ASCII round-trip
 * of forcing data itself -- the actual rain VALUES still come from
 * Rast_get_d_row on the resolved map names, never from t.rast.list's
 * own output. Revisit if this subprocess call ever proves too slow or
 * fragile for a real pipeline.
 *
 * Every entry must be an INTERVAL registration (a real end_time, not an
 * instantaneous point) -- RRI's forcing model is a step function of
 * elapsed time and needs both ends; fails loudly (not silently) if any
 * entry lacks one, matching the same requirement the now-superseded
 * Python driver's write_forcing_series enforced.
 *
 * Returns the number of steps found (>=0), or -1 on failure (already
 * G_fatal_error'd for anything that should stop the whole run; only
 * returns -1 for "empty STRDS", left to the caller to decide is fatal).
 */
static int resolve_strds_steps(const char *strds, rri_forcing_step **out)
{
    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "t.rast.list input=\"%s\" columns=name,start_time,end_time separator=\"|\" 2>&1",
             strds);
    FILE *p = popen(cmd, "r");
    if (!p) G_fatal_error(_("resolve_strds_steps: popen failed for t.rast.list"));

    typedef struct { char name[GNAME_MAX]; time_t start, end; int has_end; } raw_t;
    int cap = 16, n = 0;
    raw_t *raw = G_malloc(cap * sizeof(raw_t));

    char line[1024];
    int lineno = 0;
    while (fgets(line, sizeof(line), p)) {
        lineno++;
        if (lineno == 1) continue; /* header row: name|start_time|end_time */
        char name[GNAME_MAX] = {0}, start_s[64] = {0}, end_s[64] = {0};
        char *a = strtok(line, "|"), *b = a ? strtok(NULL, "|") : NULL,
             *c = b ? strtok(NULL, "|\n") : NULL;
        if (!a || !b) continue;
        snprintf(name, sizeof(name), "%s", a);
        snprintf(start_s, sizeof(start_s), "%s", b);
        if (c) snprintf(end_s, sizeof(end_s), "%s", c);

        struct tm tm_start = {0}, tm_end = {0};
        if (!strptime(start_s, "%Y-%m-%d %H:%M:%S", &tm_start)) continue;
        if (n == cap) { cap *= 2; raw = G_realloc(raw, cap * sizeof(raw_t)); }
        snprintf(raw[n].name, sizeof(raw[n].name), "%s", name);
        raw[n].start = timegm(&tm_start);
        raw[n].has_end = (end_s[0] != '\0' && strcmp(end_s, "None") != 0 &&
                           strptime(end_s, "%Y-%m-%d %H:%M:%S", &tm_end) != NULL);
        raw[n].end = raw[n].has_end ? timegm(&tm_end) : 0;
        n++;
    }
    pclose(p);

    if (n == 0) { G_free(raw); *out = NULL; return -1; }

    for (int i = 0; i < n; i++)
        if (!raw[i].has_end)
            G_fatal_error(_("resolve_strds_steps: <%s> in STRDS <%s> is registered as an "
                             "instant (no end_time), not an interval -- RRI's forcing format "
                             "needs both; re-register with t.register's 'name|start|end' file "
                             "format"), raw[i].name, strds);

    /* insertion sort by start time -- n is small (a forcing series'
     * timestep count, not a per-cell quantity), no need for qsort here */
    for (int i = 1; i < n; i++) {
        raw_t key = raw[i];
        int j = i - 1;
        while (j >= 0 && raw[j].start > key.start) { raw[j + 1] = raw[j]; j--; }
        raw[j + 1] = key;
    }

    time_t t0 = raw[0].start;
    rri_forcing_step *steps = G_malloc(n * sizeof(rri_forcing_step));
    for (int i = 0; i < n; i++) {
        snprintf(steps[i].name, sizeof(steps[i].name), "%s", raw[i].name);
        steps[i].elapsed_s = difftime(raw[i].end, t0);
    }
    G_free(raw);
    *out = steps;
    return n;
}

/* Reads raster_name via Rast_get_d_row into a full ny*nx grid array
 * (null -> 0.0) and indexes it into slope-idx space via
 * rri_slo_ij2idx -- the exact path increment 2 validated for a single
 * static rain= raster (see NATIVE_GRASS_PLAN.md "Progress"), factored
 * out here so STRDS iteration (increment 3) reuses it unchanged rather
 * than duplicating it. Caller owns qp_t_idx (size sc->count). */
static void read_and_index_forcing_raster(const char *raster_name, int ny, int nx,
                                           const rri_slo_cellset *sc, double *qp_t_idx)
{
    const char *mapset = G_find_raster2(raster_name, "");
    if (!mapset) G_fatal_error(_("Raster map <%s> not found"), raster_name);
    int fd = Rast_open_old(raster_name, mapset);
    DCELL *row_buf = G_malloc(nx * sizeof(DCELL));
    double *grid = G_malloc((size_t)ny * nx * sizeof(double));
    for (int row = 0; row < ny; row++) {
        Rast_get_d_row(fd, row_buf, row);
        for (int col = 0; col < nx; col++)
            grid[(size_t)row * nx + col] =
                Rast_is_d_null_value(&row_buf[col]) ? 0.0 : row_buf[col];
    }
    Rast_close(fd);
    G_free(row_buf);

    rri_slo_ij2idx(sc, grid, nx, qp_t_idx);
    G_free(grid);
}

/* Single-landuse-class defaults, matching the values the vendored
 * engine's own ASCII-driven main.c/RRI_Input.py used for the validated
 * solo30s config -- exposed here as G_parser options rather than a
 * generated RRI_Input.txt. Multi-landuse (per-category) parameterization
 * is NOT implemented in this increment -- see README "Known gaps"; land[]
 * is hardcoded to 1 everywhere, same simplification the vendored engine's
 * own main.c already had. */
/* Parses a G_parser multi-value option's opt->answers (NULL-terminated
 * string array; a plain scalar option still has exactly one entry
 * there) into a caller-allocated double[n] -- fails loudly if the
 * option provided fewer values than num_of_landuse requires, rather
 * than silently reusing the last value or defaulting missing classes
 * (a class silently inheriting another class's parameters would be a
 * much worse surprise than a clear error naming the option). */
static void parse_per_landuse_option(struct Option *opt, int n, double *out, const char *label)
{
    int count = 0;
    if (opt->answers) while (opt->answers[count]) count++;
    if (count == 1 && n > 1) {
        /* A single value with landuse='s multiple classes: apply
         * uniformly rather than fatal -- explicitly documented in the
         * option's own description as the intentional "same value for
         * every class" shorthand, not a silent bug. */
        for (int i = 0; i < n; i++) out[i] = atof(opt->answers[0]);
        return;
    }
    if (count != n)
        G_fatal_error(_("%s: got %d value(s), need exactly %d (one per landuse= "
                         "category 1..%d), or exactly 1 to apply uniformly"),
                       label, count, n, n);
    for (int i = 0; i < n; i++) out[i] = atof(opt->answers[i]);
}

static void set_landuse_from_options(rri_landuse *lu, int n,
                                      struct Option *ns_slope_opt, struct Option *soildepth_opt,
                                      struct Option *gammaa_opt, struct Option *ksv_opt,
                                      struct Option *faif_opt, struct Option *ka_opt,
                                      struct Option *gammam_opt, struct Option *beta_opt)
{
    lu->n = n;
    lu->dif = G_malloc(n * sizeof(int));
    lu->ns_slope = G_malloc(n * sizeof(double));
    lu->soildepth = G_malloc(n * sizeof(double));
    lu->gammaa = G_malloc(n * sizeof(double));
    lu->ksv = G_malloc(n * sizeof(double));
    lu->faif = G_malloc(n * sizeof(double));
    lu->ka = G_malloc(n * sizeof(double));
    lu->gammam = G_malloc(n * sizeof(double));
    lu->beta = G_malloc(n * sizeof(double));
    lu->ksg = G_calloc(n, sizeof(double));
    lu->gammag = G_calloc(n, sizeof(double));
    lu->kg0 = G_calloc(n, sizeof(double));
    lu->fpg = G_calloc(n, sizeof(double));
    lu->rgl = G_calloc(n, sizeof(double));
    lu->da = G_malloc(n * sizeof(double));
    lu->dm = G_malloc(n * sizeof(double));
    lu->infilt_limit = G_malloc(n * sizeof(double));

    parse_per_landuse_option(ns_slope_opt, n, lu->ns_slope, "ns_slope");
    parse_per_landuse_option(soildepth_opt, n, lu->soildepth, "soildepth");
    parse_per_landuse_option(gammaa_opt, n, lu->gammaa, "gammaa");
    parse_per_landuse_option(ksv_opt, n, lu->ksv, "ksv");
    parse_per_landuse_option(faif_opt, n, lu->faif, "faif");
    parse_per_landuse_option(ka_opt, n, lu->ka, "ka");
    parse_per_landuse_option(gammam_opt, n, lu->gammam, "gammam");
    parse_per_landuse_option(beta_opt, n, lu->beta, "beta");

    for (int i = 0; i < n; i++) {
        if (lu->ksv[i] > 0.0 && lu->ka[i] > 0.0)
            G_fatal_error(_("landuse class %d: ksv and ka cannot both be nonzero "
                             "(RRI's own parameter-check rule)"), i + 1);
        lu->dif[i] = 1;
        lu->da[i] = (lu->soildepth[i] > 0.0 && lu->ka[i] > 0.0) ? lu->soildepth[i] * lu->gammaa[i] : 0.0;
        lu->dm[i] = (lu->soildepth[i] > 0.0 && lu->ka[i] > 0.0 && lu->gammam[i] > 0.0)
                        ? lu->soildepth[i] * lu->gammam[i] : 0.0;
        lu->infilt_limit[i] = (lu->soildepth[i] > 0.0 && lu->ksv[i] > 0.0)
                                   ? lu->soildepth[i] * lu->gammaa[i] : 0.0;
    }
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
/* Runs r.watershed.opencl (this user's own GPU-accelerated, validated
 * addon -- see $HOME/dev/r.watershed.opencl/README.md -- a drop-in
 * replacement for r.watershed at basin scale, where stock r.watershed's
 * single-threaded core is impractically slow) to derive drainage/
 * accumulation when the caller did not supply their own, writing to
 * *drainage_out/*accum_out (caller-owned buffers, GNAME_MAX each).
 * Fails loudly (not silently falling back to stock r.watershed) if
 * r.watershed.opencl isn't installed or errors -- matches this
 * project's fail-loud convention; a silent fallback to a much slower
 * tool on a basin-scale DEM would be a worse surprise than a clear
 * error telling the caller to install the addon or supply drainage=/
 * accumulation= themselves. */
static void auto_derive_drainage_accumulation(const char *elevation, int threshold,
                                               char *drainage_out, char *accum_out)
{
    snprintf(drainage_out, GNAME_MAX, "rri_auto_drainage_%d", (int)getpid());
    snprintf(accum_out, GNAME_MAX, "rri_auto_accum_%d", (int)getpid());

    char cmd[1024];
    snprintf(cmd, sizeof(cmd),
             "r.watershed.opencl elevation=\"%s\" threshold=%d "
             "drainage=\"%s\" accumulation=\"%s\" --overwrite",
             elevation, threshold, drainage_out, accum_out);
    G_message(_("drainage=/accumulation= not given -- running: %s"), cmd);
    if (system(cmd) != 0)
        G_fatal_error(_("auto_derive_drainage_accumulation: r.watershed.opencl failed "
                         "(see stderr above) -- install $HOME/dev/r.watershed.opencl, "
                         "or supply drainage=/accumulation= yourself"));
}

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

/* A forcing series already read+converted+indexed once at startup (see
 * increments 2/3 above) into sc->count-long qp_t_idx arrays, one per
 * resolved timestep, plus each step's elapsed seconds. n==1 represents
 * a single static rain= raster applied as constant forcing for the
 * whole run (elapsed_s[0] set to lasth*3600 by the caller so it always
 * matches every bracket search below). */
typedef struct {
    int n;
    double *elapsed_s;   /* [n] */
    double **qp_t_idx;    /* [n][sc->count] */
} rri_forcing_preloaded;

/**
 * @brief Increment 4 (NATIVE_GRASS_PLAN.md "Progress"): the adaptive
 * RK45 coupled river+slope time loop, ported from the vendored engine's
 * src/main.c (lines ~406-676) with ONLY the I/O boundary changed:
 * forcing comes from @p rain (preloaded qp_t_idx per resolved STRDS
 * timestep, see rri_forcing_preloaded) instead of an ASCII rain.dat
 * block-read, and output is a per-outer-timestep mass-balance diagnostic
 * printed via G_message instead of storage.dat/hydro.txt file writes --
 * GRASS-native DB-table/STRDS output is the NEXT increment, not this
 * one (this one's job is proving the physics loop itself is correct
 * when driven by native input, via the SAME kind of diagnostic
 * cross-check every increment before it used).
 *
 * Deliberately NOT ported in this increment (kept out to bound scope,
 * matching what the vendored engine's own config already limits itself
 * to -- see engine/README.md "What's NOT implemented"): groundwater
 * (this module's set_single_landuse_defaults always sets ksg=0, so
 * gw_switch is always false -- the engine's own GW sub-loop is simply
 * omitted here rather than carried across dead code), the OpenCL
 * backend (CALL_FUNCR/FUNCS/GW/INFILT macros collapse to their plain-C
 * calls only -- no --gpu in this increment), and dam/diversion/
 * boundary conditions/custom cross-sections (none of these are
 * supported anywhere in this project's C engine, vendored or native).
 *
 * The RK45 orchestration itself -- the six-stage Cash-Karp evaluation,
 * the accept/reject step-size control using the SIGNED (not fabs'd)
 * error norm, the coupling order (river -> slope -> exchange ->
 * infiltration -> outlet drain per outer timestep) -- is copied as
 * closely as possible to the vendored engine's own proven, validated
 * code, not restructured or "cleaned up," specifically to minimize the
 * risk of a transcription bug in code this project has twice already
 * found real, hard-to-spot bugs in (see the file-level doc comment
 * above main() in engine/src/main.c for both).
 */
/* Increment 5 (NATIVE_GRASS_PLAN.md "Progress"): GRASS-native output.
 * Collected in memory during the loop, then flushed to real GRASS
 * objects (raster/STRDS/DB table) once at the end -- no ASCII files
 * anywhere. Periodic state grids and the DB table are written via
 * ordinary Rast_* calls (rasters) or a shelled-out t.create/t.register/
 * db.execute call (STRDS registration and the DB table itself have no
 * stable C API -- GRASS's temporal framework is Python-only, and DBMI's
 * C API was not attempted this pass now that a working, simple
 * subprocess-based pattern was already established for t.rast.list in
 * increment 3 -- see resolve_strds_steps' doc for that same tradeoff). */
typedef struct {
    char hs_prefix[GNAME_MAX];   /* empty string = disabled */
    int hs_interval;              /* write every Nth outer timestep, N>=1 */
    rri_forcing_step *hs_written; /* reused: name + elapsed_s per written map */
    int hs_written_n, hs_written_cap;

    double *hydro_time;   /* [hydro_n] elapsed seconds */
    double *hydro_q;      /* [hydro_n] discharge at outlet [m^3/s] */
    int hydro_n, hydro_cap;
} rri_output_sink;

static void output_sink_init(rri_output_sink *o, const char *hs_prefix, int hs_interval)
{
    memset(o, 0, sizeof(*o));
    if (hs_prefix) snprintf(o->hs_prefix, sizeof(o->hs_prefix), "%s", hs_prefix);
    o->hs_interval = hs_interval > 0 ? hs_interval : 1;
}

static void write_state_raster(const char *name, int ny, int nx, const double *grid)
{
    int fd = Rast_open_new(name, DCELL_TYPE);
    DCELL *row = G_malloc(nx * sizeof(DCELL));
    for (int i = 0; i < ny; i++) {
        for (int j = 0; j < nx; j++) row[j] = (DCELL)grid[(size_t)i * nx + j];
        Rast_put_row(fd, row, DCELL_TYPE);
    }
    Rast_close(fd);
    G_free(row);
    struct History hist;
    Rast_short_history(name, "raster", &hist);
    Rast_command_history(&hist);
    Rast_write_history(name, &hist);
}

static void output_sink_maybe_write_hs(rri_output_sink *o, int t, double elapsed_s,
                                        int ny, int nx, const double *hs)
{
    if (!o->hs_prefix[0] || t % o->hs_interval != 0) return;
    if (o->hs_written_n == o->hs_written_cap) {
        o->hs_written_cap = o->hs_written_cap ? o->hs_written_cap * 2 : 16;
        o->hs_written = G_realloc(o->hs_written, o->hs_written_cap * sizeof(rri_forcing_step));
    }
    char name[GNAME_MAX];
    snprintf(name, sizeof(name), "%s_%06d", o->hs_prefix, t);
    write_state_raster(name, ny, nx, hs);
    snprintf(o->hs_written[o->hs_written_n].name, GNAME_MAX, "%s", name);
    o->hs_written[o->hs_written_n].elapsed_s = elapsed_s;
    o->hs_written_n++;
}

static void output_sink_record_hydro(rri_output_sink *o, double elapsed_s, double q)
{
    if (o->hydro_n == o->hydro_cap) {
        o->hydro_cap = o->hydro_cap ? o->hydro_cap * 2 : 16;
        o->hydro_time = G_realloc(o->hydro_time, o->hydro_cap * sizeof(double));
        o->hydro_q = G_realloc(o->hydro_q, o->hydro_cap * sizeof(double));
    }
    o->hydro_time[o->hydro_n] = elapsed_s;
    o->hydro_q[o->hydro_n] = q;
    o->hydro_n++;
}

/* Registers every raster in sink->hs_written into a new absolute-time
 * STRDS named `strds_name`, via t.create + a single t.register call
 * (one register-file listing all maps, not one subprocess per map --
 * t.rast.list in increment 3 established this "one small subprocess
 * call, not a per-value round-trip" pattern, reused here). Timestamps
 * are POSIX epoch seconds elapsed_s past `t0`, matching run_rk45_loop's
 * own elapsed-seconds convention throughout. */
static void register_hs_strds(const rri_output_sink *o, const char *strds_name, time_t t0)
{
    if (!o->hs_prefix[0] || o->hs_written_n == 0) return;

    char cmd[512];
    snprintf(cmd, sizeof(cmd),
             "t.create output=\"%s\" type=strds temporaltype=absolute "
             "title=\"r.hydro.rri hillslope depth\" "
             "description=\"hillslope water depth [m], from r.hydro.rri -r\" "
             "--overwrite", strds_name);
    if (system(cmd) != 0)
        G_fatal_error(_("register_hs_strds: t.create failed (see stderr above)"));

    char reg_path[4096];
    snprintf(reg_path, sizeof(reg_path), "%s", G_tempfile());
    FILE *f = fopen(reg_path, "w");
    if (!f) G_fatal_error(_("register_hs_strds: cannot open temp register file"));
    for (int i = 0; i < o->hs_written_n; i++) {
        time_t ts = t0 + (time_t)o->hs_written[i].elapsed_s;
        struct tm *tmv = gmtime(&ts);
        char buf[32];
        strftime(buf, sizeof(buf), "%Y-%m-%d %H:%M:%S", tmv);
        fprintf(f, "%s|%s\n", o->hs_written[i].name, buf);
    }
    fclose(f);

    snprintf(cmd, sizeof(cmd),
             "t.register input=\"%s\" type=raster file=\"%s\"",
             strds_name, reg_path);
    if (system(cmd) != 0)
        G_fatal_error(_("register_hs_strds: t.register failed (see stderr above)"));
    remove(reg_path);
    G_message(_("registered %d hillslope-depth map(s) into STRDS <%s>"),
               o->hs_written_n, strds_name);
}

/* Writes the outlet hydrograph (elapsed_s, discharge_cms) to a new DB
 * table via one `db.execute` call over a generated SQL script (CREATE
 * TABLE + one INSERT per row) -- same "shell out once, not per-row"
 * discipline as register_hs_strds/resolve_strds_steps. */
static void write_hydrograph_table(const rri_output_sink *o, const char *table_name)
{
    if (o->hydro_n == 0) return;
    char sql_path[4096];
    snprintf(sql_path, sizeof(sql_path), "%s", G_tempfile());
    FILE *f = fopen(sql_path, "w");
    if (!f) G_fatal_error(_("write_hydrograph_table: cannot open temp SQL file"));
    fprintf(f, "DROP TABLE IF EXISTS %s;\n", table_name);
    fprintf(f, "CREATE TABLE %s (time_s DOUBLE PRECISION, discharge_cms DOUBLE PRECISION);\n",
            table_name);
    for (int i = 0; i < o->hydro_n; i++)
        fprintf(f, "INSERT INTO %s VALUES (%.2f, %.6f);\n",
                table_name, o->hydro_time[i], o->hydro_q[i]);
    fclose(f);

    /* db.connect -c is idempotent ("DB settings already defined, nothing
     * to do" if a driver/database is already configured) -- a fresh
     * mapset created via grass.script.setup.init (as this project's own
     * pytest fixtures do) does not necessarily have one set up the way
     * `grass -c` does, and db.execute fails outright ("Unable to start
     * driver <(null)>") without it. Always ensure one exists rather than
     * asserting the caller's mapset already has it configured. */
    if (system("db.connect -c") != 0)
        G_fatal_error(_("write_hydrograph_table: db.connect -c failed (see stderr above)"));

    char cmd[4200];
    snprintf(cmd, sizeof(cmd), "db.execute input=\"%s\"", sql_path);
    if (system(cmd) != 0)
        G_fatal_error(_("write_hydrograph_table: db.execute failed (see stderr above)"));
    remove(sql_path);
    G_message(_("wrote %d-row outlet hydrograph to table <%s>"), o->hydro_n, table_name);
}

static void run_rk45_loop(rri_grid *g, rri_riv_cellset *rc, rri_slo_cellset *sc,
                           double dt, double dt_riv, double lasth, int riv_thresh,
                           double ns_river, const rri_forcing_preloaded *rain,
                           rri_output_sink *out)
{
    int nx = g->nx, ny = g->ny;
    int rc_count = rc->count, sc_count = sc->count;

    double *hs = G_calloc((size_t)ny * nx, sizeof(double));
    double *hr = G_calloc((size_t)ny * nx, sizeof(double));
    /* No groundwater sub-loop in this increment (see function doc), but
     * rri_storage_calc dereferences hg unconditionally regardless of
     * whether groundwater is active -- the vendored engine always
     * calloc's a real (zeroed) hg array too (src/main.c line ~325), it
     * is never NULL there either. Passing NULL here instead crashed
     * (SIGSEGV in rri_storage_calc, caught while validating this exact
     * increment) -- a zeroed array, not a null pointer, is the correct
     * "groundwater storage is zero" representation. */
    double *hg = G_calloc((size_t)ny * nx, sizeof(double));
    double *gampt_ff = G_calloc((size_t)ny * nx, sizeof(double));
    double *qr_ave_grid = G_calloc((size_t)ny * nx, sizeof(double));

    double *hr_idx = G_calloc(rc_count, sizeof(double));
    double *vr_idx = G_calloc(rc_count, sizeof(double));
    double *qr_idx = G_calloc(rc_count, sizeof(double));
    double *qr_ave_idx = G_calloc(rc_count, sizeof(double));
    double *qr_sum_scratch = G_calloc(rc_count, sizeof(double));
    double *fr = G_calloc(rc_count, sizeof(double)), *kr2 = G_calloc(rc_count, sizeof(double)),
           *kr3 = G_calloc(rc_count, sizeof(double)), *kr4 = G_calloc(rc_count, sizeof(double)),
           *kr5 = G_calloc(rc_count, sizeof(double)), *kr6 = G_calloc(rc_count, sizeof(double)),
           *vr_temp = G_calloc(rc_count, sizeof(double)), *vr_err = G_calloc(rc_count, sizeof(double));

    double *hs_idx = G_calloc(sc_count, sizeof(double));
    double *gampt_ff_idx = G_calloc(sc_count, sizeof(double)), *gampt_f_idx = G_calloc(sc_count, sizeof(double));
    double *qp_t_idx = G_calloc(sc_count, sizeof(double));
    double *fs = G_calloc(sc_count, sizeof(double)), *ks2 = G_calloc(sc_count, sizeof(double)),
           *ks3 = G_calloc(sc_count, sizeof(double)), *ks4 = G_calloc(sc_count, sizeof(double)),
           *ks5 = G_calloc(sc_count, sizeof(double)), *ks6 = G_calloc(sc_count, sizeof(double)),
           *hs_temp = G_calloc(sc_count, sizeof(double)), *hs_err = G_calloc(sc_count, sizeof(double));
    double *qs_buf[RRI_LMAX8];
    for (int l = 0; l < RRI_LMAX8; l++) qs_buf[l] = G_calloc(sc_count, sizeof(double));

    rri_rk_coeffs rk;
    rri_rk_coeffs_init(&rk);

    int maxt = (int)(lasth * 3600.0 / dt);
    double rain_sum = 0.0, sout = 0.0;

    G_message(_("run_rk45_loop: maxt=%d dt=%.1f dt_riv=%.1f"), maxt, dt, dt_riv);

    for (int t = 1; t <= maxt; t++) {
        /* ---- RIVER: adaptive RK45, (t-1)*dt -> t*dt ------------------ */
        double time = (t - 1) * dt;
        double ddt = dt_riv;
        for (int k = 0; k < rc_count; k++) hr_idx[k] = hr[rc->idx2i[k] * nx + rc->idx2j[k]];
        for (int k = 0; k < rc_count; k++) vr_idx[k] = rri_hr2vr(hr_idx[k], g->area, rc->area_ratio[k]);
        for (int k = 0; k < rc_count; k++) qr_ave_idx[k] = 0.0;

        while (time < t * dt) {
            if (time + ddt > t * dt) ddt = t * dt - time;
            double errmax;
            double *qr_ave_temp = G_calloc(rc_count, sizeof(double));
            for (;;) {
                for (int k = 0; k < rc_count; k++) qr_ave_temp[k] = 0.0;

                rri_funcr(rc, vr_idx, ns_river, g->area, hr_idx, fr, qr_idx, qr_sum_scratch);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + rk.b21 * ddt * fr[k]; vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                rri_funcr(rc, vr_temp, ns_river, g->area, hr_idx, kr2, qr_idx, qr_sum_scratch);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + ddt * (rk.b31 * fr[k] + rk.b32 * kr2[k]); vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                rri_funcr(rc, vr_temp, ns_river, g->area, hr_idx, kr3, qr_idx, qr_sum_scratch);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + ddt * (rk.b41 * fr[k] + rk.b42 * kr2[k] + rk.b43 * kr3[k]); vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                rri_funcr(rc, vr_temp, ns_river, g->area, hr_idx, kr4, qr_idx, qr_sum_scratch);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + ddt * (rk.b51 * fr[k] + rk.b52 * kr2[k] + rk.b53 * kr3[k] + rk.b54 * kr4[k]); vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                rri_funcr(rc, vr_temp, ns_river, g->area, hr_idx, kr5, qr_idx, qr_sum_scratch);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + ddt * (rk.b61 * fr[k] + rk.b62 * kr2[k] + rk.b63 * kr3[k] + rk.b64 * kr4[k] + rk.b65 * kr5[k]); vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                rri_funcr(rc, vr_temp, ns_river, g->area, hr_idx, kr6, qr_idx, qr_sum_scratch);
                for (int k = 0; k < rc_count; k++) { double v = vr_idx[k] + ddt * (rk.c1 * fr[k] + rk.c3 * kr3[k] + rk.c4 * kr4[k] + rk.c6 * kr6[k]); vr_temp[k] = v < 0 ? 0 : v; qr_ave_temp[k] += qr_idx[k] * ddt; }

                /* signed maxval, not abs -- see file-level doc */
                errmax = -DBL_MAX;
                for (int k = 0; k < rc_count; k++) {
                    vr_err[k] = ddt * (rk.dc1 * fr[k] + rk.dc3 * kr3[k] + rk.dc4 * kr4[k] + rk.dc5 * kr5[k] + rk.dc6 * kr6[k]);
                    double he = (rc->domain[k] == 0) ? 0.0 : (vr_err[k] / (g->area * rc->area_ratio[k]));
                    if (he > errmax) errmax = he;
                }
                errmax /= rk.eps;

                if (!(errmax > 1.0 && ddt > rk.ddt_min_riv)) break;
                double s1 = rk.safety * ddt * pow(errmax, rk.pshrnk), s2 = 0.5 * ddt;
                ddt = s1 > s2 ? s1 : s2;
                if (ddt < rk.ddt_min_riv) ddt = rk.ddt_min_riv;
                if (ddt == 0) G_fatal_error(_("run_rk45_loop: stepsize underflow (riv)"));
            }
            if (ddt == rk.ddt_min_riv) {
                rri_funcr(rc, vr_temp, ns_river, g->area, hr_idx, kr6, qr_idx, qr_sum_scratch);
                for (int k = 0; k < rc_count; k++) qr_ave_temp[k] = qr_idx[k] * ddt * 6.0;
            }
            if (time + ddt > t * dt) ddt = t * dt - time;
            time += ddt;
            memcpy(vr_idx, vr_temp, sizeof(double) * rc_count);
            for (int k = 0; k < rc_count; k++) qr_ave_idx[k] += qr_ave_temp[k];
            G_free(qr_ave_temp);
        }
        for (int k = 0; k < rc_count; k++) qr_ave_idx[k] /= (dt * 6.0);
        for (int k = 0; k < rc_count; k++) hr_idx[k] = rri_vr2hr(vr_idx[k], g->area, rc->area_ratio[k]);
        rri_riv_idx2ij(rc, hr_idx, ny, nx, hr);
        rri_riv_idx2ij(rc, qr_ave_idx, ny, nx, qr_ave_grid);

        /* ---- SLOPE: adaptive RK45, re-resolving the active forcing
         * step at the start of every sub-step (ddt can span a forcing
         * timestep boundary) ------------------------------------------ */
        time = (t - 1) * dt; ddt = dt;
        rri_slo_ij2idx(sc, hs, nx, hs_idx);
        rri_slo_ij2idx(sc, gampt_ff, nx, gampt_ff_idx);
        double *qs_ave_idx = G_calloc(sc_count, sizeof(double));

        while (time < t * dt) {
            if (time + ddt > t * dt) ddt = t * dt - time;

            /* Bracket search matching RRI.f90's t_rain convention: find
             * the resolved step whose interval (elapsed_s[j-1],
             * elapsed_s[j]] contains time+ddt, with an implicit j=0
             * zero-rain block before the first resolved step (same as
             * the ASCII engine's own t_rain(0)=0 placeholder). */
            int itemp = -1;
            for (int j = 0; j < rain->n; j++) {
                double lower = (j == 0) ? 0.0 : rain->elapsed_s[j - 1];
                if (lower < (time + ddt) && (time + ddt) <= rain->elapsed_s[j]) itemp = j;
            }
            if (itemp < 0) {
                for (int k = 0; k < sc_count; k++) qp_t_idx[k] = 0.0;
            } else {
                /* rain->qp_t_idx is mm/h (read_and_index_forcing_raster's
                 * documented, tested contract -- increments 2/3's own
                 * diagnostics check this exact unit). The physics
                 * kernels (rri_funcs et al.) expect m/s, matching
                 * RRI.f90/the vendored engine's load_rain(): "qp = qp /
                 * 3600.d0 / 1000.d0" applied once at load time there;
                 * applied here at point of use instead since this
                 * increment preloads once but must not mutate what
                 * increments 2/3 already validated. Missing this
                 * conversion was a real bug caught by cross-checking
                 * rain_sum against the ASCII engine's own storage.dat on
                 * the identical synthetic domain -- off by orders of
                 * magnitude, not a rounding difference. */
                for (int k = 0; k < sc_count; k++)
                    qp_t_idx[k] = rain->qp_t_idx[itemp][k] / 3600.0 / 1000.0;
            }

            double *qs_ave_temp = G_calloc(sc_count, sizeof(double));
            double errmax;
            for (;;) {
                for (int k = 0; k < sc_count; k++) qs_ave_temp[k] = 0.0;

                rri_funcs(sc, hs_idx, qp_t_idx, g->area, fs, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + rk.b21 * ddt * fs[k]; hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = G_calloc(sc_count, sizeof(double));
                  for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k];
                  for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt;
                  G_free(qsum);
                }

                rri_funcs(sc, hs_temp, qp_t_idx, g->area, ks2, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + ddt * (rk.b31 * fs[k] + rk.b32 * ks2[k]); hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = G_calloc(sc_count, sizeof(double)); for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k]; for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt; G_free(qsum); }

                rri_funcs(sc, hs_temp, qp_t_idx, g->area, ks3, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + ddt * (rk.b41 * fs[k] + rk.b42 * ks2[k] + rk.b43 * ks3[k]); hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = G_calloc(sc_count, sizeof(double)); for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k]; for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt; G_free(qsum); }

                rri_funcs(sc, hs_temp, qp_t_idx, g->area, ks4, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + ddt * (rk.b51 * fs[k] + rk.b52 * ks2[k] + rk.b53 * ks3[k] + rk.b54 * ks4[k]); hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = G_calloc(sc_count, sizeof(double)); for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k]; for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt; G_free(qsum); }

                rri_funcs(sc, hs_temp, qp_t_idx, g->area, ks5, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + ddt * (rk.b61 * fs[k] + rk.b62 * ks2[k] + rk.b63 * ks3[k] + rk.b64 * ks4[k] + rk.b65 * ks5[k]); hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = G_calloc(sc_count, sizeof(double)); for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k]; for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt; G_free(qsum); }

                rri_funcs(sc, hs_temp, qp_t_idx, g->area, ks6, qs_buf);
                for (int k = 0; k < sc_count; k++) { double v = hs_idx[k] + ddt * (rk.c1 * fs[k] + rk.c3 * ks3[k] + rk.c4 * ks4[k] + rk.c6 * ks6[k]); hs_temp[k] = v < 0 ? 0 : v; }
                { double *qsum = G_calloc(sc_count, sizeof(double)); for (int k = 0; k < sc_count; k++) for (int l = 0; l < RRI_LMAX8; l++) qsum[k] += qs_buf[l][k]; for (int k = 0; k < sc_count; k++) qs_ave_temp[k] += qsum[k] * ddt; G_free(qsum); }

                errmax = -DBL_MAX;
                for (int k = 0; k < sc_count; k++) {
                    hs_err[k] = ddt * (rk.dc1 * fs[k] + rk.dc3 * ks3[k] + rk.dc4 * ks4[k] + rk.dc5 * ks5[k] + rk.dc6 * ks6[k]);
                    double he = (sc->domain[k] == 0) ? 0.0 : hs_err[k];
                    if (he > errmax) errmax = he;
                }
                errmax /= rk.eps;

                if (!(errmax > 1.0 && ddt > rk.ddt_min_slo)) break;
                double s1 = rk.safety * ddt * pow(errmax, rk.pshrnk), s2 = 0.5 * ddt;
                ddt = s1 > s2 ? s1 : s2;
                if (ddt < rk.ddt_min_slo) ddt = rk.ddt_min_slo;
                if (ddt == 0) G_fatal_error(_("run_rk45_loop: stepsize underflow (slo)"));
            }
            if (time + ddt > t * dt) ddt = t * dt - time;
            time += ddt;
            memcpy(hs_idx, hs_temp, sizeof(double) * sc_count);
            for (int k = 0; k < sc_count; k++) qs_ave_idx[k] += qs_ave_temp[k];
            G_free(qs_ave_temp);

            for (int k = 0; k < sc_count; k++) rain_sum += qp_t_idx[k] * g->area * ddt;
        }
        for (int k = 0; k < sc_count; k++) qs_ave_idx[k] /= (dt * 6.0);
        G_free(qs_ave_idx);

        /* No groundwater sub-loop in this increment -- see function doc. */

        rri_slo_idx2ij(sc, hs_idx, ny, nx, hs);
        rri_slo_idx2ij(sc, gampt_ff_idx, ny, nx, gampt_ff);

        /* ---- RIVER<->SLOPE EXCHANGE + INFILTRATION, once per outer
         * timestep -------------------------------------------------- */
        rri_funcrs(g, rc, dt, hr, hs);
        rri_riv_ij2idx(rc, hr, nx, hr_idx);
        rri_slo_ij2idx(sc, hs, nx, hs_idx);

        rri_infilt(sc, dt, hs_idx, gampt_ff_idx, gampt_f_idx);
        rri_slo_idx2ij(sc, hs_idx, ny, nx, hs);
        rri_slo_idx2ij(sc, gampt_ff_idx, ny, nx, gampt_ff);

        /* ---- OUTLET DRAIN ------------------------------------------- */
        for (int i = 0; i < ny; i++) {
            for (int j = 0; j < nx; j++) {
                size_t p = (size_t)i * nx + j;
                if (g->domain[p] != 2) continue;
                sout += hs[p] * g->area;
                hs[p] = 0.0;
                if (g->riv[p] == 1) {
                    int k = -1;
                    for (int kk = 0; kk < rc_count; kk++) if (rc->idx2i[kk] == i && rc->idx2j[kk] == j) { k = kk; break; }
                    if (k >= 0) sout += rri_hr2vr(hr[p], g->area, rc->area_ratio[k]);
                    hr[p] = 0.0;
                }
            }
        }

        rri_storage s = rri_storage_calc(g, hs, hr, hg, gampt_ff, sc, rc, riv_thresh);
        double storage = s.ss + s.sr + s.si + s.sg;
        double balance = rain_sum - sout - storage;
        G_message(_("t=%d/%d time=%.0f rain_sum=%.6e sout=%.6e storage=%.6e "
                     "balance=%.6e ss=%.6e sr=%.6e si=%.6e sg=%.6e"),
                   t, maxt, time, rain_sum, sout, storage, balance,
                   s.ss, s.sr, s.si, s.sg);

        /* Increment 5 output collection -- see rri_output_sink's doc.
         * Outlet discharge = qr_ave_grid summed over domain==2 (outlet)
         * river cells; qr_ave_grid already holds this timestep's
         * RK45-averaged flux, computed before the outlet-drain step
         * above zeroed hr/hs there, so reading it afterward is safe
         * (same quantity the vendored engine's own hydro.txt writer
         * uses at its hydro_i/hydro_j station cells). */
        if (out) {
            double q_outlet = 0.0;
            for (int i = 0; i < ny; i++)
                for (int j = 0; j < nx; j++) {
                    size_t p = (size_t)i * nx + j;
                    if (g->domain[p] == 2 && g->riv[p] == 1) q_outlet += qr_ave_grid[p];
                }
            output_sink_record_hydro(out, time, q_outlet);
            output_sink_maybe_write_hs(out, t, time, ny, nx, hs);
        }
    }

    G_message(_("run_rk45_loop: done"));

    for (int l = 0; l < RRI_LMAX8; l++) G_free(qs_buf[l]);
    G_free(hs); G_free(hr); G_free(hg); G_free(gampt_ff); G_free(qr_ave_grid);
    G_free(hr_idx); G_free(vr_idx); G_free(qr_idx); G_free(qr_ave_idx); G_free(qr_sum_scratch);
    G_free(fr); G_free(kr2); G_free(kr3); G_free(kr4); G_free(kr5); G_free(kr6);
    G_free(vr_temp); G_free(vr_err);
    G_free(hs_idx); G_free(gampt_ff_idx); G_free(gampt_f_idx); G_free(qp_t_idx);
    G_free(fs); G_free(ks2); G_free(ks3); G_free(ks4); G_free(ks5); G_free(ks6);
    G_free(hs_temp); G_free(hs_err);
}

int main(int argc, char *argv[])
{
    struct GModule *module;
    struct {
        struct Option *elevation, *drainage, *accumulation, *rain, *rain_strds, *rain_units;
        struct Option *landuse;
        struct Option *riv_thresh, *width_c, *width_s, *depth_c, *depth_s,
            *height_param, *height_limit;
        struct Option *ns_river, *ns_slope, *soildepth, *gammaa, *ksv,
            *faif, *ka, *gammam, *beta;
        struct Option *utm;
        struct Option *lasth, *dt, *dt_riv;
        struct Option *hs_output, *hs_interval, *hydrograph_table;
        struct Option *watershed_threshold;
    } opt;
    struct Flag *eight_dir_flag, *run_flag;

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
    opt.drainage->required = NO;
    opt.drainage->description =
        _("D8 flow direction, r.watershed/r.watershed.opencl 'drainage' "
          "convention (counter-clockwise from north-east=1, negative at "
          "domain edges). Omit to auto-derive via r.watershed.opencl "
          "(watershed_threshold= controls its threshold=; see that "
          "option's description) -- pass this explicitly instead when "
          "you already have one (e.g. from a prior r.watershed.opencl "
          "run reused across experiments, mirroring r.hydro.hbv.basins' "
          "drainage_input=-style passthrough pattern) to skip "
          "recomputing it, which can be the difference between minutes "
          "and over an hour on a basin-scale DEM.");

    opt.accumulation = G_define_standard_option(G_OPT_R_INPUT);
    opt.accumulation->key = "accumulation";
    opt.accumulation->required = NO;
    opt.accumulation->description =
        _("Flow accumulation (cell count), r.watershed/r.watershed.opencl "
          "convention. Magnitude only is used (sign, where present, is "
          "r.watershed's own per-cell reliability flag, not part of RRI's "
          "model -- see drainage_to_rri_dir's neighboring comment for the "
          "equivalent caveat on 'drainage'). Omit to auto-derive via "
          "r.watershed.opencl alongside drainage= -- see that option's "
          "description.");

    opt.watershed_threshold = G_define_option();
    opt.watershed_threshold->key = "watershed_threshold";
    opt.watershed_threshold->type = TYPE_INTEGER;
    opt.watershed_threshold->answer = "1";
    opt.watershed_threshold->description =
        _("r.watershed.opencl's own threshold= (minimum exterior basin "
          "size, cells), used only when auto-deriving drainage=/"
          "accumulation= (i.e. when at least one of them is omitted). "
          "This module never reads r.watershed.opencl's stream=/basin= "
          "outputs, so the default of 1 (finest-grained, no effect on "
          "drainage/accumulation themselves) is safe -- raise it only if "
          "you separately care about r.watershed.opencl's basin "
          "delineation on the SAME run for some other purpose.");

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

    /* Each of these 8 is PER-LANDUSE-CLASS: with landuse=, pass exactly
     * num_of_landuse (the raster's max category value) comma-separated
     * values, in category order 1..N -- e.g. ns_slope=0.4,0.1,0.02 for
     * 3 classes. A single value applies uniformly to every class
     * (including the landuse=-omitted, single-class default case).
     * See README "Land use / land cover" for a worked example. */
    opt.ns_slope = G_define_option();
    opt.ns_slope->key = "ns_slope";
    opt.ns_slope->type = TYPE_DOUBLE;
    opt.ns_slope->multiple = YES;
    opt.ns_slope->answer = "0.4";
    opt.ns_slope->description = _("Hillslope Manning's roughness coefficient, one per landuse= class (or one value for all)");

    opt.soildepth = G_define_option();
    opt.soildepth->key = "soildepth";
    opt.soildepth->type = TYPE_DOUBLE;
    opt.soildepth->multiple = YES;
    opt.soildepth->answer = "1.0";
    opt.soildepth->description = _("Soil depth [m], one per landuse= class (or one value for all)");

    opt.gammaa = G_define_option();
    opt.gammaa->key = "gammaa";
    opt.gammaa->type = TYPE_DOUBLE;
    opt.gammaa->multiple = YES;
    opt.gammaa->answer = "0.475";
    opt.gammaa->description = _("Soil porosity [-], one per landuse= class (or one value for all)");

    opt.ksv = G_define_option();
    opt.ksv->key = "ksv";
    opt.ksv->type = TYPE_DOUBLE;
    opt.ksv->multiple = YES;
    opt.ksv->answer = "0.0";
    opt.ksv->description = _("Green-Ampt vertical saturated hydraulic conductivity [m/s] (0 disables), one per landuse= class (or one value for all)");

    opt.faif = G_define_option();
    opt.faif->key = "faif";
    opt.faif->type = TYPE_DOUBLE;
    opt.faif->multiple = YES;
    opt.faif->answer = "0.316";
    opt.faif->description = _("Green-Ampt wetting front suction head [m], one per landuse= class (or one value for all)");

    opt.ka = G_define_option();
    opt.ka->key = "ka";
    opt.ka->type = TYPE_DOUBLE;
    opt.ka->multiple = YES;
    opt.ka->answer = "0.0";
    opt.ka->description = _("Lateral subsurface (Darcy) conductivity [m/s] (0 disables; mutually exclusive with ksv), one per landuse= class (or one value for all)");

    opt.gammam = G_define_option();
    opt.gammam->key = "gammam";
    opt.gammam->type = TYPE_DOUBLE;
    opt.gammam->multiple = YES;
    opt.gammam->answer = "0.0";
    opt.gammam->description = _("Matrix-flow porosity [-], one per landuse= class (or one value for all)");

    opt.beta = G_define_option();
    opt.beta->key = "beta";
    opt.beta->type = TYPE_DOUBLE;
    opt.beta->multiple = YES;
    opt.beta->answer = "8.0";
    opt.beta->description = _("Hillslope subsurface flow power-law exponent, one per landuse= class (or one value for all)");

    opt.landuse = G_define_standard_option(G_OPT_R_INPUT);
    opt.landuse->key = "landuse";
    opt.landuse->required = NO;
    opt.landuse->description =
        _("Classified land-use/land-cover raster. The raster's CELL "
          "VALUES (integer categories, read via Rast_get_c_row) are what "
          "matters -- category LABEL text (r.category/r.support) is "
          "metadata the physics never reads. Categories must be "
          "1..num_of_landuse (this module takes num_of_landuse = the "
          "raster's own max category value), each representing a "
          "distinct combination of ns_slope=/soildepth=/gammaa=/etc "
          "(a hydraulic-parameter class, not a semantic land-cover label "
          "GRASS has any opinion about). Omit for a single uniform class "
          "over the whole domain. See README for a worked r.reclass "
          "example building a 1..N raster from a real classified source "
          "(MODIS MCD12Q1, ESA WorldCover, ...). Time-varying land use "
          "(a landuse_strds= option) is not implemented -- see "
          "NATIVE_GRASS_PLAN.md.");

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
        _("Precipitation intensity, a SINGLE static raster applied as "
          "constant forcing for the whole run. Units per rain_units= "
          "(default mm_per_day, converted internally -- see rain_units' "
          "own description for why that default, not mm_per_hour, is "
          "the safe one).");

    opt.rain_strds = G_define_standard_option(G_OPT_STRDS_INPUT);
    opt.rain_strds->key = "rain_strds";
    opt.rain_strds->required = NO;
    opt.rain_strds->description =
        _("Precipitation space-time raster dataset, e.g. t.in.era5's "
          "<prefix>_precipitation. Units per rain_units=. Resolves and "
          "iterates the series (see resolve_strds_steps), feeding the RK45 "
          "time loop with -r. Mutually exclusive with rain= in practice "
          "(rain_strds= takes priority if both given).");

    opt.rain_units = G_define_option();
    opt.rain_units->key = "rain_units";
    opt.rain_units->type = TYPE_STRING;
    opt.rain_units->options = "mm_per_day,mm_per_hour";
    opt.rain_units->answer = "mm_per_day";
    opt.rain_units->description =
        _("Units of rain=/rain_strds='s raster cell values. Default "
          "mm_per_day matches t.in.era5's ONE AND ONLY output convention "
          "for precipitation/potential_evaporation (confirmed against "
          "t.in.era5.py directly: both are registered as daily sums in "
          "mm, 'Total precipitation, daily sum (mm/d)' -- t.in.era5 has "
          "no hourly-output mode, so a STRDS from it is always mm/day, "
          "never a maybe). mm_per_day is divided by 24 to a uniform "
          "hourly rate before reaching the physics kernels -- this is a "
          "real simplification (a daily total cannot recover sub-daily "
          "rainfall intensity variation, only its correct daily mean "
          "rate), not a unit-only conversion; document this when citing "
          "results from era5-driven runs. Defaulting to mm_per_hour "
          "instead would have been the wrong default for this module's "
          "primary intended forcing source and risks a silent 24x error "
          "for anyone who feeds t.in.era5 output through without "
          "noticing -- exactly the mixup this module's own test suite "
          "hit once already (see NATIVE_GRASS_PLAN.md).");

    opt.lasth = G_define_option();
    opt.lasth->key = "lasth";
    opt.lasth->type = TYPE_DOUBLE;
    opt.lasth->required = NO;
    opt.lasth->description =
        _("Simulation length [hours]. Required together with -r to "
          "actually run the RK45 time loop -- without it, rain=/"
          "rain_strds= only run their own isolated read/index diagnostic "
          "(increments 2/3), same as if -r were omitted.");

    opt.dt = G_define_option();
    opt.dt->key = "dt";
    opt.dt->type = TYPE_DOUBLE;
    opt.dt->answer = "600";
    opt.dt->description = _("Outer (slope) timestep [s]");

    opt.dt_riv = G_define_option();
    opt.dt_riv->key = "dt_riv";
    opt.dt_riv->type = TYPE_DOUBLE;
    opt.dt_riv->answer = "60";
    opt.dt_riv->description = _("Initial river RK45 sub-timestep [s]");

    eight_dir_flag = G_define_flag();
    eight_dir_flag->key = 'e';
    eight_dir_flag->description = _("8-direction hillslope routing (default: 4-direction)");

    opt.hs_output = G_define_standard_option(G_OPT_STRDS_OUTPUT);
    opt.hs_output->key = "hs_output";
    opt.hs_output->required = NO;
    opt.hs_output->description =
        _("With -r: name for an output space-time raster dataset (STRDS) "
          "of hillslope water depth [m], one map per hs_interval outer "
          "timesteps. Omit to skip periodic state output entirely.");

    opt.hs_interval = G_define_option();
    opt.hs_interval->key = "hs_interval";
    opt.hs_interval->type = TYPE_INTEGER;
    opt.hs_interval->answer = "1";
    opt.hs_interval->description =
        _("Write an hs_output= snapshot every N outer timesteps (1 = every timestep)");

    opt.hydrograph_table = G_define_option();
    opt.hydrograph_table->key = "hydrograph_table";
    opt.hydrograph_table->type = TYPE_STRING;
    opt.hydrograph_table->required = NO;
    opt.hydrograph_table->description =
        _("With -r: name for an output DB table (time_s, discharge_cms) "
          "of discharge summed over the domain's outlet river cell(s). "
          "Omit to skip.");

    run_flag = G_define_flag();
    run_flag->key = 'r';
    run_flag->description =
        _("Actually run the adaptive RK45 time loop (requires lasth= and "
          "rain= or rain_strds=). Increment 4 (NATIVE_GRASS_PLAN.md "
          "'Progress'): prints per-outer-timestep mass-balance "
          "diagnostics to stderr for cross-checking against RRI.opencl's "
          "ASCII-path storage.dat on the same domain -- does NOT yet "
          "write GRASS output (DB table / STRDS), that is the next "
          "increment.");

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

    char drainage_name[GNAME_MAX], accum_name[GNAME_MAX];
    int auto_derived = !opt.drainage->answer || !opt.accumulation->answer;
    if (auto_derived) {
        auto_derive_drainage_accumulation(opt.elevation->answer,
                                           atoi(opt.watershed_threshold->answer),
                                           drainage_name, accum_name);
    } else {
        snprintf(drainage_name, sizeof(drainage_name), "%s", opt.drainage->answer);
        snprintf(accum_name, sizeof(accum_name), "%s", opt.accumulation->answer);
    }

    mapset = G_find_raster2(drainage_name, "");
    if (!mapset) G_fatal_error(_("Raster map <%s> not found"), drainage_name);
    int fd_dir = Rast_open_old(drainage_name, mapset);
    DCELL *dir_row = G_malloc(nx * sizeof(DCELL));
    int *dir_rri = G_malloc(ncell * sizeof(int));
    for (int row = 0; row < ny; row++) {
        Rast_get_d_row(fd_dir, dir_row, row);
        for (int col = 0; col < nx; col++)
            dir_rri[(size_t)row * nx + col] = drainage_to_rri_dir(dir_row[col]);
    }
    Rast_close(fd_dir);
    G_free(dir_row);

    mapset = G_find_raster2(accum_name, "");
    if (!mapset) G_fatal_error(_("Raster map <%s> not found"), accum_name);
    int fd_acc = Rast_open_old(accum_name, mapset);
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

    /* LULC: landuse= raster's own cell VALUES (categories), read via
     * Rast_get_c_row -- NOT a category label from r.category/r.support,
     * which is metadata the physics never reads. Categories must be
     * 1..num_of_landuse contiguous, matching RRI's own per-landuse
     * parameter-array indexing; num_of_landuse is taken as the raster's
     * own maximum category value. Omit landuse= for a single uniform
     * class covering the whole domain (num_of_landuse=1, land[]=1
     * everywhere) -- see README "Land use / land cover" for a worked
     * r.reclass example (MODIS/WorldCover -> a 1..N class raster). */
    int num_of_landuse = 1;
    if (opt.landuse->answer) {
        const char *lu_mapset = G_find_raster2(opt.landuse->answer, "");
        if (!lu_mapset) G_fatal_error(_("Raster map <%s> not found"), opt.landuse->answer);
        int fd_lu = Rast_open_old(opt.landuse->answer, lu_mapset);
        CELL *lu_row = G_malloc(nx * sizeof(CELL));
        int max_cat = 1;
        for (int row = 0; row < ny; row++) {
            Rast_get_c_row(fd_lu, lu_row, row);
            for (int col = 0; col < nx; col++) {
                int v = Rast_is_c_null_value(&lu_row[col]) ? 1 : lu_row[col];
                if (v < 1) G_fatal_error(_("landuse=<%s> has a category value %d < 1 at "
                                            "row %d col %d -- categories must be 1..N"),
                                          opt.landuse->answer, v, row, col);
                g.land[(size_t)row * nx + col] = v;
                if (v > max_cat) max_cat = v;
            }
        }
        Rast_close(fd_lu);
        G_free(lu_row);
        num_of_landuse = max_cat;
        G_message(_("landuse=<%s>: %d class(es) (max category value)"),
                   opt.landuse->answer, num_of_landuse);
    } else {
        for (size_t p = 0; p < ncell; p++) g.land[p] = 1;
    }

    rri_landuse lu;
    set_landuse_from_options(&lu, num_of_landuse,
                              opt.ns_slope, opt.soildepth, opt.gammaa, opt.ksv,
                              opt.faif, opt.ka, opt.gammam, opt.beta);

    for (size_t p = 0; p < ncell; p++) {
        if (zs[p] > -100.0)
            g.domain[p] = (g.dir[p] == 0) ? 2 : 1;
        g.zb[p] = zs[p] - lu.soildepth[g.land[p] - 1];
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

    rri_riv_cellset rc;
    rri_slo_cellset sc;
    memset(&rc, 0, sizeof(rc));
    memset(&sc, 0, sizeof(sc));

    if (rri_riv_idx_setting(&g, &rc) != 0)
        G_fatal_error(_("rri_riv_idx_setting failed -- see stderr diagnostic above"));
    if (rri_slo_idx_setting(&g, &lu, eight_dir_flag->answer ? 1 : 0, &sc) != 0)
        G_fatal_error(_("rri_slo_idx_setting failed -- see stderr diagnostic above"));

    G_message(_("riv_count=%d slo_count=%d"), rc.count, sc.count);

    /* Forcing input, increment 2 (single raster) and increment 3 (STRDS
     * time series) -- see NATIVE_GRASS_PLAN.md "Progress". Neither is
     * wired into a time loop yet (it doesn't exist yet either); both
     * only prove the read+convert+index path is correct, via a
     * checkable diagnostic, in isolation and (for the STRDS case) across
     * a whole resolved series. */
    rri_forcing_preloaded preloaded = {0, NULL, NULL};

    if (opt.rain_strds->answer) {
        rri_forcing_step *steps = NULL;
        int n_steps = resolve_strds_steps(opt.rain_strds->answer, &steps);
        if (n_steps < 0)
            G_fatal_error(_("rain_strds=<%s> has no registered raster maps"),
                           opt.rain_strds->answer);
        G_message(_("rain_strds: resolved %d timestep(s)"), n_steps);

        preloaded.n = n_steps;
        preloaded.elapsed_s = G_malloc(n_steps * sizeof(double));
        preloaded.qp_t_idx = G_malloc(n_steps * sizeof(double *));
        for (int i = 0; i < n_steps; i++) {
            preloaded.qp_t_idx[i] = G_malloc(sc.count * sizeof(double));
            read_and_index_forcing_raster(steps[i].name, ny, nx, &sc, preloaded.qp_t_idx[i]);
            preloaded.elapsed_s[i] = steps[i].elapsed_s;
            double sum = 0.0;
            for (int k = 0; k < sc.count; k++) sum += preloaded.qp_t_idx[i][k];
            G_message(_("rain_strds[%d]: map=%s elapsed_s=%.0f mean qp_t_idx=%.6f mm/h"),
                       i, steps[i].name, steps[i].elapsed_s,
                       sc.count ? sum / sc.count : 0.0);
        }
        G_free(steps);
    } else if (opt.rain->answer) {
        double *qp_t_idx = G_malloc(sc.count * sizeof(double));
        read_and_index_forcing_raster(opt.rain->answer, ny, nx, &sc, qp_t_idx);

        /* Diagnostic: mean of qp_t_idx over active slope cells should
         * equal the raster's own mean over in-domain cells (a uniform
         * input raster makes this trivially checkable by eye -- see
         * tests/native_io_test.py's rain cross-check for the automated
         * version of this same idea). */
        double sum = 0.0;
        for (int k = 0; k < sc.count; k++) sum += qp_t_idx[k];
        G_message(_("rain: mean qp_t_idx over %d active slope cells = %.6f mm/h"),
                   sc.count, sc.count ? sum / sc.count : 0.0);

        /* A single static raster is constant forcing for the whole run:
         * one preloaded "step" whose elapsed_s is set past the run's
         * own end, so run_rk45_loop's bracket search always selects it
         * (see that function's rri_forcing_preloaded doc). */
        preloaded.n = 1;
        preloaded.elapsed_s = G_malloc(sizeof(double));
        preloaded.elapsed_s[0] = (opt.lasth->answer ? atof(opt.lasth->answer) : 1.0) * 3600.0 + 1.0;
        preloaded.qp_t_idx = G_malloc(sizeof(double *));
        preloaded.qp_t_idx[0] = qp_t_idx;
    }

    /* mm/day -> mm/h, applied to the PRELOADED arrays only (what
     * run_rk45_loop actually consumes), never to the raw values the
     * rain/rain_strds diagnostics above already printed and that
     * tests/native_io_test.py checks exactly -- read_and_index_forcing_
     * raster's contract (returns the raster's own raw cell values) does
     * not change. See opt.rain_units' description for why mm_per_day is
     * the default and what physical simplification it represents. */
    if (preloaded.n > 0 && strcmp(opt.rain_units->answer, "mm_per_day") == 0) {
        for (int i = 0; i < preloaded.n; i++)
            for (int k = 0; k < sc.count; k++)
                preloaded.qp_t_idx[i][k] /= 24.0;
    }

    if (run_flag->answer) {
        if (!opt.lasth->answer)
            G_fatal_error(_("-r requires lasth="));
        if (preloaded.n == 0)
            G_fatal_error(_("-r requires rain= or rain_strds="));

        rri_output_sink sink;
        output_sink_init(&sink, opt.hs_output->answer, atoi(opt.hs_interval->answer));

        run_rk45_loop(&g, &rc, &sc, atof(opt.dt->answer), atof(opt.dt_riv->answer),
                       atof(opt.lasth->answer), (int)riv_thresh, atof(opt.ns_river->answer),
                       &preloaded, &sink);

        /* t0 for STRDS timestamps: "now" is as reasonable a default as
         * any absolute start time, since neither rain_strds= nor rain=
         * carries a real-world simulation start date into this
         * increment (rain_strds='s own resolved steps are timestamped
         * relative to ITS first map's start, not necessarily "now" --
         * a real deployment would want the STRDS's own start_time
         * threaded through here instead; not done this pass, flagged in
         * NATIVE_GRASS_PLAN.md). */
        time_t t0 = time(NULL);
        if (opt.hs_output->answer) register_hs_strds(&sink, opt.hs_output->answer, t0);
        if (opt.hydrograph_table->answer)
            write_hydrograph_table(&sink, opt.hydrograph_table->answer);
    }

    rri_riv_cellset_free(&rc);
    rri_slo_cellset_free(&sc);

    if (auto_derived) {
        char cmd[512];
        snprintf(cmd, sizeof(cmd), "g.remove -f type=raster name=\"%s,%s\" --quiet",
                 drainage_name, accum_name);
        system(cmd); /* best-effort cleanup of the auto-derived temp rasters; not fatal if it fails */
    }

    return EXIT_SUCCESS;
}
