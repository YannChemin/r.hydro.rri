/**
 * @file rri_io.c
 * @brief Input plumbing: ESRI ASCII grid (topography/landuse/etc.)
 * reading, geodesic distance for lat/lon domains, and RRI_Input.txt
 * config-file parsing.
 *
 * No hydrology happens here -- this is the "get the numbers off disk and
 * into memory, validated" layer everything else in the solver builds on.
 * It runs once at startup (src/main.c's setup phase), never inside the
 * time loop, so unlike the physics kernels it has no OpenMP/OpenCL
 * concerns and is written straightforwardly rather than for parallelism.
 *
 * Fortran reference: RRI_Sub.f90 (`read_gis_int`/`read_gis_real`,
 * `hubeny_sub`), RRI_Read.f90 (the whole file: `RRI_Read` subroutine).
 */
#include "rri/rri.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* ---- small line-oriented reader, tolerant of CRLF and comma-separated
 * numeric rows (some real-world grids, e.g. solo30s's "_mod" files, use
 * commas; others use plain whitespace -- accept both). ---------------- */

typedef struct {
    FILE *f;
    char line[1 << 20]; /* solo30s rows are ~336 numbers; this generously covers larger grids too */
} rri_reader;

static int reader_open(rri_reader *r, const char *path)
{
    r->f = fopen(path, "r");
    return r->f ? 0 : -1;
}

static void reader_close(rri_reader *r)
{
    if (r->f) fclose(r->f);
    r->f = NULL;
}

/* Reads the next line, strips trailing \r\n, returns 0 on success. */
static int reader_getline(rri_reader *r)
{
    if (!fgets(r->line, (int)sizeof(r->line), r->f)) return -1;
    size_t n = strlen(r->line);
    while (n > 0 && (r->line[n - 1] == '\n' || r->line[n - 1] == '\r')) {
        r->line[--n] = '\0';
    }
    return 0;
}

/* Trims leading/trailing whitespace in place, returns pointer into the
 * same buffer (Fortran trim(adjustl(...))). */
static char *rtrim_ltrim(char *s)
{
    while (*s == ' ' || *s == '\t') s++;
    size_t n = strlen(s);
    while (n > 0 && (s[n - 1] == ' ' || s[n - 1] == '\t')) s[--n] = '\0';
    return s;
}

static const char *TOK_DELIMS = " \t,";

/* ---- ESRI ASCII grid I/O -------------------------------------------- */

int rri_read_gis_header(const char *path, int *ny, int *nx, double *xllcorner,
                         double *yllcorner, double *cellsize)
{
    rri_reader r;
    char key[64];
    if (reader_open(&r, path) != 0) {
        fprintf(stderr, "rri_read_gis_header: cannot open %s\n", path);
        return -1;
    }
    double vals[5];
    for (int i = 0; i < 5; i++) {
        if (reader_getline(&r) != 0) { reader_close(&r); return -1; }
        if (sscanf(r.line, "%63s %lf", key, &vals[i]) != 2) { reader_close(&r); return -1; }
    }
    reader_close(&r);
    *nx = (int)vals[0];
    *ny = (int)vals[1];
    *xllcorner = vals[2];
    *yllcorner = vals[3];
    *cellsize = vals[4];
    return 0;
}

/* Shared header-check logic for read_gis_real/read_gis_int. */
static int read_gis_header_checked(rri_reader *r, const char *path, int ny, int nx,
                                    double xllcorner, double yllcorner, double cellsize)
{
    char key[64];
    double v;
    int itemp;

    if (reader_getline(r) != 0 || sscanf(r->line, "%63s %d", key, &itemp) != 2 || itemp != nx) {
        fprintf(stderr, "error in gis input data: ncols mismatch in %s\n", path);
        return -1;
    }
    if (reader_getline(r) != 0 || sscanf(r->line, "%63s %d", key, &itemp) != 2 || itemp != ny) {
        fprintf(stderr, "error in gis input data: nrows mismatch in %s\n", path);
        return -1;
    }
    if (reader_getline(r) != 0 || sscanf(r->line, "%63s %lf", key, &v) != 2 || fabs(v - xllcorner) > 0.01) {
        fprintf(stderr, "error in gis input data: xllcorner mismatch in %s\n", path);
        return -1;
    }
    if (reader_getline(r) != 0 || sscanf(r->line, "%63s %lf", key, &v) != 2 || fabs(v - yllcorner) > 0.01) {
        fprintf(stderr, "error in gis input data: yllcorner mismatch in %s\n", path);
        return -1;
    }
    if (reader_getline(r) != 0 || sscanf(r->line, "%63s %lf", key, &v) != 2 || fabs(v - cellsize) > 0.01) {
        fprintf(stderr, "error in gis input data: cellsize mismatch in %s\n", path);
        return -1;
    }
    if (reader_getline(r) != 0) return -1; /* NODATA_value line, unused */
    return 0;
}

int rri_read_gis_real(const char *path, int ny, int nx, double xllcorner,
                       double yllcorner, double cellsize, double *out)
{
    rri_reader r;
    if (reader_open(&r, path) != 0) {
        fprintf(stderr, "rri_read_gis_real: cannot open %s\n", path);
        return -1;
    }
    if (read_gis_header_checked(&r, path, ny, nx, xllcorner, yllcorner, cellsize) != 0) {
        reader_close(&r);
        return -1;
    }
    for (int i = 0; i < ny; i++) {
        if (reader_getline(&r) != 0) { reader_close(&r); return -1; }
        char *tok = strtok(r.line, TOK_DELIMS);
        for (int j = 0; j < nx; j++) {
            if (!tok) { fprintf(stderr, "rri_read_gis_real: short row %d in %s\n", i, path); reader_close(&r); return -1; }
            out[(size_t)i * nx + j] = atof(tok);
            tok = strtok(NULL, TOK_DELIMS);
        }
    }
    reader_close(&r);
    return 0;
}

int rri_read_gis_int(const char *path, int ny, int nx, double xllcorner,
                      double yllcorner, double cellsize, int *out)
{
    rri_reader r;
    if (reader_open(&r, path) != 0) {
        fprintf(stderr, "rri_read_gis_int: cannot open %s\n", path);
        return -1;
    }
    if (read_gis_header_checked(&r, path, ny, nx, xllcorner, yllcorner, cellsize) != 0) {
        reader_close(&r);
        return -1;
    }
    for (int i = 0; i < ny; i++) {
        if (reader_getline(&r) != 0) { reader_close(&r); return -1; }
        char *tok = strtok(r.line, TOK_DELIMS);
        for (int j = 0; j < nx; j++) {
            if (!tok) { fprintf(stderr, "rri_read_gis_int: short row %d in %s\n", i, path); reader_close(&r); return -1; }
            out[(size_t)i * nx + j] = (int)atof(tok); /* atof, not atoi: grids may write "-9999.0" */
            tok = strtok(NULL, TOK_DELIMS);
        }
    }
    reader_close(&r);
    return 0;
}

/* ---- geodesic distance (RRI_Sub.f90: hubeny_sub, WGS84) ------------- */

double rri_hubeny_sub(double x1_deg, double y1_deg, double x2_deg, double y2_deg)
{
    const double pi = 3.1415926535897;
    double x1 = x1_deg * pi / 180.0;
    double y1 = y1_deg * pi / 180.0;
    double x2 = x2_deg * pi / 180.0;
    double y2 = y2_deg * pi / 180.0;

    double dy = y1 - y2;
    double dx = x1 - x2;
    double mu = (y1 + y2) / 2.0;

    const double a = 6378137.0000;
    const double b = 6356752.3140;

    double e = sqrt((a * a - b * b) / (a * a));
    double W = sqrt(1.0 - e * e * sin(mu) * sin(mu));
    double N = a / W;
    double M = a * (1.0 - e * e) / (W * W * W);

    return sqrt((dy * M) * (dy * M) + (dx * N * cos(mu)) * (dx * N * cos(mu)));
}

/* ---- RRI_Input.txt config parsing (RRI_Read.f90 field order) -------
 *
 * RRI_Input.txt alternates value lines with blank "separator" lines
 * (Fortran's `read(1,*)` with no variable list, used purely to skip a
 * record). next_nonblank_line() plays that role here: it skips blank
 * lines automatically so callers don't have to explicitly consume the
 * separators one-for-one with the Fortran source -- but the VALUE lines
 * themselves must still be read in exactly RRI_Read.f90's field order
 * (rri_config_read below), since nothing here validates that a value
 * line "means" what its position implies; a config with fields
 * transposed would parse without error and silently populate the wrong
 * struct members. ------------------------------------------------- */

static int next_nonblank_line(rri_reader *r, char *out, size_t outsz)
{
    for (;;) {
        if (reader_getline(r) != 0) return -1;
        char *t = rtrim_ltrim(r->line);
        if (*t != '\0') { strncpy(out, r->line, outsz - 1); out[outsz - 1] = '\0'; return 0; }
    }
}

/* Reads a value line, strips a trailing "# comment" first. */
static int value_line(rri_reader *r, char *out, size_t outsz)
{
    if (next_nonblank_line(r, out, outsz) != 0) return -1;
    char *hash = strchr(out, '#');
    if (hash) *hash = '\0';
    return 0;
}

static int read_path(rri_reader *r, char *dst, size_t dstsz)
{
    char buf[512];
    if (next_nonblank_line(r, buf, sizeof(buf)) != 0) return -1;
    char *t = rtrim_ltrim(buf);
    strncpy(dst, t, dstsz - 1);
    dst[dstsz - 1] = '\0';
    return 0;
}

static int read_ints(rri_reader *r, int *vals, int n)
{
    char buf[512];
    if (value_line(r, buf, sizeof(buf)) != 0) return -1;
    char *tok = strtok(buf, TOK_DELIMS);
    for (int i = 0; i < n; i++) {
        if (!tok) return -1;
        vals[i] = atoi(tok);
        tok = strtok(NULL, TOK_DELIMS);
    }
    return 0;
}

static int read_doubles(rri_reader *r, double *vals, int n)
{
    char buf[512];
    if (value_line(r, buf, sizeof(buf)) != 0) return -1;
    char *tok = strtok(buf, TOK_DELIMS);
    for (int i = 0; i < n; i++) {
        if (!tok) return -1;
        vals[i] = atof(tok);
        tok = strtok(NULL, TOK_DELIMS);
    }
    return 0;
}

static double *alloc_doubles(int n) { return (double *)calloc((size_t)n, sizeof(double)); }
static int *alloc_ints(int n) { return (int *)calloc((size_t)n, sizeof(int)); }

int rri_config_read(const char *path, rri_config *cfg)
{
    memset(cfg, 0, sizeof(*cfg));
    rri_reader r;
    if (reader_open(&r, path) != 0) {
        fprintf(stderr, "rri_config_read: cannot open %s\n", path);
        return -1;
    }

    char buf[512];
    if (next_nonblank_line(&r, buf, sizeof(buf)) != 0) goto fail;
    if (strcmp(rtrim_ltrim(buf), "RRI_Input_Format_Ver1_4_2") != 0) {
        fprintf(stderr, "rri_config_read: unsupported format version '%s'\n", rtrim_ltrim(buf));
        goto fail;
    }

    if (read_path(&r, cfg->rainfile, sizeof(cfg->rainfile)) != 0) goto fail;
    if (read_path(&r, cfg->demfile, sizeof(cfg->demfile)) != 0) goto fail;
    if (read_path(&r, cfg->accfile, sizeof(cfg->accfile)) != 0) goto fail;
    if (read_path(&r, cfg->dirfile, sizeof(cfg->dirfile)) != 0) goto fail;

    { int v[1];
      if (read_ints(&r, v, 1) != 0) goto fail; cfg->utm = v[0];
      if (read_ints(&r, v, 1) != 0) goto fail; cfg->eight_dir = v[0];
      if (read_ints(&r, v, 1) != 0) goto fail; cfg->lasth = v[0];
    }
    { double v[1];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->dt = v[0];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->dt_riv = v[0];
    }
    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->outnum = v[0]; }
    { double v[1];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->xllcorner_rain = v[0];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->yllcorner_rain = v[0];
    }
    { double v[2]; if (read_doubles(&r, v, 2) != 0) goto fail;
      cfg->cellsize_rain_x = v[0]; cfg->cellsize_rain_y = v[1]; }

    { double v[1]; if (read_doubles(&r, v, 1) != 0) goto fail; cfg->ns_river = v[0]; }
    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->num_of_landuse = v[0]; }

    int n = cfg->num_of_landuse;
    rri_landuse *lu = &cfg->lu;
    lu->n = n;
    lu->dif = alloc_ints(n);
    lu->ns_slope = alloc_doubles(n); lu->soildepth = alloc_doubles(n); lu->gammaa = alloc_doubles(n);
    lu->ksv = alloc_doubles(n); lu->faif = alloc_doubles(n);
    lu->ka = alloc_doubles(n); lu->gammam = alloc_doubles(n); lu->beta = alloc_doubles(n);
    lu->ksg = alloc_doubles(n); lu->gammag = alloc_doubles(n); lu->kg0 = alloc_doubles(n);
    lu->fpg = alloc_doubles(n); lu->rgl = alloc_doubles(n);
    lu->da = alloc_doubles(n); lu->dm = alloc_doubles(n); lu->infilt_limit = alloc_doubles(n);

    if (read_ints(&r, lu->dif, n) != 0) goto fail;
    if (read_doubles(&r, lu->ns_slope, n) != 0) goto fail;
    if (read_doubles(&r, lu->soildepth, n) != 0) goto fail;
    if (read_doubles(&r, lu->gammaa, n) != 0) goto fail;
    if (read_doubles(&r, lu->ksv, n) != 0) goto fail;
    if (read_doubles(&r, lu->faif, n) != 0) goto fail;
    if (read_doubles(&r, lu->ka, n) != 0) goto fail;
    if (read_doubles(&r, lu->gammam, n) != 0) goto fail;
    if (read_doubles(&r, lu->beta, n) != 0) goto fail;
    if (read_doubles(&r, lu->ksg, n) != 0) goto fail;
    if (read_doubles(&r, lu->gammag, n) != 0) goto fail;
    if (read_doubles(&r, lu->kg0, n) != 0) goto fail;
    if (read_doubles(&r, lu->fpg, n) != 0) goto fail;
    if (read_doubles(&r, lu->rgl, n) != 0) goto fail;

    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->riv_thresh = v[0]; }
    { double v[1];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->width_param_c = v[0];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->width_param_s = v[0];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->depth_param_c = v[0];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->depth_param_s = v[0];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->height_param = v[0];
    }
    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->height_limit_param = v[0]; }

    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->rivfile_switch = v[0]; }
    if (read_path(&r, cfg->widthfile, sizeof(cfg->widthfile)) != 0) goto fail;
    if (read_path(&r, cfg->depthfile, sizeof(cfg->depthfile)) != 0) goto fail;
    if (read_path(&r, cfg->heightfile, sizeof(cfg->heightfile)) != 0) goto fail;

    { int v[4]; if (read_ints(&r, v, 4) != 0) goto fail;
      cfg->init_slo_switch = v[0]; cfg->init_riv_switch = v[1];
      cfg->init_gw_switch = v[2]; cfg->init_gampt_ff_switch = v[3]; }
    if (read_path(&r, cfg->initfile_slo, sizeof(cfg->initfile_slo)) != 0) goto fail;
    if (read_path(&r, cfg->initfile_riv, sizeof(cfg->initfile_riv)) != 0) goto fail;
    if (read_path(&r, cfg->initfile_gw, sizeof(cfg->initfile_gw)) != 0) goto fail;
    if (read_path(&r, cfg->initfile_gampt_ff, sizeof(cfg->initfile_gampt_ff)) != 0) goto fail;

    { int v[2]; if (read_ints(&r, v, 2) != 0) goto fail;
      cfg->bound_slo_wlev_switch = v[0]; cfg->bound_riv_wlev_switch = v[1]; }
    if (read_path(&r, cfg->boundfile_slo_wlev, sizeof(cfg->boundfile_slo_wlev)) != 0) goto fail;
    if (read_path(&r, cfg->boundfile_riv_wlev, sizeof(cfg->boundfile_riv_wlev)) != 0) goto fail;

    { int v[2]; if (read_ints(&r, v, 2) != 0) goto fail;
      cfg->bound_slo_disc_switch = v[0]; cfg->bound_riv_disc_switch = v[1]; }
    if (read_path(&r, cfg->boundfile_slo_disc, sizeof(cfg->boundfile_slo_disc)) != 0) goto fail;
    if (read_path(&r, cfg->boundfile_riv_disc, sizeof(cfg->boundfile_riv_disc)) != 0) goto fail;

    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->land_switch = v[0]; }
    if (read_path(&r, cfg->landfile, sizeof(cfg->landfile)) != 0) goto fail;

    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->dam_switch = v[0]; }
    if (read_path(&r, cfg->damfile, sizeof(cfg->damfile)) != 0) goto fail;

    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->div_switch = v[0]; }
    if (read_path(&r, cfg->divfile, sizeof(cfg->divfile)) != 0) goto fail;

    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->evp_switch = v[0]; }
    if (read_path(&r, cfg->evpfile, sizeof(cfg->evpfile)) != 0) goto fail;
    { double v[1];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->xllcorner_evp = v[0];
      if (read_doubles(&r, v, 1) != 0) goto fail; cfg->yllcorner_evp = v[0]; }
    { double v[2]; if (read_doubles(&r, v, 2) != 0) goto fail;
      cfg->cellsize_evp_x = v[0]; cfg->cellsize_evp_y = v[1]; }

    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->sec_length_switch = v[0]; }
    if (read_path(&r, cfg->sec_length_file, sizeof(cfg->sec_length_file)) != 0) goto fail;

    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->sec_switch = v[0]; }
    if (read_path(&r, cfg->sec_map_file, sizeof(cfg->sec_map_file)) != 0) goto fail;
    if (read_path(&r, cfg->sec_file, sizeof(cfg->sec_file)) != 0) goto fail;

    { int v[10]; if (read_ints(&r, v, 10) != 0) goto fail;
      cfg->outswitch_hs = v[0]; cfg->outswitch_hr = v[1]; cfg->outswitch_hg = v[2];
      cfg->outswitch_qr = v[3]; cfg->outswitch_qu = v[4]; cfg->outswitch_qv = v[5];
      cfg->outswitch_gu = v[6]; cfg->outswitch_gv = v[7]; cfg->outswitch_gampt_ff = v[8];
      cfg->outswitch_storage = v[9]; }
    if (read_path(&r, cfg->outfile_hs, sizeof(cfg->outfile_hs)) != 0) goto fail;
    if (read_path(&r, cfg->outfile_hr, sizeof(cfg->outfile_hr)) != 0) goto fail;
    if (read_path(&r, cfg->outfile_hg, sizeof(cfg->outfile_hg)) != 0) goto fail;
    if (read_path(&r, cfg->outfile_qr, sizeof(cfg->outfile_qr)) != 0) goto fail;
    if (read_path(&r, cfg->outfile_qu, sizeof(cfg->outfile_qu)) != 0) goto fail;
    if (read_path(&r, cfg->outfile_qv, sizeof(cfg->outfile_qv)) != 0) goto fail;
    if (read_path(&r, cfg->outfile_gu, sizeof(cfg->outfile_gu)) != 0) goto fail;
    if (read_path(&r, cfg->outfile_gv, sizeof(cfg->outfile_gv)) != 0) goto fail;
    if (read_path(&r, cfg->outfile_gampt_ff, sizeof(cfg->outfile_gampt_ff)) != 0) goto fail;
    if (read_path(&r, cfg->outfile_storage, sizeof(cfg->outfile_storage)) != 0) goto fail;

    { int v[1]; if (read_ints(&r, v, 1) != 0) goto fail; cfg->hydro_switch = v[0]; }
    if (read_path(&r, cfg->location_file, sizeof(cfg->location_file)) != 0) goto fail;

    reader_close(&r);

    /* Parameter check + derived fields (RRI_Read.f90 tail). */
    for (int i = 0; i < n; i++) {
        if (lu->ksv[i] > 0.0 && lu->ka[i] > 0.0) {
            fprintf(stderr, "rri_config_read: both ksv and ka are non-zero for landuse %d\n", i + 1);
            return -1;
        }
        if (lu->gammam[i] > lu->gammaa[i]) {
            fprintf(stderr, "rri_config_read: gammam must be smaller than gammaa for landuse %d\n", i + 1);
            return -1;
        }
    }
    cfg->gw_switch = 0;
    for (int i = 0; i < n; i++) {
        if (lu->soildepth[i] > 0.0 && lu->ksv[i] > 0.0) lu->infilt_limit[i] = lu->soildepth[i] * lu->gammaa[i];
        if (lu->soildepth[i] > 0.0 && lu->ka[i] > 0.0) lu->da[i] = lu->soildepth[i] * lu->gammaa[i];
        if (lu->soildepth[i] > 0.0 && lu->ka[i] > 0.0 && lu->gammam[i] > 0.0) lu->dm[i] = lu->soildepth[i] * lu->gammam[i];
        if (lu->ksg[i] > 0.0) {
            cfg->gw_switch = 1;
        } else {
            lu->gammag[i] = 0.0; lu->kg0[i] = 0.0; lu->fpg[i] = 0.0; lu->rgl[i] = 0.0;
        }
    }
    return 0;

fail:
    fprintf(stderr, "rri_config_read: parse error in %s\n", path);
    reader_close(&r);
    return -1;
}

void rri_config_free(rri_config *cfg)
{
    rri_landuse *lu = &cfg->lu;
    free(lu->dif);
    free(lu->ns_slope); free(lu->soildepth); free(lu->gammaa);
    free(lu->ksv); free(lu->faif);
    free(lu->ka); free(lu->gammam); free(lu->beta);
    free(lu->ksg); free(lu->gammag); free(lu->kg0); free(lu->fpg); free(lu->rgl);
    free(lu->da); free(lu->dm); free(lu->infilt_limit);
    memset(lu, 0, sizeof(*lu));
}
