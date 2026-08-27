/**
 * @file rri_opencl.c
 * @brief OpenCL host-side backend implementation: device selection,
 * program build (kernels.h + cl/rri_kernels.cl concatenated at runtime),
 * and the dispatch functions declared in rri/opencl.h.
 *
 * ## Buffer lifecycle (PLAN.md milestone 10 -- persistent-buffer redesign)
 *
 * An earlier pass of this backend used a "create every buffer fresh,
 * upload via CL_MEM_COPY_HOST_PTR, run, read back, release" pattern for
 * EVERY argument on EVERY call, including the topology/parameter arrays
 * (`down`, `dis`, `len`, `zb`, `width`, `ns_slope`, `ka`, `da`, `dm`,
 * `beta`, `soildepth`, `gammaa`, `ksv`, `faif`, `ksg`, `gammag`, `kg0`,
 * `fpg`, `infilt_limit`, `dif`) that never change during a run -- these
 * were re-uploaded (and, for the slope/groundwater kernels, re-PACKED
 * from the array-of-pointers layout into the flat `[l*count+k]` layout,
 * see cl/rri_kernels.cl's file-level comment) on literally every RK45
 * stage call. Measured on solo30s (360h): ~115s wall on the actual
 * remote GPU vs ~35s for 32-core OpenMP on the same host -- confirmed
 * by inspection (not an external profiler; the code made the dominant
 * cost obvious) to be this repeated topology upload/repack, not kernel
 * compute time itself.
 *
 * This version separates each cellset's data into two buffer lifetimes:
 *
 * - **Topology/parameter buffers** (`rri_cl_riv_topology`/
 *   `rri_cl_slo_topology` below): uploaded ONCE, the first time a given
 *   `rri_riv_cellset`/`rri_slo_cellset` pointer is seen (cached by
 *   pointer identity in the backend -- `main.c` always passes the SAME
 *   `&m.rc`/`&m.sc` for the life of a run, so this is a correctness-safe
 *   simplification, not a heuristic: if the pointer ever changed
 *   mid-run, the cache would need invalidating, which it currently does
 *   NOT check for -- see `ensure_riv_topology`/`ensure_slo_topology`'s
 *   doc). Released only in `rri_cl_backend_free`. The slope/groundwater
 *   flat-packing (`pack_lmax8_*`) also happens exactly once here, not
 *   per call.
 * - **Dynamic state buffers** (the trial depth going in, the discharge
 *   coming out): persistent device-side allocations, reused across
 *   every call via `clEnqueueWriteBuffer`/`clEnqueueReadBuffer` in
 *   place -- still transferred every RK45 stage (this IS live,
 *   stage-varying state the host-side RK45 control flow and flux-scatter
 *   need to see, per rri.h's file-level comment on why that part of the
 *   solver stays host-side), but no `clCreateBuffer`/`clReleaseMemObject`
 *   churn and no COPY_HOST_PTR allocation overhead on the hot path
 *   anymore -- just a write, a kernel launch, and a read into
 *   already-existing buffer objects.
 *
 * This is option (a) from the milestone 10 discussion, not (b): the
 * flux-scatter step (shared-destination write across cells sharing a
 * downstream neighbor) stays host-side/serial, unchanged from the
 * previous pass. Moving it to the GPU would need either atomics or
 * restructuring as a per-destination-cell gather (summing inflow FROM
 * known upstream neighbors, which the current data model doesn't
 * precompute -- only downstream neighbors are indexed) -- judged not
 * worth the added correctness risk for this pass given the topology
 * upload was already the dominant cost; left as future work if a
 * profile after this change still points at the scatter step.
 *
 * Deliberately uses the OpenCL 1.x API surface only (`clCreateCommandQueue`,
 * not the OpenCL 2.0 `clCreateCommandQueueWithProperties`) so the exact
 * same source builds and runs unchanged against PoCL (OpenCL 3.0,
 * local validation) and the Mesa Clover platform on the AMD Polaris10
 * GPU this was validated against (OpenCL 1.1 -- see README.md's OpenCL
 * section for that validation run).
 */
#define CL_USE_DEPRECATED_OPENCL_1_1_APIS
#define CL_TARGET_OPENCL_VERSION 110
#include <CL/cl.h>

#include "rri/opencl.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifndef RRI_KERNELS_H_PATH
#error "RRI_KERNELS_H_PATH must be defined by the build (path to include/rri/kernels.h)"
#endif
#ifndef RRI_CL_SRC_PATH
#error "RRI_CL_SRC_PATH must be defined by the build (path to cl/rri_kernels.cl)"
#endif

/* Persistent river topology + dynamic state, cached against one
 * rri_riv_cellset pointer -- see file-level comment. */
typedef struct {
    const rri_riv_cellset *cellset; /* cache key: pointer identity */
    int count;
    cl_mem domain, zb, dis, down, width; /* topology, uploaded once */
    cl_mem hr, qr;                        /* dynamic state, persistent buffer objects */
} rri_cl_riv_topology;

/* Persistent slope/groundwater topology + dynamic state, cached against
 * one rri_slo_cellset pointer. Shared by qs_calc, qg_calc, AND infilt --
 * all three operate on the same cellset, so one upload serves all three
 * kernels' static arguments. */
typedef struct {
    const rri_slo_cellset *cellset;
    int count;
    /* topology / per-landuse parameters, uploaded once */
    cl_mem zb, ns_slope, ka, da, dm, beta, soildepth, gammaa, dif;
    cl_mem down, dis, len;       /* flat [l*count+k], packed once */
    cl_mem down_1d, dis_1d, len_1d;
    cl_mem gammag, kg0, fpg, ksg;               /* groundwater params */
    cl_mem ksv, faif, infilt_limit;              /* infiltration params */
    /* dynamic state, persistent buffer objects reused every call */
    cl_mem hs, qs;       /* qs_calc */
    cl_mem hg, qg;       /* qg_calc */
    cl_mem hs_rw, gff_rw, gf_out; /* infilt (in-place hs/gampt_ff, output rate) */
} rri_cl_slo_topology;

struct rri_cl_backend {
    cl_platform_id platform;
    cl_device_id device;
    cl_context context;
    cl_command_queue queue;
    cl_program program;
    cl_kernel k_qr, k_qs, k_qg, k_infilt;
    char device_name[256];

    rri_cl_riv_topology riv;
    rri_cl_slo_topology slo;
};

static char *read_file(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "rri_opencl: cannot open %s\n", path); return NULL; }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *buf = malloc((size_t)n + 1);
    size_t got = fread(buf, 1, (size_t)n, f);
    buf[got] = '\0';
    fclose(f);
    return buf;
}

/* RRI_KERNELS_H_PATH/RRI_CL_SRC_PATH are absolute, compile-time paths
 * into wherever this binary happened to be BUILT (CMakeLists.txt bakes
 * in ${CMAKE_SOURCE_DIR}) -- fine for the source-tree development
 * workflow this file was originally written for (RRI.opencl, run in
 * place), but wrong once this file is vendored into a GRASS addon and
 * built+installed somewhere else entirely (r.hydro.rri's Makefile
 * installs the binary under $GISBASE/etc/r.hydro.rri/bin/, and the
 * build directory it was compiled in may not even still exist by then).
 * RRI_ENGINE_SHARE_DIR, if set, overrides both compile-time paths to
 * <dir>/kernels.h and <dir>/rri_kernels.cl -- r.hydro.rri.py sets this
 * to wherever its own Makefile installed those two files alongside the
 * binary before invoking it. Falls back to the compile-time path when
 * unset, so RRI.opencl's own original build/run-in-place workflow is
 * untouched by this vendored copy's change. */
static char *resolved_kernel_path(const char *compile_time_path, const char *filename)
{
    const char *share_dir = getenv("RRI_ENGINE_SHARE_DIR");
    if (!share_dir) return read_file(compile_time_path);
    char path[4096];
    snprintf(path, sizeof(path), "%s/%s", share_dir, filename);
    return read_file(path);
}

static int device_has_fp64(cl_device_id dev)
{
    char ext[4096] = {0};
    clGetDeviceInfo(dev, CL_DEVICE_EXTENSIONS, sizeof(ext) - 1, ext, NULL);
    return strstr(ext, "cl_khr_fp64") != NULL;
}

/* Pick the first fp64-capable device, preferring a GPU when prefer_gpu is
 * set and one is actually available -- otherwise fall through to
 * whatever fp64-capable device exists (CPU, e.g. PoCL). Scans every
 * platform: multiple OpenCL platforms can be installed side by side
 * (e.g. this project's remote GPU host has both Clover and a rusticl
 * platform that reports 0 devices -- skipped naturally here since it
 * has nothing to enumerate). */
static int select_device(cl_platform_id *out_plat, cl_device_id *out_dev, int prefer_gpu)
{
    cl_uint nplat = 0;
    clGetPlatformIDs(0, NULL, &nplat);
    if (nplat == 0) { fprintf(stderr, "rri_opencl: no OpenCL platforms found\n"); return -1; }
    cl_platform_id *plats = malloc(sizeof(cl_platform_id) * nplat);
    clGetPlatformIDs(nplat, plats, NULL);

    cl_platform_id fallback_plat = 0; cl_device_id fallback_dev = 0;
    cl_platform_id gpu_plat = 0; cl_device_id gpu_dev = 0;

    for (cl_uint p = 0; p < nplat; p++) {
        cl_uint ndev = 0;
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, 0, NULL, &ndev);
        if (ndev == 0) continue;
        cl_device_id *devs = malloc(sizeof(cl_device_id) * ndev);
        clGetDeviceIDs(plats[p], CL_DEVICE_TYPE_ALL, ndev, devs, NULL);
        for (cl_uint d = 0; d < ndev; d++) {
            if (!device_has_fp64(devs[d])) continue;
            cl_device_type t;
            clGetDeviceInfo(devs[d], CL_DEVICE_TYPE, sizeof(t), &t, NULL);
            if (t == CL_DEVICE_TYPE_GPU && !gpu_dev) { gpu_plat = plats[p]; gpu_dev = devs[d]; }
            if (!fallback_dev) { fallback_plat = plats[p]; fallback_dev = devs[d]; }
        }
        free(devs);
    }
    free(plats);

    if (prefer_gpu && gpu_dev) { *out_plat = gpu_plat; *out_dev = gpu_dev; return 0; }
    if (fallback_dev) { *out_plat = fallback_plat; *out_dev = fallback_dev; return 0; }
    fprintf(stderr, "rri_opencl: no cl_khr_fp64-capable OpenCL device found\n");
    return -1;
}

rri_cl_backend *rri_cl_backend_init(int prefer_gpu)
{
    rri_cl_backend *b = calloc(1, sizeof(*b));
    if (select_device(&b->platform, &b->device, prefer_gpu) != 0) { free(b); return NULL; }

    char vendor[128] = {0}, name[128] = {0}, ver[128] = {0};
    clGetDeviceInfo(b->device, CL_DEVICE_VENDOR, sizeof(vendor) - 1, vendor, NULL);
    clGetDeviceInfo(b->device, CL_DEVICE_NAME, sizeof(name) - 1, name, NULL);
    clGetDeviceInfo(b->device, CL_DEVICE_VERSION, sizeof(ver) - 1, ver, NULL);
    snprintf(b->device_name, sizeof(b->device_name), "%s / %s / %s", vendor, name, ver);

    cl_int err;
    b->context = clCreateContext(NULL, 1, &b->device, NULL, NULL, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "rri_opencl: clCreateContext failed (%d)\n", err); free(b); return NULL; }
    b->queue = clCreateCommandQueue(b->context, b->device, 0, &err);
    if (err != CL_SUCCESS) { fprintf(stderr, "rri_opencl: clCreateCommandQueue failed (%d)\n", err); clReleaseContext(b->context); free(b); return NULL; }

    char *kernels_h = resolved_kernel_path(RRI_KERNELS_H_PATH, "kernels.h");
    char *cl_src = resolved_kernel_path(RRI_CL_SRC_PATH, "rri_kernels.cl");
    if (!kernels_h || !cl_src) { free(kernels_h); free(cl_src); rri_cl_backend_free(b); return NULL; }

    /* Concatenated at the OpenCL API level (three separate source
     * strings, joined by the driver's own compiler) rather than by
     * manual string concatenation in this file -- kernels.h's content
     * is handed to clCreateProgramWithSource completely unmodified,
     * per that file's hard requirement. */
    const char *pragma = "#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n";
    const char *srcs[3] = { pragma, kernels_h, cl_src };
    b->program = clCreateProgramWithSource(b->context, 3, srcs, NULL, &err);
    free(kernels_h); free(cl_src);
    if (err != CL_SUCCESS) { fprintf(stderr, "rri_opencl: clCreateProgramWithSource failed (%d)\n", err); rri_cl_backend_free(b); return NULL; }

    err = clBuildProgram(b->program, 1, &b->device, "", NULL, NULL);
    if (err != CL_SUCCESS) {
        size_t logsz = 0;
        clGetProgramBuildInfo(b->program, b->device, CL_PROGRAM_BUILD_LOG, 0, NULL, &logsz);
        char *log = malloc(logsz + 1);
        clGetProgramBuildInfo(b->program, b->device, CL_PROGRAM_BUILD_LOG, logsz, log, NULL);
        log[logsz] = '\0';
        fprintf(stderr, "rri_opencl: kernel build failed on %s:\n%s\n", b->device_name, log);
        free(log);
        rri_cl_backend_free(b);
        return NULL;
    }

    cl_int e1, e2, e3, e4;
    b->k_qr = clCreateKernel(b->program, "rri_cl_k_qr_calc", &e1);
    b->k_qs = clCreateKernel(b->program, "rri_cl_k_qs_calc", &e2);
    b->k_qg = clCreateKernel(b->program, "rri_cl_k_qg_calc", &e3);
    b->k_infilt = clCreateKernel(b->program, "rri_cl_k_infilt", &e4);
    if (e1 != CL_SUCCESS || e2 != CL_SUCCESS || e3 != CL_SUCCESS || e4 != CL_SUCCESS) {
        fprintf(stderr, "rri_opencl: clCreateKernel failed (%d,%d,%d,%d)\n", e1, e2, e3, e4);
        rri_cl_backend_free(b);
        return NULL;
    }
    return b;
}

static void release_riv_topology(rri_cl_riv_topology *t)
{
    if (t->domain) clReleaseMemObject(t->domain);
    if (t->zb) clReleaseMemObject(t->zb);
    if (t->dis) clReleaseMemObject(t->dis);
    if (t->down) clReleaseMemObject(t->down);
    if (t->width) clReleaseMemObject(t->width);
    if (t->hr) clReleaseMemObject(t->hr);
    if (t->qr) clReleaseMemObject(t->qr);
    memset(t, 0, sizeof(*t));
}

static void release_slo_topology(rri_cl_slo_topology *t)
{
    cl_mem *fields[] = { &t->zb, &t->ns_slope, &t->ka, &t->da, &t->dm, &t->beta, &t->soildepth,
                          &t->gammaa, &t->dif, &t->down, &t->dis, &t->len, &t->down_1d, &t->dis_1d,
                          &t->len_1d, &t->gammag, &t->kg0, &t->fpg, &t->ksg, &t->ksv, &t->faif,
                          &t->infilt_limit, &t->hs, &t->qs, &t->hg, &t->qg, &t->hs_rw, &t->gff_rw, &t->gf_out };
    for (size_t i = 0; i < sizeof(fields) / sizeof(fields[0]); i++)
        if (*fields[i]) clReleaseMemObject(*fields[i]);
    memset(t, 0, sizeof(*t));
}

void rri_cl_backend_free(rri_cl_backend *b)
{
    if (!b) return;
    release_riv_topology(&b->riv);
    release_slo_topology(&b->slo);
    if (b->k_qr) clReleaseKernel(b->k_qr);
    if (b->k_qs) clReleaseKernel(b->k_qs);
    if (b->k_qg) clReleaseKernel(b->k_qg);
    if (b->k_infilt) clReleaseKernel(b->k_infilt);
    if (b->program) clReleaseProgram(b->program);
    if (b->queue) clReleaseCommandQueue(b->queue);
    if (b->context) clReleaseContext(b->context);
    free(b);
}

const char *rri_cl_backend_device_name(const rri_cl_backend *b) { return b->device_name; }

/* ---- small helpers -------------------------------------------------- */

static cl_mem buf_ro(rri_cl_backend *b, const void *data, size_t bytes)
{
    cl_int err;
    cl_mem m = clCreateBuffer(b->context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, bytes, (void *)data, &err);
    return m;
}
/* Persistent, uninitialized device buffer for state that's written by
 * clEnqueueWriteBuffer before each kernel launch and/or read by
 * clEnqueueReadBuffer after -- allocated once by ensure_*_topology,
 * reused (NOT recreated) on every subsequent dispatch call. */
static cl_mem buf_state(rri_cl_backend *b, cl_mem_flags flags, size_t bytes)
{
    cl_int err;
    cl_mem m = clCreateBuffer(b->context, flags, bytes, NULL, &err);
    return m;
}

/* Pack rri_slo_cellset's RRI_LMAX8 array-of-pointers fields into one
 * flat [l * count + k] host buffer for upload -- see cl/rri_kernels.cl's
 * file-level comment for why this repacking exists. Called exactly once
 * per cellset now (from ensure_slo_topology), not once per kernel call. */
static double *pack_lmax8_double(double *const src[RRI_LMAX8], int count)
{
    double *flat = malloc(sizeof(double) * 4 * (size_t)count);
    for (int l = 0; l < 4; l++) memcpy(flat + (size_t)l * count, src[l], sizeof(double) * (size_t)count);
    return flat;
}
static int *pack_lmax8_int(int *const src[RRI_LMAX8], int count)
{
    int *flat = malloc(sizeof(int) * 4 * (size_t)count);
    for (int l = 0; l < 4; l++) memcpy(flat + (size_t)l * count, src[l], sizeof(int) * (size_t)count);
    return flat;
}
static void unpack_lmax8_double(const double *flat, double *dst[RRI_LMAX8], int count)
{
    for (int l = 0; l < 4; l++) memcpy(dst[l], flat + (size_t)l * count, sizeof(double) * (size_t)count);
}

/**
 * @brief Ensure @p b->riv holds topology buffers uploaded for @p rc,
 * uploading (once) if this is a new cellset pointer or the very first
 * call. NOTE: cache invalidation is by pointer identity only -- if the
 * SAME pointer were ever reused for a DIFFERENT cellset (not how
 * main.c uses this: it passes one stable `&m.rc` for a run's whole
 * lifetime), this cache would serve stale topology. Safe for this
 * port's actual call pattern; would need a generation counter or
 * explicit invalidation call if that assumption ever changes.
 */
static void ensure_riv_topology(rri_cl_backend *b, const rri_riv_cellset *rc)
{
    if (b->riv.cellset == rc) return;
    release_riv_topology(&b->riv);

    size_t n = (size_t)rc->count;
    b->riv.cellset = rc;
    b->riv.count = rc->count;
    b->riv.domain = buf_ro(b, rc->domain, n * sizeof(int));
    b->riv.zb = buf_ro(b, rc->zb, n * sizeof(double));
    b->riv.dis = buf_ro(b, rc->dis, n * sizeof(double));
    b->riv.down = buf_ro(b, rc->down, n * sizeof(int));
    b->riv.width = buf_ro(b, rc->width, n * sizeof(double));
    b->riv.hr = buf_state(b, CL_MEM_READ_ONLY, n * sizeof(double));
    b->riv.qr = buf_state(b, CL_MEM_WRITE_ONLY, n * sizeof(double));
}

/** @brief Slope/groundwater/infiltration counterpart of
 * ensure_riv_topology -- see that function's doc for the caching
 * contract. Uploads every static array AND the flattened per-direction
 * buffers (packed once here via pack_lmax8_*) for @p sc. */
static void ensure_slo_topology(rri_cl_backend *b, const rri_slo_cellset *sc)
{
    if (b->slo.cellset == sc) return;
    release_slo_topology(&b->slo);

    size_t n = (size_t)sc->count;
    int *down_flat = pack_lmax8_int(sc->down, sc->count);
    double *dis_flat = pack_lmax8_double(sc->dis, sc->count);
    double *len_flat = pack_lmax8_double(sc->len, sc->count);

    b->slo.cellset = sc;
    b->slo.count = sc->count;
    b->slo.zb = buf_ro(b, sc->zb, n * sizeof(double));
    b->slo.ns_slope = buf_ro(b, sc->ns_slope, n * sizeof(double));
    b->slo.ka = buf_ro(b, sc->ka, n * sizeof(double));
    b->slo.da = buf_ro(b, sc->da, n * sizeof(double));
    b->slo.dm = buf_ro(b, sc->dm, n * sizeof(double));
    b->slo.beta = buf_ro(b, sc->beta, n * sizeof(double));
    b->slo.soildepth = buf_ro(b, sc->soildepth, n * sizeof(double));
    b->slo.gammaa = buf_ro(b, sc->gammaa, n * sizeof(double));
    b->slo.dif = buf_ro(b, sc->dif, n * sizeof(int));
    b->slo.down = buf_ro(b, down_flat, 4 * n * sizeof(int));
    b->slo.dis = buf_ro(b, dis_flat, 4 * n * sizeof(double));
    b->slo.len = buf_ro(b, len_flat, 4 * n * sizeof(double));
    b->slo.down_1d = buf_ro(b, sc->down_1d, n * sizeof(int));
    b->slo.dis_1d = buf_ro(b, sc->dis_1d, n * sizeof(double));
    b->slo.len_1d = buf_ro(b, sc->len_1d, n * sizeof(double));
    b->slo.gammag = buf_ro(b, sc->gammag, n * sizeof(double));
    b->slo.kg0 = buf_ro(b, sc->kg0, n * sizeof(double));
    b->slo.fpg = buf_ro(b, sc->fpg, n * sizeof(double));
    b->slo.ksg = buf_ro(b, sc->ksg, n * sizeof(double));
    b->slo.ksv = buf_ro(b, sc->ksv, n * sizeof(double));
    b->slo.faif = buf_ro(b, sc->faif, n * sizeof(double));
    b->slo.infilt_limit = buf_ro(b, sc->infilt_limit, n * sizeof(double));

    b->slo.hs = buf_state(b, CL_MEM_READ_ONLY, n * sizeof(double));
    b->slo.qs = buf_state(b, CL_MEM_WRITE_ONLY, 4 * n * sizeof(double));
    b->slo.hg = buf_state(b, CL_MEM_READ_ONLY, n * sizeof(double));
    b->slo.qg = buf_state(b, CL_MEM_WRITE_ONLY, 4 * n * sizeof(double));
    b->slo.hs_rw = buf_state(b, CL_MEM_READ_WRITE, n * sizeof(double));
    b->slo.gff_rw = buf_state(b, CL_MEM_READ_WRITE, n * sizeof(double));
    b->slo.gf_out = buf_state(b, CL_MEM_WRITE_ONLY, n * sizeof(double));

    free(down_flat); free(dis_flat); free(len_flat);
}

void rri_cl_qr_calc(rri_cl_backend *b, const rri_riv_cellset *rc, const double *hr_idx,
                     double ns_river, double *qr_idx)
{
    ensure_riv_topology(b, rc);
    size_t n = (size_t)rc->count;

    clEnqueueWriteBuffer(b->queue, b->riv.hr, CL_FALSE, 0, n * sizeof(double), hr_idx, 0, NULL, NULL);

    int i = 0;
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &b->riv.domain);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &b->riv.zb);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &b->riv.dis);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &b->riv.down);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &b->riv.width);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &b->riv.hr);
    clSetKernelArg(b->k_qr, i++, sizeof(double), &ns_river);
    clSetKernelArg(b->k_qr, i++, sizeof(cl_mem), &b->riv.qr);

    size_t gws = n;
    clEnqueueNDRangeKernel(b->queue, b->k_qr, 1, NULL, &gws, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(b->queue, b->riv.qr, CL_TRUE, 0, n * sizeof(double), qr_idx, 0, NULL, NULL);
}

void rri_cl_qs_calc(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hs_idx,
                     double area, double *qs_idx[RRI_LMAX8])
{
    ensure_slo_topology(b, sc);
    size_t n = (size_t)sc->count;

    clEnqueueWriteBuffer(b->queue, b->slo.hs, CL_FALSE, 0, n * sizeof(double), hs_idx, 0, NULL, NULL);

    int lmax = sc->lmax, count = sc->count;
    int i = 0;
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.zb);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.ns_slope);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.ka);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.da);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.dm);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.beta);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.soildepth);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.gammaa);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.dif);
    clSetKernelArg(b->k_qs, i++, sizeof(int), &lmax);
    clSetKernelArg(b->k_qs, i++, sizeof(int), &count);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.down);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.dis);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.len);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.down_1d);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.dis_1d);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.len_1d);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.hs);
    clSetKernelArg(b->k_qs, i++, sizeof(double), &area);
    clSetKernelArg(b->k_qs, i++, sizeof(cl_mem), &b->slo.qs);

    size_t gws = n;
    clEnqueueNDRangeKernel(b->queue, b->k_qs, 1, NULL, &gws, NULL, 0, NULL, NULL);
    double *qs_flat = malloc(4 * n * sizeof(double));
    clEnqueueReadBuffer(b->queue, b->slo.qs, CL_TRUE, 0, 4 * n * sizeof(double), qs_flat, 0, NULL, NULL);
    unpack_lmax8_double(qs_flat, qs_idx, sc->count);
    free(qs_flat);
}

void rri_cl_qg_calc(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hg_idx,
                     double area, double *qg_idx[RRI_LMAX8])
{
    ensure_slo_topology(b, sc);
    size_t n = (size_t)sc->count;

    clEnqueueWriteBuffer(b->queue, b->slo.hg, CL_FALSE, 0, n * sizeof(double), hg_idx, 0, NULL, NULL);

    int lmax = sc->lmax, count = sc->count;
    int i = 0;
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.zb);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.gammag);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.kg0);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.fpg);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.ksg);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.dif);
    clSetKernelArg(b->k_qg, i++, sizeof(int), &lmax);
    clSetKernelArg(b->k_qg, i++, sizeof(int), &count);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.down);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.dis);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.len);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.down_1d);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.dis_1d);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.len_1d);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.hg);
    clSetKernelArg(b->k_qg, i++, sizeof(double), &area);
    clSetKernelArg(b->k_qg, i++, sizeof(cl_mem), &b->slo.qg);

    size_t gws = n;
    clEnqueueNDRangeKernel(b->queue, b->k_qg, 1, NULL, &gws, NULL, 0, NULL, NULL);
    double *qg_flat = malloc(4 * n * sizeof(double));
    clEnqueueReadBuffer(b->queue, b->slo.qg, CL_TRUE, 0, 4 * n * sizeof(double), qg_flat, 0, NULL, NULL);
    unpack_lmax8_double(qg_flat, qg_idx, sc->count);
    free(qg_flat);
}

void rri_cl_infilt(rri_cl_backend *b, const rri_slo_cellset *sc, double dt,
                    double *hs_idx, double *gampt_ff_idx, double *gampt_f_idx)
{
    ensure_slo_topology(b, sc);
    size_t n = (size_t)sc->count;

    clEnqueueWriteBuffer(b->queue, b->slo.hs_rw, CL_FALSE, 0, n * sizeof(double), hs_idx, 0, NULL, NULL);
    clEnqueueWriteBuffer(b->queue, b->slo.gff_rw, CL_FALSE, 0, n * sizeof(double), gampt_ff_idx, 0, NULL, NULL);

    int i = 0;
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &b->slo.ksv);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &b->slo.faif);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &b->slo.gammaa);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &b->slo.infilt_limit);
    clSetKernelArg(b->k_infilt, i++, sizeof(double), &dt);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &b->slo.hs_rw);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &b->slo.gff_rw);
    clSetKernelArg(b->k_infilt, i++, sizeof(cl_mem), &b->slo.gf_out);

    size_t gws = n;
    clEnqueueNDRangeKernel(b->queue, b->k_infilt, 1, NULL, &gws, NULL, 0, NULL, NULL);
    clEnqueueReadBuffer(b->queue, b->slo.hs_rw, CL_TRUE, 0, n * sizeof(double), hs_idx, 0, NULL, NULL);
    clEnqueueReadBuffer(b->queue, b->slo.gff_rw, CL_TRUE, 0, n * sizeof(double), gampt_ff_idx, 0, NULL, NULL);
    clEnqueueReadBuffer(b->queue, b->slo.gf_out, CL_TRUE, 0, n * sizeof(double), gampt_f_idx, 0, NULL, NULL);
}

/* ---- RK45-stage drivers: identical control flow to rri_funcr/rri_funcs/
 * rri_funcg (src/rri_riv.c, rri_slope.c, rri_gw.c) -- only the discharge
 * kernel call is swapped for the OpenCL dispatch above; the host-side
 * flux-scatter step is copied verbatim (kept as plain C, not itself an
 * OpenCL kernel -- see this file's file-level comment on why that part
 * of the solver stays host-side even after the buffer-lifecycle
 * redesign). Any change to the CPU versions' scatter logic must be
 * mirrored here. ---------------------------------------------------- */

void rri_cl_funcr(rri_cl_backend *b, const rri_riv_cellset *rc, const double *vr_idx,
                   double ns_river, double area, double *hr_idx, double *fr_idx,
                   double *qr_idx, double *qr_sum_scratch)
{
    for (int k = 0; k < rc->count; k++) hr_idx[k] = rri_vr2hr(vr_idx[k], area, rc->area_ratio[k]);

    rri_cl_qr_calc(b, rc, hr_idx, ns_river, qr_idx);

    for (int k = 0; k < rc->count; k++) qr_sum_scratch[k] = 0.0;
    for (int k = 0; k < rc->count; k++) {
        qr_sum_scratch[k] += qr_idx[k];
        int kk = rc->down[k];
        if (rc->domain[kk] != 0) qr_sum_scratch[kk] -= qr_idx[k];
    }
    for (int k = 0; k < rc->count; k++) fr_idx[k] = -qr_sum_scratch[k];
}

void rri_cl_funcs(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hs_idx,
                   const double *qp_t_idx, double area, double *fs_idx, double *qs_idx[RRI_LMAX8])
{
    rri_cl_qs_calc(b, sc, hs_idx, area, qs_idx);

    for (int k = 0; k < sc->count; k++) {
        double outflow = 0.0;
        for (int l = 0; l < RRI_LMAX8; l++) outflow += qs_idx[l][k];
        fs_idx[k] = qp_t_idx[k] - outflow;
    }
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

void rri_cl_funcg(rri_cl_backend *b, const rri_slo_cellset *sc, const double *hg_idx,
                   double area, double *fg_idx, double *qg_idx[RRI_LMAX8])
{
    rri_cl_qg_calc(b, sc, hg_idx, area, qg_idx);

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
