/**
 * @file rri_kernels.cl
 * @brief OpenCL __kernel wrappers around include/rri/kernels.h's shared
 * per-cell math. This file is NEVER compiled standalone -- src/rri_opencl.c
 * concatenates it at runtime as `"#pragma OPENCL EXTENSION cl_khr_fp64 : enable\n"
 * + <verbatim contents of kernels.h> + <verbatim contents of this file>`
 * into one OpenCL C source string handed to clCreateProgramWithSource.
 * kernels.h's math bodies are pulled in completely unmodified -- see that
 * file's top-of-file comment for why that's a hard requirement (this is
 * the seam PLAN.md's design exists to make possible; do not fork the math
 * into a second, OpenCL-specific copy).
 *
 * @note The `cl_khr_fp64` pragma is NOT optional boilerplate: OpenCL C on
 * an OpenCL 1.1 device (e.g. the AMD Polaris10 GPU via Mesa's Clover
 * platform this was validated against, see README.md) refuses to compile
 * any kernel using `double` without it explicitly enabled at the top of
 * the source. OpenCL 1.2+ implementations (including PoCL, used for local
 * validation) accept the pragma too, so prepending it unconditionally is
 * safe everywhere rather than needing a device-version branch.
 *
 * @par Data layout for the per-direction (slope/groundwater) kernels
 * `rri_slo_cellset`'s `down[RRI_LMAX8]`/`dis[RRI_LMAX8]`/`len[RRI_LMAX8]`
 * fields are, on the host, an ARRAY OF SEPARATE POINTERS (one buffer per
 * direction slot) -- convenient on the CPU, but each would need its own
 * `clSetKernelArg` call and there's no clean way to pass "4 buffers" as
 * one OpenCL kernel argument. src/rri_opencl.c instead packs them into
 * ONE flat buffer per field, `[l * count + k]` layout (see that file's
 * `pack_lmax8`/`unpack_lmax8` helpers), before upload; the kernels below
 * index accordingly. This repacking is a host-side transformation only
 * -- it does not touch the validated `rri_slo_cellset` struct or CPU
 * code path at all.
 *
 * @par Parallelization: one work-item per active cell (`get_global_id(0)`
 * over the cellset's `count`), exactly the same embarrassingly-parallel
 * structure as the OpenMP `#pragma omp parallel for` loops in
 * src/rri_riv.c/rri_slope.c/rri_gw.c/rri_infilt.c -- see rri.h's
 * file-level comment for why the sparse index representation makes this
 * safe (no cell-to-cell dependency within one kernel invocation).
 */

/**
 * @brief River lateral discharge (OpenCL counterpart of
 * src/rri_riv.c: rri_qr_calc). One work-item per river cell.
 */
__kernel void rri_cl_k_qr_calc(
    __global const int *domain,      /* rri_riv_cellset::domain */
    __global const double *zb,       /* rri_riv_cellset::zb (river bed, NOT grid::zb) */
    __global const double *dis,      /* rri_riv_cellset::dis */
    __global const int *down,        /* rri_riv_cellset::down */
    __global const double *width,    /* rri_riv_cellset::width */
    __global const double *hr_idx,
    double ns_river,
    __global double *qr_idx)         /* output */
{
    int k = get_global_id(0);
    if (domain[k] == 2) { qr_idx[k] = 0.0; return; }

    double zb_p = zb[k], hr_p = hr_idx[k];
    double distance = dis[k];
    int kk = down[k];
    double zb_n = zb[kk], hr_n = hr_idx[kk];

    double dh = ((zb_p + hr_p) - (zb_n + hr_n)) / distance;
    if (domain[kk] == 2) dh = (zb_p + hr_p - zb_n) / distance;

    double q;
    if (dh >= 0.0) {
        double hw = hr_p;
        if (zb_p < zb_n) { double t = zb_p + hr_p - zb_n; hw = t > 0.0 ? t : 0.0; }
        q = rri_k_hq_riv(hw, dh, width[k], ns_river);
    } else {
        double hw = hr_n;
        if (zb_n < zb_p) { double t = zb_n + hr_n - zb_p; hw = t > 0.0 ? t : 0.0; }
        q = -rri_k_hq_riv(hw, fabs(dh), width[kk], ns_river);
    }
    qr_idx[k] = q;
}

/**
 * @brief Hillslope lateral discharge (OpenCL counterpart of
 * src/rri_slope.c: rri_qs_calc). One work-item per slope cell; the
 * per-direction fields are the flattened `[l * count + k]` layout
 * described in the file-level comment above.
 */
__kernel void rri_cl_k_qs_calc(
    __global const double *zb, __global const double *ns_slope,
    __global const double *ka, __global const double *da,
    __global const double *dm, __global const double *beta,
    __global const double *soildepth, __global const double *gammaa,
    __global const int *dif, int lmax, int count,
    __global const int *down_flat, __global const double *dis_flat, __global const double *len_flat,
    __global const int *down_1d, __global const double *dis_1d, __global const double *len_1d,
    __global const double *hs_idx, double area,
    __global double *qs_flat)  /* output, [l * count + k] layout, RRI_LMAX8*count entries */
{
    int k = get_global_id(0);
    double zb_p = zb[k], hs_p = hs_idx[k];
    double ns_p = ns_slope[k], ka_p = ka[k], da_p = da[k], dm_p = dm[k], b_p = beta[k];
    int dif_p = dif[k];

    for (int l = 0; l < 4; l++) qs_flat[l * count + k] = 0.0;

    for (int l = 0; l < lmax; l++) {
        if (dif_p == 0 && l == 1) break;
        int kk = (dif_p == 0) ? down_1d[k] : down_flat[l * count + k];
        if (kk == -1) continue;
        double distance = (dif_p == 0) ? dis_1d[k] : dis_flat[l * count + k];
        double length = (dif_p == 0) ? len_1d[k] : len_flat[l * count + k];

        double zb_n = zb[kk], hs_n = hs_idx[kk];
        double lev_p = rri_k_h2lev(hs_p, soildepth[k], gammaa[k]);
        double lev_n = rri_k_h2lev(hs_n, soildepth[kk], gammaa[kk]);

        double dh = ((zb_p + lev_p) - (zb_n + lev_n)) / distance;
        if (dif_p == 0) { double t = (zb_p - zb_n) / distance; dh = t > 0.001 ? t : 0.001; }

        double q;
        if (dh >= 0.0) {
            double hw = hs_p;
            if (zb_p < zb_n) { double t = zb_p + hs_p - zb_n; hw = t > 0.0 ? t : 0.0; }
            q = rri_k_hq_slope(ns_p, ka_p, da_p, dm_p, b_p, hw, dh, length, area);
        } else {
            double ns_n = ns_slope[kk], ka_n = ka[kk], da_n = da[kk], dm_n = dm[kk], b_n = beta[kk];
            double hw = hs_n;
            dh = fabs(dh);
            if (zb_n < zb_p) { double t = zb_n + hs_n - zb_p; hw = t > 0.0 ? t : 0.0; }
            q = -rri_k_hq_slope(ns_n, ka_n, da_n, dm_n, b_n, hw, dh, length, area);
        }
        qs_flat[l * count + k] = q;
    }
}

/**
 * @brief Lateral groundwater discharge (OpenCL counterpart of
 * src/rri_gw.c: rri_qg_calc). Same flattened layout as rri_cl_k_qs_calc.
 */
__kernel void rri_cl_k_qg_calc(
    __global const double *zb, __global const double *gammag,
    __global const double *kg0, __global const double *fpg, __global const double *ksg,
    __global const int *dif, int lmax, int count,
    __global const int *down_flat, __global const double *dis_flat, __global const double *len_flat,
    __global const int *down_1d, __global const double *dis_1d, __global const double *len_1d,
    __global const double *hg_idx, double area,
    __global double *qg_flat)
{
    int k = get_global_id(0);
    for (int l = 0; l < 4; l++) qg_flat[l * count + k] = 0.0;
    if (ksg[k] <= 0.0) return;

    double zb_p = zb[k], hg_p = hg_idx[k];
    double gammag_p = gammag[k], kg0_p = kg0[k], fpg_p = fpg[k];
    int dif_p = dif[k];

    for (int l = 0; l < lmax; l++) {
        if (dif_p == 0 && l == 1) break;
        int kk = (dif_p == 0) ? down_1d[k] : down_flat[l * count + k];
        if (kk == -1) continue;
        if (ksg[kk] <= 0.0) continue;
        double distance = (dif_p == 0) ? dis_1d[k] : dis_flat[l * count + k];
        double length = (dif_p == 0) ? len_1d[k] : len_flat[l * count + k];

        double zb_n = zb[kk], hg_n = hg_idx[kk];
        double dh = ((zb_p - hg_p) - (zb_n - hg_n)) / distance;
        if (dif_p == 0) { double t = (zb_p - zb_n) / distance; dh = t > 0.001 ? t : 0.001; }

        if (dh >= 0.0) {
            qg_flat[l * count + k] = rri_k_hg_calc(gammag_p, kg0_p, fpg_p, hg_p, dh, length, area);
        } else {
            double gammag_n = gammag[kk], kg0_n = kg0[kk], fpg_n = fpg[kk];
            qg_flat[l * count + k] = -rri_k_hg_calc(gammag_n, kg0_n, fpg_n, hg_n, fabs(dh), length, area);
        }
    }
}

/**
 * @brief Green-Ampt infiltration (OpenCL counterpart of
 * src/rri_infilt.c: rri_infilt). No neighbor lookups at all -- every
 * field here is a plain per-cell array, the simplest kernel in this file.
 */
__kernel void rri_cl_k_infilt(
    __global const double *ksv, __global const double *faif, __global const double *gammaa,
    __global const double *infilt_limit, double dt,
    __global double *hs_idx, __global double *gampt_ff_idx, __global double *gampt_f_idx)
{
    int k = get_global_id(0);
    double gampt_ff_temp = gampt_ff_idx[k];
    if (gampt_ff_temp <= 0.01) gampt_ff_temp = 0.01;

    double f = ksv[k] * (1.0 + faif[k] * gammaa[k] / gampt_ff_temp);
    if (f >= hs_idx[k] / dt) f = hs_idx[k] / dt;
    if (infilt_limit[k] >= 0.0 && gampt_ff_idx[k] >= infilt_limit[k]) f = 0.0;

    gampt_f_idx[k] = f;
    gampt_ff_idx[k] += f * dt;
    hs_idx[k] -= f * dt;
    if (hs_idx[k] <= 0.0) hs_idx[k] = 0.0;
}
