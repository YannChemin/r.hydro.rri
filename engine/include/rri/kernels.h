/**
 * @file kernels.h
 * @brief Shared per-cell hydraulic formulas: river/slope/groundwater
 * discharge relationships and the slope water-level correction.
 *
 * These are the innermost physics of RRI: given a water depth and a head
 * gradient to a neighboring cell, how much water moves per unit time.
 * Everything else in the solver (RK45 time-stepping, sparse index
 * bookkeeping, mass-balance accounting) exists to call these functions in
 * the right place with the right neighbor.
 *
 * ## Why this file exists separately from rri_riv.c / rri_slope.c / rri_gw.c
 *
 * PLAN.md's design (section 5, decision 5) is to write each kernel's
 * math body exactly once, so the OpenMP path (src/rri_riv.c etc.,
 * `#pragma omp parallel for` over cell index k) and the OpenCL path
 * (cl/rri_kernels.cl, `__kernel` + global-id lookup) cannot numerically diverge --
 * they call the *same compiled function*, not two independently-written
 * translations of the same Fortran formula. That divergence risk is not
 * hypothetical: the C port's full-length validation against the Fortran
 * reference on solo30s (see ../../README.md, section "Root-caused bug:
 * missing river-channel bed incision") found a real, silent, slowly
 * compounding mass-balance bug from a single array being reused where the
 * Fortran reference kept two distinct ones. A second independently-typed
 * copy of these formulas for OpenCL would be exactly the kind of place
 * that class of bug hides again, invisibly, until a long enough run
 * exposes it.
 *
 * ## Hard constraint: stay portable OpenCL C
 *
 * **Every function in this file must remain a pure function of scalar
 * arguments: no pointers, no structs, no globals, no host-only library
 * calls beyond `sqrt`/`pow`/`exp`/`fabs`.** This is not a style
 * preference -- OpenCL C (the language `.cl` kernel source is written in)
 * is a restricted C99 dialect that does not support pointers-to-struct
 * across the host/device boundary, and the whole point of writing the
 * math once is that this header must `#include` unchanged into both a
 * plain C11 translation unit (as it does today, from rri_riv.c /
 * rri_slope.c / rri_gw.c) and, in the next milestone, into an OpenCL C
 * kernel source file with no edits. If you find yourself wanting to pass
 * an `rri_riv_cellset*` or similar into one of these functions to "clean
 * up" the call site, that's a sign the neighbor-lookup / cellset-walking
 * code belongs in the caller (rri_riv.c / rri_slope.c / rri_gw.c, or the
 * eventual `.cl` kernel wrapper), not here. Do not weaken this
 * constraint without updating this comment and re-deriving whether the
 * OpenCL extraction still works.
 *
 * `#ifndef __OPENCL_VERSION__` below is how that unchanged-under-both-
 * compilers property is implemented in practice: `__OPENCL_VERSION__` is
 * predefined by an OpenCL C device compiler and absent under a normal C
 * compiler, so `<math.h>` is only pulled in on the host side (OpenCL C
 * has `sqrt`/`pow`/`exp`/`fabs` as language builtins, no header needed --
 * including `<math.h>` there would fail to compile). The same branch
 * also picks `RRI_INLINE`'s expansion: plain C wants `static inline`
 * (internal linkage, so this header can be included from multiple .c
 * translation units without a duplicate-symbol link error); OpenCL C
 * 1.1 -- confirmed against the Mesa Clover platform on an AMD Polaris10
 * GPU, this port's actual remote-GPU validation target, see README.md --
 * rejects the `static` storage-class specifier on a function at this
 * scope entirely (a hard compile error, not a portability nicety), so
 * the OpenCL branch drops it and uses plain `inline`; this is safe
 * there because each OpenCL kernel source is its own translation unit,
 * so there's no multi-TU duplicate-symbol concern to guard against.
 */
#ifndef RRI_KERNELS_H
#define RRI_KERNELS_H

#ifndef __OPENCL_VERSION__
#include <math.h>
#define RRI_INLINE static inline
#else
#define RRI_INLINE inline
#endif

/**
 * @brief River channel discharge from water depth and head gradient
 * (Manning's equation, rectangular channel).
 *
 * Fortran reference: RRI_Riv.f90, `hq_riv`, the `sec_map_idx(k)<=0`
 * branch only -- custom cross-section geometry (`sec_map>0`, a lookup
 * table of surveyed channel shapes) is a separate code path in the
 * Fortran that this port does not implement; every river cell is treated
 * as a rectangular channel of the given width.
 *
 * @param h   Water depth in the channel driving the flow [m]. Not
 *            necessarily the raw cell water depth -- the caller (see
 *            rri_riv.c: rri_qr_calc) picks the upstream or downstream
 *            cell's depth depending on flow direction, and clamps it to
 *            the bank-relative depth when the bed elevations differ.
 * @param dh  Head gradient driving the flow, i.e. (water surface
 *            elevation difference) / (distance to neighbor) [dimensionless,
 *            m/m]. Always used non-negative here (`fabs`) -- sign/direction
 *            of flow is decided by the caller before calling this
 *            function twice (once for each candidate direction) and
 *            negating the result for the reverse case; see rri_qr_calc's
 *            `dh >= 0.0` branch.
 * @param w   Channel width [m].
 * @param ns_river  Manning's roughness coefficient for the river channel
 *            [s/m^(1/3)], a single global value in this port (`ns_river`
 *            from RRI_Input.txt -- Fortran allows this to vary by
 *            cross-section table entry via `sec_ns_river`, unused here).
 * @return Discharge [m^3/s]. Caller applies the sign for flow direction.
 */
RRI_INLINE double rri_k_hq_riv(double h, double dh, double w, double ns_river)
{
    double a = sqrt(fabs(dh)) / ns_river;
    double r = (w * h) / (w + 2.0 * h); /* hydraulic radius, rectangular channel: wetted area / wetted perimeter */
    return a * pow(r, 2.0 / 3.0) * w * h;
}

/**
 * @brief Slope (overland + subsurface) discharge from water depth and
 * head gradient, combining Darcy subsurface flow, a Manning-like matrix
 * flow term, and free-surface overland flow as depth increases past the
 * soil column.
 *
 * Fortran reference: RRI_Slope.f90, `hq`. This is the model's core
 * hillslope hydrology: below `dm_p` (a landuse-specific "matrix flow"
 * threshold depth) water moves through the soil matrix with a power-law
 * profile; between `dm_p` and `da_p` (the depth at which the soil column
 * saturates) flow is linear Darcy flow; above `da_p` free-surface
 * Manning overland flow is added on top. See `rri_k_h2lev` below for how
 * a raw water depth `h` is converted to the *elevation* used in the head
 * gradient this function's `dh` is derived from -- the two functions are
 * a matched pair.
 *
 * @param ns_p  Manning's roughness for overland flow above da_p [s/m^(1/3)].
 * @param ka_p  Saturated hydraulic conductivity for lateral subsurface
 *              (Darcy) flow [m/s]; landuse parameter `ka`.
 * @param da_p  Soil column saturation depth [m] (`soildepth * gammaa`
 *              for this landuse); the transition to free-surface flow.
 * @param dm_p  Matrix-flow threshold depth [m] (`soildepth * gammam`);
 *              below this, subsurface flow follows a `beta`-power profile
 *              instead of linear Darcy flow.
 * @param b_p   Power-law exponent (`beta` landuse parameter) for the
 *              matrix-flow regime below dm_p, dimensionless.
 * @param h     Water depth driving the flow (see rri_k_hq_riv's `h` --
 *              same upstream/downstream selection logic in the caller) [m].
 * @param dh    Head gradient (elevation difference / distance) [m/m],
 *              always non-negative here; see rri_k_hq_riv.
 * @param length  Contour length of the cell edge the flow crosses [m]
 *              (precomputed per direction in slo_idx_setting -- NOT the
 *              same as the distance to the neighbor cell's centroid).
 * @param area  Grid cell area [m^2], used to convert the volumetric rate
 *              this formula naturally produces into a per-unit-area rate
 *              consistent with the water-depth state variable's units.
 * @return Discharge per unit cell area [m/s] (i.e. an equivalent depth
 *         rate, matching the units of the `hs`/RK45 state it feeds into).
 */
RRI_INLINE double rri_k_hq_slope(double ns_p, double ka_p, double da_p,
                                  double dm_p, double b_p, double h, double dh,
                                  double length, double area)
{
    double km = (b_p > 0.0) ? ka_p / b_p : 0.0;
    double vm = km * dh;
    double va = (da_p > 0.0) ? ka_p * dh : 0.0;
    double dh_pos = (dh < 0.0) ? 0.0 : dh;
    double al = sqrt(dh_pos) / ns_p;
    double m = 5.0 / 3.0;
    double q;

    if (h < dm_p) {
        q = vm * dm_p * pow(h / dm_p, b_p);
    } else if (h < da_p) {
        q = vm * dm_p + va * (h - dm_p);
    } else {
        q = vm * dm_p + va * (h - dm_p) + al * pow(h - da_p, m);
    }
    return q * length / area; /* volumetric rate -> per-unit-area rate, matching the hs state variable's units */
}

/**
 * @brief Convert a raw slope water depth to the effective water-surface
 * *level* used in head-gradient calculations, accounting for the soil
 * column acting as a porous reservoir.
 *
 * Fortran reference: RRI_Slope.f90, `h2lev`. Physically: water sitting
 * in the soil pores (depth below `da_temp = soildepth * gammaa`, the
 * depth at which the pore space saturates) occupies less *effective*
 * surface elevation than the same depth of free water would, because the
 * porosity `gammaa` < 1 means the soil "compresses" that depth of stored
 * water into a smaller apparent water-table rise. Above `da_temp` the
 * soil is saturated and any further depth is free surface water, so it
 * counts 1:1 with `h`. This function's output, added to the cell's bed
 * elevation `zb`, is what actually drives the head gradient `dh` passed
 * into rri_k_hq_slope -- i.e. the model routes based on apparent water
 * level, not raw depth, which is why a shallow soil-saturated cell can
 * still drive flow even at small `h`.
 *
 * @param h          Raw water depth [m].
 * @param soildepth  Soil column depth for this cell's landuse [m]; 0
 *                   means "no soil layer" (e.g. bare rock/impervious),
 *                   in which case level == depth exactly (no porosity
 *                   correction applies).
 * @param gammaa     Effective porosity for this cell's landuse [-],
 *                   0 < gammaa <= 1.
 * @return Effective water level above the bed [m].
 */
RRI_INLINE double rri_k_h2lev(double h, double soildepth, double gammaa)
{
    double da_temp = soildepth * gammaa;
    if (soildepth == 0.0) {
        return h;
    } else if (h >= da_temp) {
        return soildepth + (h - da_temp);
    } else {
        double rho = (soildepth > 0.0) ? da_temp / soildepth : 1.0;
        return h / rho;
    }
}

/**
 * @brief Lateral groundwater (bedrock aquifer) discharge from a
 * groundwater-table depth and head gradient (Darcy-type flow with
 * exponentially depth-decaying transmissivity).
 *
 * Fortran reference: RRI_GW.f90, `hg_calc`. `hg` (the state variable
 * this drives) is a *deficit* below the ground surface, not a depth
 * above it like `hs`/`hr` -- see rri_setup.c: rri_storage_calc's
 * `sg -= hg * gammag * area` for the sign convention this implies (a
 * larger `hg` deficit means *less* stored groundwater, hence the minus).
 *
 * @param gammag_p  Specific yield / drainable porosity for the aquifer
 *                  at this cell's landuse [-]. Present in the signature
 *                  for symmetry with the Fortran call site but unused in
 *                  this formula (`(void)gammag_p` below) -- Fortran's
 *                  `hg_calc` doesn't use it either; the caller (rri_gw.c:
 *                  rri_qg_calc/rri_funcg) uses gammag separately to
 *                  convert the discharge into a depth-deficit rate.
 * @param kg0_p     Reference bedrock hydraulic conductivity at the
 *                  surface [m/s] (landuse parameter `kg0`).
 * @param fpg_p     Exponential decay rate of conductivity with depth
 *                  [1/m] (landuse parameter `fpg`) -- deeper groundwater
 *                  moves through less transmissive material.
 * @param hg_p      Groundwater deficit at the source cell [m] (see note
 *                  above on sign convention).
 * @param dh        Head gradient [m/m], always non-negative here; see
 *                  rri_k_hq_riv.
 * @param length    Cell-edge contour length [m]; see rri_k_hq_slope.
 * @param area      Grid cell area [m^2]; see rri_k_hq_slope.
 * @return Discharge per unit cell area [m/s].
 */
RRI_INLINE double rri_k_hg_calc(double gammag_p, double kg0_p, double fpg_p,
                                 double hg_p, double dh, double length, double area)
{
    double qg = dh * kg0_p / fpg_p * exp(-fpg_p * hg_p);
    (void)gammag_p;
    return qg * length / area;
}

#endif /* RRI_KERNELS_H */
