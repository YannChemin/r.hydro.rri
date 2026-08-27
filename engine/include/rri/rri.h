/**
 * @file rri.h
 * @brief Public API for the RRI.opencl core: data structures and function
 * declarations for a C11 port of RRI (Rainfall-Runoff-Inundation), a
 * distributed hydrology model coupling diffusive-wave river routing,
 * diffusive-wave/kinematic hillslope routing, a shallow bedrock
 * groundwater aquifer, and Green-Ampt infiltration, all advanced together
 * by an adaptive-step Runge-Kutta-Fehlberg (Cash-Karp) time integrator.
 *
 * Corresponds to the Fortran module-level declarations spread across
 * RRI_Mod.f90, RRI_Mod2.f90, RRI_Mod_Dam.f90, RRI_Mod_Tecout.f90, and the
 * argument lists of RRI_Sub.f90's setup routines -- Fortran keeps these
 * as global module variables; this port groups them into the structs
 * below and threads them explicitly as function arguments (needed for
 * OpenCL portability: kernels can't read Fortran-style module globals,
 * and passing explicit buffers is what lets the same pointers later hand
 * off to clCreateBuffer with no repacking step).
 *
 * ## Scope
 *
 * Implemented: diffusive-wave river + hillslope routing, river<->slope
 * exchange, shallow-aquifer groundwater lateral flow with
 * recharge/exfiltration, Green-Ampt infiltration -- i.e. everything
 * needed to route rainfall through a watershed to its outlet.
 *
 * NOT implemented (see README.md "What's NOT implemented" for the full
 * list and why each is safe to omit for the validated reference config):
 * dam operation, diversion, boundary conditions (prescribed water level
 * or discharge), custom river cross-sections (`sec_map`), TSAS particle
 * tracking. `main.c` errors out at startup if a config file requests any
 * of these rather than silently ignoring them.
 *
 * ## Data layout: why a sparse index representation
 *
 * Grids (`rri_grid`) are the full (ny, nx) raster, row-major, row 0 =
 * northernmost row (matching ESRI ASCII grid convention -- this is also
 * exactly the order `rri_read_gis_real`/`rri_read_gis_int` produce and
 * consume). But the actual river and hillslope *state* the RK45
 * integrator advances is NOT stored per raster cell -- it's compressed
 * into 1D arrays covering only the active cells (`rri_riv_cellset`,
 * `rri_slo_cellset`, built once by `rri_riv_idx_setting` /
 * `rri_slo_idx_setting`), because most rasters have large inactive
 * (no-data) regions and, for the river cellset, only a small fraction of
 * total cells are river cells at all.
 *
 * Each active cell `k` in a cellset carries its own precomputed neighbor
 * lookup -- `down[k]` (or `down[l][k]` for the up-to-4 hillslope
 * directions), `dis[k]`/`dis[l][k]` (distance to that neighbor), and
 * `len[k]`/`len[l][k]` (the shared edge's contour length) -- built once
 * at startup instead of being re-derived from (i,j) arithmetic on every
 * timestep. This is a CSR-like sparse graph: each cell's physics update
 * (see kernels.h and rri_riv.c/rri_slope.c/rri_gw.c) reads only its own
 * state and its precomputed neighbors' state from the *previous*
 * timestep, and writes only its own output -- there is no cell-to-cell
 * dependency *within* a single kernel invocation. That is precisely what
 * makes the per-cell loops (`rri_qr_calc`, `rri_qs_calc`, `rri_qg_calc`)
 * embarrassingly parallel: each is a `parallel_for` over the cellset's
 * `count`, safe under OpenMP today and, via kernels.h's shared math
 * bodies, intended to become an OpenCL `clEnqueueNDRangeKernel` over the
 * same index next (see PLAN.md milestone 8). The parts of the solver
 * that are NOT structured this way -- the RK45 accept/reject control
 * flow, the flux-scatter step that sums each cell's *outflow* into its
 * downstream neighbor's *inflow* (a shared-destination write, so kept
 * serial; see rri_riv.c/rri_slope.c), dam/diversion bookkeeping -- are
 * exactly the parts PLAN.md section 2 identifies as staying host-side.
 */
#ifndef RRI_H
#define RRI_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ---- grid-level (ny*nx) state ------------------------------------- */

/**
 * @brief Full-raster grid state: topography, land cover, and derived
 * river geometry, at (ny, nx) resolution.
 *
 * Everything here is indexed `[i * nx + j]`, row-major, row 0 =
 * northernmost (see file-level comment). This struct is the source data
 * that `rri_riv_idx_setting`/`rri_slo_idx_setting` compress into the two
 * cellset structs below; after that compression, the main time loop
 * (src/main.c) reads/writes almost exclusively through the cellsets and
 * only touches this struct's fields when converting between the two
 * representations (`rri_riv_ij2idx`/`idx2ij` etc.) or for one-time setup.
 */
typedef struct {
    int ny, nx;
    double xllcorner, yllcorner, cellsize;
    double dx, dy;      /**< Cell size in metres. For utm==0 (lat/lon input), computed via
                              rri_hubeny_sub as the geodesic average of the domain's north/south
                              and east/west edge lengths divided by nx/ny -- NOT simply
                              cellsize-in-degrees times a constant, since degree-to-metre scaling
                              varies with latitude. */
    double area;         /**< dx*dy [m^2]; the unit-conversion factor between a per-area rate
                               (e.g. rainfall intensity, most kernel outputs) and a volumetric
                               flow rate. */
    double length;        /**< sqrt(dx*dy) [m]; used as the river reach length (len_riv) for
                                every river cell in the parametric (rivfile_switch==0) geometry
                                path -- i.e. this port assumes one river reach spans exactly one
                                grid cell's characteristic length, not the actual meander length. */

    /**
     * @name Bed elevation: two DISTINCT arrays, do not conflate
     *
     * `zb` and `zb_riv` are separate physical quantities and must stay
     * separate. Conflating them (using one array for both slope and
     * river bed elevation, e.g. by feeding raw DEM values to both, or
     * reusing `zb` when building the river cellset) was a real bug found
     * during full-length validation against the Fortran reference on
     * solo30s: it does not crash and does not show up in short-window
     * comparisons, because with small water depths the missing few-metre
     * channel-incision offset barely changes the computed head gradient.
     * It instead produces a *slowly growing, one-directional* water-mass
     * drift -- the river channel's elevation advantage over the
     * surrounding slope is understated, so hillslope water systematically
     * under-drains into the channel and accumulates on the slope instead,
     * an effect invisible in the first day of simulation and reaching
     * >50% relative error in cumulative storage by the time of a
     * multi-day flood peak. See README.md's "Root-caused bug" section for
     * the full bisection writeup. The lesson generalizes: a mass-balance
     * bug that only manifests after many accumulated timesteps will not
     * be caught by a short smoke test -- validate at least one full-length
     * run before trusting a change to bed-elevation or channel-geometry code.
     * @{
     */
    double *zb;           /**< SLOPE bed elevation = dem - soildepth[land] [m] (RRI.f90 ~line 223).
                                Applies to EVERY cell (river cells included) as the elevation used
                                by hillslope routing (rri_qs_calc) and groundwater (rri_qg_calc). */
    double *zb_riv;        /**< RIVER CHANNEL bed elevation = dem - depth, for river cells only;
                                 equal to the raw dem elsewhere (unused there) (RRI.f90 ~line 224).
                                 This is what rri_riv_idx_setting copies into rri_riv_cellset::zb --
                                 river routing (rri_qr_calc) must never read grid::zb for bed
                                 elevation, only grid::zb_riv. */
    /** @} */

    double *acc;          /**< Upstream contributing area, in grid cells (flow accumulation),
                                read directly from the accfile grid. Drives both the river-cell
                                mask (`acc > riv_thresh`) and the parametric channel-geometry
                                power laws (width/depth as a function of accumulated area). */
    int    *dir;           /**< D8 flow direction code per RRI's convention: 1=E, 2=SE, 4=S, 8=SW,
                                 16=W, 32=NW, 64=N, 128=NE, 0 or -1 = outlet/no downstream cell.
                                 Powers of two so a cell's direction can (in the original Fortran
                                 tooling, not used here) be OR-combined across multiple flow paths;
                                 this port only ever uses a single direction per cell. */
    int    *domain;        /**< 0 = outside the modeled domain (nodata), 1 = interior, 2 = outlet
                                 (either dir==0/-1 in the input, or discovered as an outlet during
                                 rri_riv_idx_setting when a river cell's downstream neighbor turns
                                 out to be out of bounds or itself domain==0 -- see that function's
                                 doc for why this mutation happens in-place and why it can leave
                                 rri_riv_cellset::domain[k] stale relative to this array for the
                                 specific cell where the mutation occurs). */
    int    *riv;           /**< 1 = river cell. Set from `acc > riv_thresh` when riv_thresh > 0
                                 (main.c); NOT read from a file in this port's only supported
                                 configuration (rivfile_switch==0). */
    int    *land;          /**< Landuse id, 1-based, indexing into rri_config::lu's per-landuse
                                 parameter arrays. This port hardcodes land[]=1 everywhere
                                 (land_switch is not implemented -- see README.md). */

    double *width, *depth, *height;   /**< River channel geometry [m], meaningful only where riv==1:
                                            width and depth of the rectangular channel, and levee
                                            height above the bank (height==0 in the validated
                                            solo30s config, so the "no exchange while below levee
                                            height" case in rri_funcrs is never exercised there --
                                            see that function's doc). Computed from `acc` via the
                                            width_param_c/_s and depth_param_c/_s power laws in main.c
                                            (rivfile_switch==0 path; reading these from a file,
                                            rivfile_switch>=1, is not implemented). */
    double *len_riv;                    /**< River reach length per river cell [m]; == `length`
                                              (grid::length) everywhere in this port's parametric
                                              geometry path. */
    double *area_ratio;                 /**< River plan-view area / total cell area [-], i.e.
                                              `width * len_riv / area`; the fraction of a grid
                                              cell's footprint occupied by the channel itself, used
                                              by rri_hr2vr/rri_vr2hr to convert between river water
                                              depth and the actual water *volume* stored in that
                                              fraction of the cell. */
} rri_grid;

/* ---- per-landuse parameter table (1-based ids 1..num_of_landuse) -- */

/**
 * @brief Per-landuse hydraulic and infiltration parameters, indexed
 * 0-based internally (`n` entries) but referenced via 1-based landuse
 * ids everywhere else (grid::land, matching RRI_Input.txt's convention).
 *
 * Corresponds to the per-landuse arrays RRI_Read.f90 reads (`ns_slope`,
 * `soildepth`, `gammaa`, ... one value per line, `num_of_landuse` values
 * per line). `da`, `dm`, `infilt_limit` are DERIVED (computed in
 * rri_config_read from soildepth/gammaa/gammam/ksv, matching RRI_Read.f90's
 * own post-parse derivation), not read directly from the input file.
 */
typedef struct {
    int n;
    int    *dif;                 /**< 1 = diffusive-wave routing, 0 = kinematic-wave (single
                                       downstream-direction) routing, per landuse. Affects both
                                       river... no -- affects SLOPE routing only (via
                                       rri_slo_cellset::dif, copied per-cell from this); river
                                       routing in this port is unconditionally diffusive. */
    double *ns_slope, *soildepth, *gammaa;  /**< Manning's n for overland flow [s/m^(1/3)]; soil
                                                  column depth [m]; effective porosity [-]. See
                                                  kernels.h: rri_k_h2lev for how soildepth/gammaa
                                                  combine into the water-level correction. */
    double *ksv, *faif;            /**< Green-Ampt saturated hydraulic conductivity [m/s] and
                                         wetting-front suction head [m] (rri_infilt.c). */
    double *ka, *gammam, *beta;    /**< Lateral subsurface (Darcy) conductivity [m/s], matrix-flow
                                         porosity [-], matrix-flow power-law exponent [-] -- see
                                         kernels.h: rri_k_hq_slope. */
    double *ksg, *gammag, *kg0, *fpg, *rgl;  /**< Groundwater: bedrock presence flag (ksg>0 enables
                                                   groundwater for this landuse; also drives
                                                   rri_config::gw_switch globally), specific yield
                                                   [-], surface hydraulic conductivity [m/s],
                                                   conductivity depth-decay rate [1/m], and a
                                                   constant loss rate [m/s] (rri_gw_lose) -- see
                                                   kernels.h: rri_k_hg_calc. */
    double *da, *dm, *infilt_limit; /**< Derived (see struct doc above): soil saturation depth
                                          [m] (= soildepth*gammaa where ka>0), matrix-flow
                                          threshold depth [m] (= soildepth*gammam where ka>0 and
                                          gammam>0), and Green-Ampt cumulative infiltration cap [m]
                                          (= soildepth*gammaa where ksv>0). */
} rri_landuse;

/* ---- compressed river cellset -------------------------------------- */

/**
 * @brief Compressed 1D state for every active river cell -- the sparse
 * representation described in the file-level comment, specialized to
 * the river network's topology (a directed tree: each cell has exactly
 * one downstream neighbor, `down[k]`).
 *
 * Built once by rri_riv_idx_setting from `rri_grid`; every field here is
 * indexed `[k]`, `k` in `[0, count)`, in the (arbitrary but fixed)
 * row-major discovery order rri_riv_idx_setting visits river cells in.
 */
typedef struct {
    int count;                /**< Number of active river cells. */
    int    *idx2i, *idx2j;     /**< Map cellset index k back to grid row/column, for converting
                                     back to a full raster (rri_riv_idx2ij) or looking up other
                                     grid-level fields for cell k. */
    int    *down;              /**< down[k]: cellset index of k's single downstream neighbor.
                                     For an outlet cell (domain[k]==2), down[k]==k (points to
                                     itself) -- rri_qr_calc relies on this self-reference together
                                     with the domain==2 check to give outlet cells zero net lateral
                                     discharge rather than needing a separate no-neighbor branch. */
    double *dis;                /**< Planimetric distance from cell k to its downstream neighbor
                                      [m] (grid::dx, grid::dy, or their hypotenuse depending on
                                      flow direction) -- NOT adjusted for bed-elevation difference
                                      (i.e. this is horizontal distance, not slope distance). */
    double *zb;                  /**< River channel bed elevation for cell k [m], copied from
                                       grid::zb_riv (NEVER grid::zb -- see that field's doc) during
                                       rri_riv_idx_setting. */
    int    *domain;             /**< 0/1/2, captured from grid::domain at cell k's (i,j) location
                                      DURING the first index-building pass, before the
                                      neighbor-search pass that follows may mutate grid::domain to
                                      2 for a cell just discovered to be an outlet. This means
                                      domain[k] here can, for that one specific cell, be stale
                                      (still 1) relative to grid::domain (now 2) at the same
                                      location -- this is intentional, matching RRI_Sub.f90 /
                                      RRIpy's own behavior exactly (Fortran captures
                                      `domain_riv_idx(riv_count) = domain(i,j)` in an earlier loop
                                      than the one that can still mutate `domain(i,j)`), not a bug
                                      to "fix" by re-reading grid::domain after the fact. */
    double *width, *depth, *height, *area_ratio, *len_riv;  /**< Per-cell copies of the matching
                                                                   grid:: fields (see grid's doc). */
    double *dif;                 /**< Per-cell diffusive/kinematic flag, copied from the cell's
                                       landuse -- present for parity with the slope cellset's field
                                       of the same name, but unused: river routing in this port is
                                       unconditionally diffusive (rri_qr_calc has no kinematic
                                       branch), matching the validated solo30s config (`dif=1`). */
} rri_riv_cellset;

/* ---- compressed slope cellset (up to 4 neighbor directions) -------- */

/** @brief Number of distinct neighbor-direction slots stored per cell in
 * the 8-direction hillslope routing scheme (right, down, right-down,
 * left-down). A neighbor that would be reached by flowing in the
 * opposite sense (left, up, left-up, right-up) is represented by the
 * *same* slot with a negated flux, since "cell A's rightward neighbor is
 * B" and "cell B's leftward neighbor is A" describe the same edge -- see
 * rri_qs_calc's `dh >= 0.0` branch, which picks which of the two cells
 * sharing an edge is the source of the (always non-negative, per
 * kernels.h) head-gradient magnitude and negates the result for the
 * other. */
#define RRI_LMAX8 4

/**
 * @brief Compressed 1D state for every active hillslope cell (i.e. every
 * cell with grid::domain != 0 -- unlike the river cellset, this is
 * essentially the whole modeled domain, not a sparse subset of it).
 *
 * Each cell can exchange water with up to `lmax` neighbors (4 for
 * 8-direction routing, 2 for 4-direction -- see grid's `dif`/eight_dir
 * handling), stored as `RRI_LMAX8`-sized arrays of per-direction
 * neighbor index/distance/length even when `lmax==2` (the unused slots
 * are simply never populated/read). A cell using kinematic (single-
 * direction) routing instead uses `down_1d`/`dis_1d`/`len_1d`, its
 * D8-direction-following single downstream neighbor, regardless of
 * `lmax` -- see rri_qs_calc/rri_qg_calc's `dif_p == 0` branches.
 */
typedef struct {
    int count;                /**< Number of active hillslope cells. */
    int lmax;                  /**< 4 (eight_dir config == 1) or 2 (eight_dir == 0); how many of
                                     the RRI_LMAX8 direction slots are actually populated/used. */
    int    *idx2i, *idx2j;      /**< See rri_riv_cellset::idx2i/idx2j. */
    int    *domain;             /**< See rri_riv_cellset::domain -- captured similarly early,
                                      though slope index setting does not itself mutate
                                      grid::domain the way river index setting does, so no
                                      staleness caveat applies here. */
    double *zb;                   /**< SLOPE bed elevation for cell k [m], copied from grid::zb
                                        (never grid::zb_riv). */
    int    *down[RRI_LMAX8];       /**< down[l][k]: cellset index of cell k's neighbor in
                                         direction slot l, or -1 if that direction has no valid
                                         in-domain neighbor (edge of the raster, or the neighbor
                                         cell is domain==0). */
    double *dis[RRI_LMAX8];         /**< Planimetric distance to that neighbor [m]. */
    double *len[RRI_LMAX8];          /**< Shared-edge contour length for that direction [m] --
                                           the flux-carrying width of the interface between the two
                                           cells, distinct from `dis` (center-to-center distance).
                                           Precomputed geometrically per direction in
                                           rri_slo_idx_setting (RRI_Sub.f90's l1/l2/l3 constants). */
    int    *down_1d;                  /**< Kinematic-routing (dif==0) single downstream neighbor,
                                            following grid::dir like the river network does --
                                            see struct doc. */
    double *dis_1d, *len_1d;           /**< Distance/length for down_1d, analogous to dis/len. */

    double *ns_slope, *soildepth, *gammaa;         /**< Per-cell copies of the cell's landuse
                                                          parameters -- see rri_landuse's doc for
                                                          each field's meaning. */
    double *ksv, *faif, *infilt_limit;
    double *ka, *gammam, *beta, *da, *dm;
    double *ksg, *gammag, *kg0, *fpg, *rgl;
    int    *dif;                                     /**< 1 = diffusive-wave (up to lmax neighbor
                                                            directions), 0 = kinematic-wave
                                                            (down_1d only) routing for this cell. */
} rri_slo_cellset;

/* ---- rainfall time series (block-constant intensity, m/s) ---------- */

/**
 * @brief Rainfall forcing: a sequence of spatial rainfall-intensity
 * grids, each holding constant over a time block, on its own (typically
 * coarser) raster distinct from the model grid.
 *
 * Corresponds to the rainfile format read in RRI.f90 (and by
 * `load_rain` in main.c): repeated blocks of a header line (`time
 * nx_rain ny_rain`) followed by `ny_rain` rows of `nx_rain` values.
 * Units are converted from the file's mm/h to this struct's m/s at load
 * time (main.c: `load_rain`, `/ 3600.0 / 1000.0`).
 */
typedef struct {
    int nt;                /**< Number of time blocks. */
    double *t;               /**< Block END-times [s] (i.e. block `tt` is valid for time in
                                  `(t[tt-1], t[tt]]`), size nt. Matches the Fortran/RRIpy
                                  half-open-interval convention used to look up the active block
                                  in the main time loop (`t_rain[jt-1] < (time+ddt) <= t_rain[jt]`). */
    int ny_rain, nx_rain;
    double xllcorner, yllcorner, cellsize_x, cellsize_y;   /**< Georeference of the rain raster,
                                                                  independent of and generally
                                                                  coarser than the model grid's --
                                                                  main.c computes a per-model-row/
                                                                  column mapping (`rain_i`/`rain_j`)
                                                                  once at startup from these. */
    double *qp;              /**< [nt][ny_rain][nx_rain], row-major, rainfall intensity [m/s]. */
} rri_rain;

/* ---- Cash-Karp RK45 coefficients (verbatim from RRI.f90 / RRI_Mod2.f90,
 * ported into RRIpy's RRI_global_vars.py -- do not re-derive) --------- */

/**
 * @brief Coefficients for the embedded Runge-Kutta-Fehlberg (Cash-Karp)
 * 4(5) adaptive-step integrator RRI.f90 uses to advance river depth,
 * slope depth, and groundwater depth independently (each gets its own
 * accept/reject `ddt` sequence within a shared outer timestep `dt`).
 *
 * These are the standard Cash-Karp tableau values (a well-known embedded
 * RK pair choice, not RRI-specific) -- ported VERBATIM from RRI.f90's
 * `runge_mod` module (RRI_Mod2.f90) rather than re-derived, and must
 * stay that way: a different (even if also valid) embedded RK45
 * coefficient set would change the accept/reject step-size sequence and
 * silently produce a different-but-plausible-looking trajectory that
 * would not match the Fortran reference. See src/rri_rk.c for the actual
 * values and src/main.c's time loop for how `eps`/`ddt_min_*`/`safety`/
 * `pshrnk` drive the accept/reject decision -- in particular, note that
 * decision uses the SIGNED maximum error across cells (`errmax =
 * maxval(err)/eps` in Fortran), not `maxval(fabs(err))`; using `fabs`
 * there was a real bug found during full-length validation (see
 * README.md) -- a large negative error must NOT trigger a step shrink.
 */
typedef struct {
    double eps;           /**< Error tolerance controlling step acceptance: a step is accepted
                                when errmax/eps <= 1. Smaller eps -> smaller accepted steps ->
                                higher accuracy, lower throughput. */
    double ddt_min_riv, ddt_min_slo;  /**< Floor on the adaptive step size for river and slope
                                            integration respectively [s]; hitting this floor while
                                            still failing the error test raises a "stepsize
                                            underflow" error (matching Fortran's `stop
                                            'stepsize underflow'`) rather than looping forever. */
    double safety;         /**< Safety factor (<1) applied when shrinking a rejected step, so the
                                 next attempt is a bit more conservative than the error estimate
                                 alone would suggest. */
    double pgrow, pshrnk;   /**< Exponents for growing an overly-conservative accepted step
                                  (pgrow; NOT ACTUALLY USED for growth in this port's time loop --
                                  RRI.f90 also never grows ddt back up mid-run, only shrinks on
                                  rejection and resets to the nominal dt/dt_riv each outer
                                  timestep) and for shrinking a rejected one (pshrnk, used as
                                  `ddt * errmax**pshrnk`). */
    double errcon;          /**< Cash-Karp's standard "error control" threshold below which the
                                  growth-exponent formula would otherwise misbehave -- present for
                                  completeness with the Fortran source's module constants; unused
                                  since this port doesn't implement step growth (see pgrow's doc). */
    double b21;              /**< @name Cash-Karp tableau coefficients: the intermediate-stage
                                    weights (b*) for the 6 function evaluations, and the two
                                    linear combinations used for the 5th-order solution (c*) and
                                    the 5th-vs-4th-order error estimate (dc* = c* minus the
                                    embedded 4th-order weights). See any standard reference on the
                                    Cash-Karp method (e.g. Numerical Recipes) for the tableau these
                                    correspond to; RRI.f90's variable names are used verbatim here
                                    so the two can be compared line-by-line. @{ */
    double b31, b32;
    double b41, b42, b43;
    double b51, b52, b53, b54;
    double b61, b62, b63, b64, b65;
    double c1, c3, c4, c6;
    double dc1, dc3, dc4, dc5, dc6;
    /** @} */
} rri_rk_coeffs;

/** @brief Fill @p rk with the Cash-Karp coefficients verbatim from
 * RRI.f90 / RRI_Mod2.f90 (src/rri_rk.c). Call once at startup. */
void rri_rk_coeffs_init(rri_rk_coeffs *rk);

/* ---- config (mirrors RRI_Read.f90 / RRI_Input.txt field order) ----- */

/**
 * @brief Parsed contents of an RRI_Input.txt config file.
 *
 * Field order and grouping matches RRI_Read.f90's read sequence exactly
 * (rri_config_read reads the file in this same order) so the two can be
 * compared line-by-line against a real RRI_Input.txt. See that file's
 * comments (or README.md's config walkthrough) for the meaning of
 * individual switches; most are self-explanatory from RRI_Input.txt's
 * own inline `# comment` annotations, which this port's parser discards
 * (rri_io.c strips everything from `#` onward on each value line).
 *
 * NOT implemented, and rejected with an error at startup in main.c
 * rather than silently ignored if requested (rivfile_switch != 0,
 * sec_switch/dam_switch/div_switch/bound_*_switch/evp_switch != 0): see
 * README.md for why each is safe to omit for the validated solo30s
 * config, and rri_grid's field docs for what IS implemented in their
 * place (parametric, not file-based, river geometry).
 */
typedef struct {
    char rainfile[256], demfile[256], accfile[256], dirfile[256];
    int utm, eight_dir;      /**< utm: 1 = projected coordinates (dx=dy=cellsize directly), 0 =
                                   lat/lon (dx/dy computed geodesically via rri_hubeny_sub).
                                   eight_dir: 1 = 8-direction hillslope routing (lmax=4 per
                                   rri_slo_cellset), 0 = 4-direction (lmax=2). */
    int lasth;                /**< Simulation length [hours]. Overridable from the command line
                                    (main.c argv[2]) for fast partial-length comparison runs
                                    without editing RRI_Input.txt. */
    double dt, dt_riv;         /**< Outer coupling timestep [s] (river/slope/groundwater state is
                                     synchronized at this cadence) and the river integrator's
                                     initial (pre-adaptive-shrink) sub-step [s]; slope and
                                     groundwater integration both start their adaptive stepping at
                                     the full `dt` instead. */
    int outnum;                 /**< Target number of periodic output snapshots over the run
                                      (drives `out_dt`/`out_next` in the Fortran/RRIpy references);
                                      NOT used by this port, since periodic full-grid snapshot
                                      output (hs_, hr_, qr_, ... files) is not implemented -- see
                                      README.md. Parsed and stored for config-format completeness
                                      only. */
    double xllcorner_rain, yllcorner_rain, cellsize_rain_x, cellsize_rain_y;  /**< Rainfall
                                                                                    raster's own
                                                                                    georeference,
                                                                                    independent of
                                                                                    the model
                                                                                    grid's -- see
                                                                                    rri_rain. */

    double ns_river;    /**< Manning's n for the river channel [s/m^(1/3)], a single global value
                              (see kernels.h: rri_k_hq_riv). */
    int num_of_landuse;
    rri_landuse lu;

    int riv_thresh;                                                          /**< Flow-accumulation
                                                                                    threshold [grid
                                                                                    cells]: a cell
                                                                                    is a river cell
                                                                                    if acc >
                                                                                    riv_thresh. */
    double width_param_c, width_param_s, depth_param_c, depth_param_s;        /**< Power-law
                                                                                     coefficient/
                                                                                     exponent pairs
                                                                                     for parametric
                                                                                     channel width
                                                                                     and depth from
                                                                                     contributing
                                                                                     area (km^2):
                                                                                     `width =
                                                                                     width_param_c *
                                                                                     area_km2 **
                                                                                     width_param_s`,
                                                                                     similarly for
                                                                                     depth. */
    double height_param;                                                      /**< Levee height
                                                                                     [m] applied to
                                                                                     river cells
                                                                                     with acc >
                                                                                     height_limit_param;
                                                                                     0 in the
                                                                                     validated
                                                                                     solo30s config
                                                                                     (see
                                                                                     rri_grid::height's
                                                                                     doc). */
    int height_limit_param;                                                    /**< Flow-accumulation
                                                                                      threshold [grid
                                                                                      cells] above
                                                                                      which
                                                                                      height_param
                                                                                      applies. */

    int rivfile_switch;   /**< Must be 0 (parametric river geometry) -- any other value is
                                rejected at startup; see struct doc. */
    char widthfile[256], depthfile[256], heightfile[256];  /**< Parsed but unused (only relevant
                                                                  to rivfile_switch>=1, not
                                                                  implemented). */

    int init_slo_switch, init_riv_switch, init_gw_switch, init_gampt_ff_switch;   /**< Initial-
                                                                                        condition-
                                                                                        from-file
                                                                                        switches;
                                                                                        NOT
                                                                                        implemented
                                                                                        --
                                                                                        every run
                                                                                        starts from
                                                                                        zero
                                                                                        depth/
                                                                                        storage
                                                                                        regardless
                                                                                        of these
                                                                                        values. */
    char initfile_slo[256], initfile_riv[256], initfile_gw[256], initfile_gampt_ff[256];

    int bound_slo_wlev_switch, bound_riv_wlev_switch;    /**< Must be 0 -- boundary conditions not
                                                                implemented; rejected at startup if
                                                                nonzero. */
    char boundfile_slo_wlev[256], boundfile_riv_wlev[256];
    int bound_slo_disc_switch, bound_riv_disc_switch;      /**< Must be 0, see above. */
    char boundfile_slo_disc[256], boundfile_riv_disc[256];

    int land_switch;    /**< Must be 0 in practice: this port hardcodes uniform landuse
                              (grid::land[]=1 everywhere, main.c) regardless of this switch's
                              value or num_of_landuse. Not rejected at startup (unlike the
                              switches above) since a single-landuse config with land_switch=0,
                              as in solo30s, is unaffected either way -- but a config with
                              num_of_landuse>1 relying on a real landuse raster will silently get
                              landuse-1 parameters everywhere instead of erroring. */
    char landfile[256];
    int dam_switch;       /**< Must be 0 -- dam operation not implemented; rejected at startup. */
    char damfile[256];
    int div_switch;        /**< Must be 0 -- diversion not implemented; rejected at startup. */
    char divfile[256];

    int evp_switch;    /**< Must be 0 -- evapotranspiration not implemented; rejected at
                             startup. */
    char evpfile[256];
    double xllcorner_evp, yllcorner_evp, cellsize_evp_x, cellsize_evp_y;

    int sec_length_switch;     /**< Parsed but unused (only relevant together with custom
                                     cross-sections, not implemented). */
    char sec_length_file[256];
    int sec_switch;              /**< Must be 0 -- custom river cross-sections not implemented
                                       (rri_hr2vr/rri_vr2hr only support the rectangular-channel
                                       fallback); rejected at startup. */
    char sec_map_file[256], sec_file[256];

    int outswitch_hs, outswitch_hr, outswitch_hg, outswitch_qr, outswitch_qu,
        outswitch_qv, outswitch_gu, outswitch_gv, outswitch_gampt_ff, outswitch_storage;  /**<
        Parsed for format completeness; only outswitch_storage's corresponding output
        (storage.dat, via rri_storage_calc) and the separate hydro_switch-gated hydro.txt/
        hydro_hr.txt are actually written -- see README.md. */
    char outfile_hs[256], outfile_hr[256], outfile_hg[256], outfile_qr[256],
         outfile_qu[256], outfile_qv[256], outfile_gu[256], outfile_gv[256],
         outfile_gampt_ff[256], outfile_storage[256];

    int hydro_switch;      /**< 1 enables hydro.txt/hydro_hr.txt (per-location discharge/depth
                                 time series, at the points listed in location_file) -- this and
                                 storage.dat are the only two outputs this port writes; they're
                                 also exactly what's needed to validate against the Fortran
                                 reference (see README.md). */
    char location_file[256];

    int gw_switch;  /**< DERIVED (not read from the file directly): 1 if any landuse has ksg>0,
                          computed in rri_config_read exactly as RRI_Read.f90 computes it. Gates
                          whether the groundwater RK45 sub-loop (rri_funcg et al.) runs at all
                          each outer timestep (src/main.c). */
} rri_config;

/**
 * @brief Parse an RRI_Input.txt-format file into @p cfg.
 * @param path Path to the config file.
 * @param cfg  Output; on success, caller owns the allocated per-landuse
 *             arrays inside cfg->lu and must call rri_config_free() when done.
 * @return 0 on success; nonzero (with a diagnostic on stderr) on a
 *         missing file, format-version mismatch, or malformed field --
 *         this port fails loudly rather than guessing at intent for a
 *         malformed config.
 */
int rri_config_read(const char *path, rri_config *cfg);
/** @brief Free the per-landuse arrays allocated by rri_config_read. */
void rri_config_free(rri_config *cfg);

/* ---- GIS grid I/O (RRI_Sub.f90: read_gis_int / read_gis_real) ------ */

/**
 * @brief Read an ESRI ASCII grid file's numeric body into @p out, after
 * validating its 6-line header (ncols/nrows/xllcorner/yllcorner/
 * cellsize/NODATA_value) against the expected @p ny / @p nx /
 * @p xllcorner / @p yllcorner / @p cellsize (0.01 absolute tolerance on
 * the georeference values, matching RRI_Sub.f90's own tolerance).
 *
 * Row tokenizing accepts BOTH whitespace- and comma-separated values
 * (some real-world grids -- e.g. solo30s's `*_mod.txt` variants -- use
 * commas; Fortran's list-directed `read(unit,*)` accepts commas as a
 * field delimiter natively, so a C port needs to as well rather than
 * assuming one convention).
 *
 * @param path       Grid file path.
 * @param ny, nx     Expected row/column count.
 * @param xllcorner, yllcorner, cellsize  Expected georeference (checked, not just informational).
 * @param out        Output buffer, (ny*nx) doubles, row-major, row 0 =
 *                   northernmost (file order, unmodified).
 * @return 0 on success; nonzero on a header mismatch, short file, or I/O error.
 */
int rri_read_gis_real(const char *path, int ny, int nx, double xllcorner,
                       double yllcorner, double cellsize, double *out);
/** @brief Integer-valued counterpart to rri_read_gis_real; same header
 * format and validation, values parsed via `atof` then truncated to int
 * (not `atoi` directly -- grid files sometimes write integer-valued
 * fields with a trailing ".0"). */
int rri_read_gis_int(const char *path, int ny, int nx, double xllcorner,
                      double yllcorner, double cellsize, int *out);
/** @brief Read only a grid file's header (no value-count or tolerance
 * checks) -- used to discover a domain's ny/nx/georeference before
 * allocating grids sized to match, and before the tolerance-checked
 * reads above are meaningful. */
int rri_read_gis_header(const char *path, int *ny, int *nx, double *xllcorner,
                         double *yllcorner, double *cellsize);

/** @brief Geodesic (WGS84 ellipsoid, Hubeny's formula) distance in
 * metres between two lat/lon points in decimal degrees. Used to convert
 * a domain's lat/lon extent into metric dx/dy when rri_config::utm==0.
 * Fortran reference: RRI_Sub.f90, `hubeny_sub`. */
double rri_hubeny_sub(double x1_deg, double y1_deg, double x2_deg, double y2_deg);

/* ---- index setting (RRI_Sub.f90: riv_idx_setting / slo_idx_setting) */

/**
 * @brief Build the compressed river cellset from @p grid, discovering
 * outlet cells (mutating grid::domain / grid::dir to 2 / 0 in place for
 * any river cell whose downstream neighbor is out of bounds or itself
 * domain==0) along the way.
 *
 * This in-place mutation matches RRI_Sub.f90 / RRIpy's own semantics --
 * outlet discovery is a side effect of walking the flow-direction graph
 * here, not a separate pass -- see rri_riv_cellset::domain's doc for the
 * resulting (intentional) staleness caveat.
 *
 * @param grid  In/out: grid::domain and grid::dir may be mutated as above.
 * @param rc    Out: populated cellset; caller must rri_riv_cellset_free() when done.
 * @return 0 on success; nonzero (with a diagnostic on stderr) if a flow
 *         direction code is invalid or a river cell's resolved
 *         downstream neighbor is not itself marked as a river cell
 *         (grid::riv), both of which indicate an inconsistent input
 *         dataset rather than something recoverable.
 */
int rri_riv_idx_setting(rri_grid *grid, rri_riv_cellset *rc);
/** @brief Build the compressed hillslope cellset from @p grid and the
 * per-landuse parameter table @p lu. Does not mutate @p grid.
 * @param eight_dir  1 for 8-direction routing (lmax=4), 0 for 4-direction (lmax=2).
 * @return 0 on success; nonzero if @p eight_dir is neither 0 nor 1, or a
 *         flow-direction code is invalid (needed for the kinematic
 *         single-direction neighbor, down_1d). */
int rri_slo_idx_setting(rri_grid *grid, const rri_landuse *lu, int eight_dir,
                         rri_slo_cellset *sc);

/** @brief Free the arrays allocated by rri_riv_idx_setting. */
void rri_riv_cellset_free(rri_riv_cellset *rc);
/** @brief Free the arrays allocated by rri_slo_idx_setting. */
void rri_slo_cellset_free(rri_slo_cellset *sc);

/**
 * @name Cellset <-> full-grid conversion
 *
 * Convert between a compressed 1D per-cellset array and a full (ny, nx)
 * raster. Called at the start/end of each outer timestep's river/slope/
 * groundwater sub-loop in main.c (the RK45 integration itself operates
 * entirely on the compressed idx representation; the raster form is
 * only needed for cross-cutting operations that aren't naturally
 * per-cellset, like rri_funcrs's river<->slope exchange, which walks the
 * full grid because it needs BOTH a cell's slope depth AND its
 * collocated river depth together).
 * @{
 */
/** @brief Gather: `idx_out[k] = grid[idx2i[k] * nx + idx2j[k]]` for every k. */
void rri_riv_ij2idx(const rri_riv_cellset *rc, const double *grid_ny_nx, int nx, double *idx_out);
/** @brief Scatter: `grid_out[idx2i[k] * nx + idx2j[k]] = idx[k]` for every k;
 * grid_out is zeroed first (cells outside the cellset read back as 0, not
 * left stale from a previous call). */
void rri_riv_idx2ij(const rri_riv_cellset *rc, const double *idx, int ny, int nx, double *grid_out);
/** @brief Slope-cellset counterpart to rri_riv_ij2idx. */
void rri_slo_ij2idx(const rri_slo_cellset *sc, const double *grid_ny_nx, int nx, double *idx_out);
/** @brief Slope-cellset counterpart to rri_riv_idx2ij. */
void rri_slo_idx2ij(const rri_slo_cellset *sc, const double *idx, int ny, int nx, double *grid_out);
/** @} */

/* ---- river cross-section (rectangular fallback only; sec_map>0 custom
 * cross-sections are NOT implemented -- solo30s and the tested synthetic
 * domains both use sec_switch=0, i.e. this path exclusively). ------- */

/**
 * @name River water depth <-> stored volume (rectangular channel only)
 *
 * Fortran reference: RRI_Section.f90, `hr2vr`/`vr2hr`, the
 * `sec_map_idx(k)<=0` (no custom cross-section) branch only -- the
 * lookup-table branch for surveyed cross-sections is not implemented.
 * For a rectangular channel these are simple linear conversions via the
 * cell's river plan-area fraction; kept as separate named functions
 * (rather than inlined at call sites) so a future custom-cross-section
 * implementation has one place to add the general (non-rectangular)
 * branch back in, matching the Fortran functions' role exactly.
 * @{
 */
/** @param hr Water depth in the channel [m]. @param area Grid cell area [m^2].
 * @param area_ratio Channel plan-area fraction of the cell [-] (rri_grid::area_ratio
 * / rri_riv_cellset::area_ratio for this cell). @return Stored volume [m^3]. */
double rri_hr2vr(double hr, double area, double area_ratio);
/** @brief Inverse of rri_hr2vr. @param vr Stored volume [m^3]. @return Water depth [m]. */
double rri_vr2hr(double vr, double area, double area_ratio);
/** @} */

/* ---- storage / mass balance (RRI_Sub.f90: storage_calc) ------------ */

/** @brief Total water storage, broken down by compartment, at one instant. */
typedef struct {
    double ss;  /**< Slope (hillslope surface) water storage [m^3]: sum(hs * area) over
                     in-domain cells. */
    double sr;  /**< River channel water storage [m^3]: sum(rri_hr2vr(hr)) over river cells,
                     included only when @p riv_thresh (rri_storage_calc's argument) is >= 0 --
                     NOT only when it's exactly 0. Using `== 0` there instead of `>= 0` was a
                     real, separate bug found during this port's development (a regression test,
                     tests/test_storage_calc.c, exists specifically for this): with the
                     validated solo30s config's riv_thresh=100, an `== 0` check would silently
                     exclude the entire river network's storage from every mass-balance
                     computation without crashing or producing an obviously wrong number, only a
                     water balance that doesn't close. */
    double si;  /**< Cumulative Green-Ampt infiltration storage [m^3]: sum(gampt_ff * area). */
    double sg;  /**< Groundwater storage [m^3], NEGATIVE of the deficit-weighted sum
                     (`-sum(hg * gammag * area)`) -- see kernels.h: rri_k_hg_calc's doc on hg's
                     deficit (not depth) sign convention; this is where that convention gets
                     converted into an actual (positive-when-more-water) storage contribution. */
} rri_storage;

/**
 * @brief Compute the instantaneous total water storage across all
 * compartments, for the mass-balance check written to storage.dat each
 * outer timestep (src/main.c).
 * @param riv_thresh  See rri_storage::sr's doc -- controls whether river
 *                     storage is included; pass rri_config::riv_thresh.
 */
rri_storage rri_storage_calc(const rri_grid *grid, const double *hs,
                              const double *hr, const double *hg,
                              const double *gampt_ff, const rri_slo_cellset *sc,
                              const rri_riv_cellset *rc, int riv_thresh);

/* ---- physics kernels (src/rri_riv.c, rri_slope.c, rri_gw.c, rri_infilt.c,
 * rri_rivslo.c) ------------------------------------------------------- */

/**
 * @name River routing
 *
 * See src/rri_riv.c for full documentation of each function; declared
 * here for linkage. Fortran reference: RRI_Riv.f90.
 * @{
 */
void rri_qr_calc(const rri_riv_cellset *rc, const double *hr_idx, double ns_river, double *qr_idx);
void rri_funcr(const rri_riv_cellset *rc, const double *vr_idx, double ns_river, double area,
                double *hr_idx, double *fr_idx, double *qr_idx, double *qr_sum_scratch);
/** @} */

/**
 * @name Hillslope routing
 * See src/rri_slope.c. Fortran reference: RRI_Slope.f90.
 * @{
 */
void rri_qs_calc(const rri_slo_cellset *sc, const double *hs_idx, double area, double *qs_idx[RRI_LMAX8]);
void rri_funcs(const rri_slo_cellset *sc, const double *hs_idx, const double *qp_t_idx,
                double area, double *fs_idx, double *qs_idx[RRI_LMAX8]);
/** @} */

/**
 * @name Groundwater
 * See src/rri_gw.c. Fortran reference: RRI_GW.f90.
 * @{
 */
void rri_qg_calc(const rri_slo_cellset *sc, const double *hg_idx, double area, double *qg_idx[RRI_LMAX8]);
void rri_funcg(const rri_slo_cellset *sc, const double *hg_idx, double area,
                double *fg_idx, double *qg_idx[RRI_LMAX8]);
void rri_gw_recharge(const rri_slo_cellset *sc, double dt, double *hs_idx,
                      double *gampt_ff_idx, double *hg_idx);
void rri_gw_lose(const rri_slo_cellset *sc, double dt, double *hg_idx);
void rri_gw_exfilt(const rri_slo_cellset *sc, double dt, double *hs_idx,
                    double *gampt_ff_idx, double *hg_idx);
/** @} */

/** @brief Green-Ampt infiltration; see src/rri_infilt.c. Fortran reference: RRI_Infilt.f90. */
void rri_infilt(const rri_slo_cellset *sc, double dt, double *hs_idx,
                 double *gampt_ff_idx, double *gampt_f_idx);

/** @brief River<->slope water exchange; see src/rri_rivslo.c. Fortran reference: RRI_RivSlo.f90. */
void rri_funcrs(const rri_grid *g, const rri_riv_cellset *rc, double dt, double *hr, double *hs);

#ifdef __cplusplus
}
#endif
#endif /* RRI_H */
