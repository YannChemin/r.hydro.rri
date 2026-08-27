# r.hydro.rri: native-GRASS rewrite plan

Status: **Increment 1 (static input + index setting) implemented and
validated on both a synthetic domain and the real dem_jamuna DEM.** See
"Progress" at the bottom for exactly what that means, what's next, and a
real data-quality issue increment 1's own consistency checking surfaced
on real data. The rest of this document below "Progress" is still the
original design (§1-7), left as written since nothing in it turned out
to be wrong once implementation started.

## Progress

**Increment 1 implemented**: `main.c` (root of this repo, not `engine/`)
+ `rri_setup.c` (a copy of the vendored engine's file, taken out of the
`engine/`-is-frozen symlink for this increment -- see "next steps"
below) + `rri_geo.c` (a from-scratch extraction of just
`rri_hubeny_sub`, so this module links against zero ASCII-I/O code, not
even an unused object file). Reads `elevation=`, `drainage=`,
`accumulation=` directly via `Rast_open_old`/`Rast_get_d_row` -- no
files written or read outside GRASS's own raster API. `drainage=` is
expected pre-computed (by `r.watershed`/`r.watershed.opencl`, run
separately) -- auto-invoking it from within this module is deferred, not
attempted this pass. Single land-use class only in this increment
(matches the vendored engine's own existing simplification); the
multi-class LULC table (item 3 of the original follow-up request) is
NOT yet implemented -- next increment.

**Validated in two stages, per this project's own established discipline
of not trusting a synthetic-only check:**

1. **Synthetic 10x10 domain** (same domain, same `r.watershed -s` output,
   as the old ASCII-path pytest suite already validated end-to-end
   against the compiled RRI engine): the native module reproduces the
   ASCII path's exact known numbers -- `riv_count=16 slo_count=100`,
   `dx=110479.427 dy=110582.711 area=12217114553.479` -- bit-for-bit
   identical. Automated as `tests/native_io_test.py` (passing, run
   alongside the existing suite: 5/5 pass total).
2. **Real dem_jamuna DEM** (2.86M cells, UTM, on `yann@10.42.0.89`,
   per the user's explicit go-ahead to use this dataset once increment 1
   was validated synthetically first): built and ran the native module
   there over SSH. `r.watershed.opencl elevation=dem_jamuna threshold=500`
   completed in ~4s on the real GPU host. Running the native module
   against its output found a **real, previously-unknown data-quality
   issue**, not a bug in this module's own logic:

   `rri_riv_idx_setting`'s own consistency check (a river cell's
   downstream neighbor must also be a river cell) failed at grid cell
   (row 629, col 0) -- a west-edge-of-region cell. Its immediate
   upstream neighbor at (630, 0) has `accumulation=3044` (river, given
   `riv_thresh=500`) and flows into (629, 0) via `drainage=2` (north).
   But (629, 0) itself has `accumulation=19` -- implausibly low given it
   receives ~3044 cells' worth of upstream flow -- and `drainage=0`
   (no further downstream cell, i.e. r.watershed.opencl's own boundary/
   outlet marker). This is exactly the kind of inconsistency
   `rri_riv_idx_setting`'s consistency check exists to catch (per its
   own doc comment in `rri.h`) -- **this is the check working as
   intended**, not a failure of this increment's own code.

   Root cause not yet determined -- two candidate explanations, neither
   confirmed: (a) a genuine `r.watershed.opencl` edge-of-region
   accumulation artifact (map-boundary cells' true contributing area
   extends past the mapped region, a known category of issue for any
   D8 flow-accumulation algorithm, not unique to this one -- stock
   `r.watershed` flags this with a negative-accumulation "likely an
   underestimate" convention that this module's `main.c` already
   discards via `fabs()`, on the assumption it only ever meant
   "underestimate," not "wildly wrong"; that assumption may not hold at
   *this* specific kind of boundary case); or (b) something specific to
   `dem_jamuna`'s own boundary geometry/nodata pattern. **Not
   investigated further this pass** -- flagging clearly rather than
   guessing. Verified the direction-code convention itself is not the
   cause: read `r.watershed.opencl`'s `flow_direction.c` directly and
   confirmed its 8-code table (`off_dr`/`off_dc`/`codes` arrays) is
   identical to stock `r.watershed`'s counter-clockwise-from-NE
   convention, so `drainage_to_rri_dir`'s reclass table is not
   misinterpreting anything here.

**dem_jamuna boundary anomaly, closed out (bounded check as instructed,
not a rabbit hole)**: the anomalous cell (row 629, col 0) is at `col=0`
out of `cols=1062` -- literally the region's west boundary by
construction (confirmed directly: `g.region raster=dem_jamuna` reports
`cols: 1062`, so column index 0 is the first/westmost column). This is
consistent with a plausible, mundane explanation: a DEM clipped to a
study-area boundary cannot know its true upstream contributing area
outside that boundary, so a boundary cell's computed accumulation can be
an artifact of the clip, not a routing bug. **Recorded as a known
caveat, not fixed or investigated further**: `riv_thresh`-based river
classification should not be fully trusted on cells at a clipped
domain's edge without a wider-than-needed DEM extent or an explicit
edge-buffer/mask strategy -- a real, documented limitation of any
DEM-clip-then-classify pipeline (not unique to this module or to
`r.watershed.opencl`), left for whoever prepares a real study-area DEM
to be aware of, not something this module's own code should try to
paper over.

**Increment 2 implemented and validated**: a `rain=` option reads a
single static precipitation raster (mm/h), converts it via
`Rast_get_d_row`, and indexes it into slope-idx space via
`rri_slo_ij2idx` (unchanged, reused). Deliberately scoped narrow: a
single static raster, not yet a `rain_strds=` time series, and NOT YET
wired into a time loop at all -- this increment only proves the
read+convert+index path is correct in isolation. Validated two ways: a
uniform raster (12.5 mm/h) produces `qp_t_idx` mean == 12.5 exactly, and
-- a stronger check, since a uniform input would pass even with a
broken index mapping as long as the cell *count* happened to be right
-- a spatially-varying raster (`col()`, domain mean 5.5 confirmed via
`r.univar`) produces `qp_t_idx` mean == 5.5 exactly. Automated as
`tests/native_io_test.py::test_rain_read_and_index_matches_raster_mean`
(6/6 tests passing overall now).

**Increment 3 implemented and validated**: `rain_strds=` resolves a
STRDS's `(map, start_time, end_time)` list by shelling to `t.rast.list`
(§3's (a)/(b) choice resolved as (b): linking `libtgis` directly from a
`Module.make` build was not attempted this pass -- `t.rast.list` is a
small, once-per-run subprocess call, not a per-timestep or per-value
ASCII round-trip, so it doesn't reintroduce the "no ASCII files"
violation this rewrite exists to fix; actual rain VALUES still come
only from `Rast_get_d_row`). Every map must be interval-registered
(explicit start AND end) -- fails loudly, not silently, on an
instant-registered map, same requirement the superseded Python driver
had. Sorts chronologically (insertion sort; a forcing series' timestep
count is small, no need for anything fancier), computes elapsed seconds
relative to the series' own first start (matching RRI.f90's `t_rain`
convention), and re-runs increment 2's now-shared
`read_and_index_forcing_raster` helper once per resolved timestep.

Validated against a synthetic 3-day STRDS with known, distinct
per-day values (5.0, 10.0, 20.0 mm/h): all three timesteps resolved in
correct chronological order, `elapsed_s` exactly 86400/172800/259200,
and each timestep's `qp_t_idx` mean exactly matches its input value.
Automated as
`tests/native_io_test.py::test_rain_strds_resolves_and_indexes_every_timestep`
(7/7 tests passing overall now).

**Still NOT wired into a time loop** -- `rain=`/`rain_strds=` both stop
at "prove the input is read and indexed correctly," same as before this
increment. The RK45 loop itself remains the next, larger, higher-risk
increment, deliberately kept separate.

**Next steps, in order** (per this document's own §5 validation
discipline -- do not skip ahead):

1. **The RK45 time loop itself** -- this is the big, high-risk one.
   Reuse the *unchanged* `rri_funcr`/`rri_funcs`/`rri_funcg`/
   `rri_funcrs`/`rri_infilt`/`rri_rk_coeffs_init` from the vendored
   engine, per §3's table, and the vendored engine's own `main.c`
   (~330 lines, `engine/src/main.c` lines ~330-682) as the reference
   control-flow to adapt -- but treat this as its own careful,
   dedicated increment, not something to rush alongside forcing/output
   plumbing. Validate incrementally within this increment too (e.g. a
   handful of timesteps' mass balance before a full run), not as one
   untested leap -- this is exactly the class of code (adaptive
   step-size control, six-stage embedded RK) where this project has
   twice already found real, hard-to-spot bugs (the signed-vs-fabs
   error norm; the zb/zb_riv conflation) that only surfaced after
   sustained validation, not a quick look.
2. Output: hydrograph + storage to GRASS DB tables via direct DBMI
   linkage (verify this is practical from a `Module.make` build before
   committing to it over shelling out to `db.execute`); periodic
   full-grid STRDS output requires first implementing the underlying
   feature at all (the vendored engine never had it, ASCII or native --
   see `engine/VENDORED.md`).
3. Multi-land-use LULC table (`table_input.c`-style pattern, not yet
   looked at this pass).
4. `r.watershed.opencl` auto-invocation from within this module (G_spawn
   or documented as a required separate step -- undecided).
5. Once the native path covers the same ground the ASCII path did and is
   validated to the same standard, delete `r.hydro.rri.py`, `engine/`,
   and the `Makefile`'s `engine:` target, and switch the `Makefile` to
   `Module.make` (`PGM = r.hydro.rri`, `LIBES = $(GISLIB) $(RASTERLIB)
   $(MATHLIB)`, mirroring `r.watershed.opencl`'s own Makefile) --
   NOT done yet; this increment was still built/tested via ad hoc `gcc`
   invocations (see `tests/native_io_test.py`'s own docstring) precisely
   to avoid disturbing the still-referenced old Makefile mid-transition. This supersedes the earlier
"Python wrapper shells out to a vendored `rri_cpu` subprocess reading/
writing ASCII files" architecture (still present in this directory as of
this writing — `r.hydro.rri.py`, `engine/`, `Makefile`'s `engine:` target
— see "What happens to the existing code" below). Per explicit user
direction: **no ASCII intermediate files anywhere in the pipeline.**
DEM/land-use/river-geometry read directly from GRASS rasters, forcing
read directly from a registered STRDS's timestep rasters, all output
written directly as GRASS rasters/STRDS/DB tables, config from `G_parser`
options rather than a generated `RRI_Input.txt`.

## 1. What's confirmed reusable (verified, not assumed)

Checked by grepping every file's `#include`s and searching for
`fopen`/`fread`/`fscanf`/`fprintf` (2026-08-27, against the vendored
`engine/` copy of RRI.opencl):

| File | I/O coupling found | Verdict |
|---|---|---|
| `include/rri/kernels.h` | none (`<math.h>` only) | reusable unchanged |
| `src/rri_riv.c` | none | reusable unchanged |
| `src/rri_slope.c` | none | reusable unchanged |
| `src/rri_gw.c` | none | reusable unchanged |
| `src/rri_infilt.c` | none | reusable unchanged |
| `src/rri_rivslo.c` | one `fprintf(stderr, ...)` diagnostic on an unhandled-case branch, not data I/O | reusable unchanged |
| `src/rri_rk.c` | none | reusable unchanged |
| `src/rri_opencl.c` | operates only on `rri_riv_cellset`/`rri_slo_cellset` pointers + the two kernel-source files (already made runtime-relocatable, see `engine/VENDORED.md`) | reusable unchanged |
| `src/rri_setup.c` | takes an already-populated `rri_grid`/`rri_landuse`, does index/geometry math only | reusable unchanged (verify: has not been grepped as closely as the solver files above — do this before relying on it) |
| `src/rri_io.c` | **is** the ASCII I/O layer (`rri_config_read`, `rri_read_gis_real/int/header`) | **replace entirely** |
| `src/main.c` | config read, static-grid reads, rain-file parsing (~line 97-227), location-file parsing (~line 305-317), `hydro.txt`/`hydro_hr.txt`/`storage.dat` writes (~line 395-682) | **replace entirely** (the RK45 time-loop control flow *between* those I/O calls is the reusable part — see §3) |

The function-signature boundary is exact and clean: everything downstream
of grid/cellset population (`rri_riv_idx_setting`, `rri_slo_idx_setting`,
`rri_funcr`/`rri_funcs`/`rri_funcg`/`rri_funcrs`/`rri_infilt`,
`rri_storage_calc`, the `rri_cl_*` OpenCL dispatchers) takes a
`const rri_grid *`/`rri_riv_cellset *`/`rri_slo_cellset *` and never
touches a file path. Confirmed at `include/rri/rri.h`'s declarations
(`rri_riv_idx_setting(rri_grid *grid, rri_riv_cellset *rc)` etc.) --
these structs are the actual interface to preserve; only *how they get
filled in* changes.

## 2. New module architecture

Pattern A (compiled C, per this user's GRASS addon conventions):

```
r.hydro.rri/
├── Makefile              # PGM=r.hydro.rri, LIBES=$(GISLIB) $(MATHLIB) $(RASTERLIB) $(TEMPORALLIB) $(DBMILIB) ..., Module.make
├── main.c                # G_parser entry point, replaces engine/src/main.c
├── io_grass.c/.h         # NEW: the native-GRASS I/O layer, replaces rri_io.c
├── (reused unchanged from engine/, moved up or kept included via EXTRA_INC)
│   rri_setup.c, rri_riv.c, rri_slope.c, rri_gw.c, rri_infilt.c,
│   rri_rivslo.c, rri_rk.c, rri_opencl.c, kernels.h, rri.h, opencl.h
├── cl/rri_kernels.cl      # unchanged
└── tests/                 # new: pytest + grass.tools, native-GRASS equivalents
                            # of RRI.opencl's own synthetic-domain fixtures
```

`RRI.opencl`'s own CMake build (and this addon's now-superseded
`engine/`+Makefile-cmake-hook) goes away for the solver/kernel files --
they become ordinary sources in a GRASS `Module.make` build alongside
`io_grass.c`. OpenCL linking follows the same pattern GRASS's own
`r.watershed.opencl` already uses (check that addon's Makefile for the
`EXTRA_LIBS`/`EXTRA_INC` OpenCL linkage convention, don't re-derive it
from scratch).

## 3. New I/O layer: function-by-function replacement plan

Each ASCII-coupled entry point in the old `main.c`/`rri_io.c`, and its
native-GRASS replacement:

| Old (ASCII) | New (GRASS-native) |
|---|---|
| `rri_config_read(RRI_Input.txt)` | `G_parser()` options -- see §4 for the option list. No generated text file; values land directly in a `rri_config` struct built from `G_OPT_*` option strings the normal `atof`/`atoi` way GRASS modules already do it. |
| `rri_read_gis_header`/`rri_read_gis_real` (DEM/acc/dir/landuse) | `Rast_open_old()` + `Rast_get_row()` per row into the same `rri_grid` arrays -- current GRASS region (`G_OPT_R_ELEV`-style inputs) defines `ny`/`nx`/`xllcorner`/`yllcorner`/`cellsize` via `Rast_window_rows()`/`Rast_window_cols()`/`G_get_window()`, replacing the ASCII header's role. Direction-convention reclassing (`r.watershed`'s CCW-from-NE -> RRI's D8 bitmask, already implemented once in Python in the now-superseded `r.hydro.rri.py`) needs porting to C here, or done as a pre-pass via `r.mapcalc` before this module runs (simpler, reuses working logic, GRASS-idiomatic composition of one module calling another via the CLI rather than reimplementing reclassing in C -- prefer this unless a concrete reason emerges not to). |
| rain-file block parser (`main.c` ~97-227) | Resolve, once at startup, which raster map covers each simulation timestep from the forcing STRDS. Two options, pick one and document why: (a) link GRASS's temporal C library (`libtgis`) directly and query it from C: check whether this is practical/stable to link from a Module.make-built addon (uncertain -- verify before committing to this path); (b) have a thin Python/`G_parser` pre-step (or the same module's own startup code shelling to `t.rast.list`) resolve the STRDS's (map name, start, end) list once, pass the resolved list of raster map names + elapsed-seconds timestamps into the C engine as a repeated option or a small manifest read via `G_parser`-adjacent means (NOT a written ASCII forcing file with rain VALUES in it -- only a lightweight map-name/timestamp manifest, if unavoidable, is a meaningfully different thing from the banned ASCII rain-block format; prefer avoiding even that if (a) is workable). Whichever is chosen, actual rain VALUES are read timestep-by-timestep via `Rast_get_row()` directly from the resolved raster map, not from any intermediate file. |
| `location.txt` (outlet stations) | A `G_OPT_V_INPUT`/`G_OPT_STRDS_INPUT`-adjacent option, or reuse of the existing `outlet=` heuristic (peak accumulation) computed in C from the already-open accumulation raster -- no file. |
| `hydro.txt`/`hydro_hr.txt` (outlet discharge time series) | Write directly to a DB table via GRASS's DBMI C API (`db_open_database`/`db_execute_immediate`, or the simpler path of shelling to `db.execute`/`v.db.*` if linking `libdbmi` directly proves impractical -- decide after trying the direct-link path first, don't default to shelling out). |
| `storage.dat` (mass-balance log) | Same DB-table treatment as the hydrograph, or a second table -- no ASCII file. |
| Periodic full-grid output (`hs_*`/`hr_*`/etc, if/when implemented -- currently NOT implemented even in the ASCII version, see `engine/VENDORED.md`) | `Rast_open_new()` + `Rast_put_row()` per output timestep, then `t.register` (a normal module call, not an ASCII round-trip) to build an STRDS incrementally as the run progresses. This item additionally requires implementing periodic-grid output in the solver-adjacent code at all, which the ASCII version never had either -- do not treat "supporting STRDS output" as free just because the sink changed; the source-side feature (deciding what to write and when, per `outswitch_*`) still needs building. |

## 4. G_parser option list (replaces `RRI_Input.txt` fields)

Sketch, to refine during implementation -- one option/flag per config
field that has a GIS-native equivalent, grouped the same way
`RRI_Input.txt` groups them:

- `elevation=` (`G_OPT_R_ELEV`), `landuse=` (`G_OPT_R_INPUT`, optional),
  `direction=`/`accumulation=` (`G_OPT_R_INPUT`, optional -- reused
  from the superseded Python driver's `drainage_input=`/
  `accumulation_input=` pass-through idea, see §6)
- `rain_strds=` (`G_OPT_STRDS_INPUT`), `pet_strds=` (optional)
- `lasth=`, `dt=`, `dt_riv=`, `riv_thresh=`, `ns_river=`, `ns_slope=`,
  `soildepth=`, `gammaa=`, `ksv=`, `faif=`, `beta=`, ... (unchanged
  scalar options from the superseded driver -- these were already
  GIS-native in spirit, just previously templated into a text file
  instead of read straight into the config struct)
- `-g` flag for the OpenCL/GPU backend (unchanged)
- `hydrograph_table=` (DB table name, unchanged in spirit)
- New: an output STRDS name option once periodic-grid output exists
  (§3's last row)

Multi-land-use support (parameter-per-category) still needs the
category-to-parameters mapping mechanism the earlier item 3 (LULC) asked
for -- design that as a `G_OPT_F_INPUT` params file or DB table keyed by
category, following `r.hydro.hbv`'s `table_input.c` precedent (not yet
looked at in this pass -- do so before implementing).

## 5. Validation discipline (same as every prior pass in this project)

1. Get static-input-only reading working and unit-tested first --
   mirror `RRI.opencl`'s own milestone order (data structures + GIS I/O
   before any physics wiring), *not* a rewrite-everything-at-once leap.
2. Build a tiny synthetic GRASS region (mirroring the existing pytest
   suite's 10x10 domain, and/or `RRI.opencl`'s own synthetic fixtures)
   and confirm the native-GRASS-populated `rri_grid`/cellsets produce
   *identical* index-setting/geometry results to the ASCII path on the
   same domain, before trusting the new I/O layer at all.
3. Re-run the full physics validation (adaptive RK45 river+slope
   routing) on that domain once wired, and ultimately re-validate against
   the Fortran reference / `RRI.opencl`'s existing 0.073%/0.40% bar on
   `solo30s` or `dem_jamuna` (see §6) once the new I/O layer can express
   that domain's full input set.
4. Confirm OpenCL dispatch is genuinely unaffected (it should be, since
   it only touches the same in-memory cellset structs) by running the
   existing OpenMP-vs-OpenCL cross-backend check against data populated
   through the *new* I/O layer, not just the old one.

## 6. Real reference data now available (from a separate coordinator note)

`yann@10.42.0.89`, GISDBASE `~/grassdata/jamuna_corridor/PERMANENT`:
`dem_jamuna` (2.86M cells, practical for iteration) and
`dem_brahmaputra_basin` (270M cells, whole-basin -- do not use for
iterative testing, ~76 minutes per `r.watershed`-class computation even
on GPU) plus already-computed `dem_brahmaputra_basin_drainage_direction`/
`_flow_accumulation` from `r.watershed.opencl`, `jamuna_forcing_{precipitation,temperature,potential_evaporation}_2020070{1..5}`
(5 days, real daily forcing rasters -- confirm/register as an STRDS
before use), and `hbv_basins`/`hbv_outlets_partial` for a realistic
outlet location. Use `dem_jamuna` + the forcing rasters as the real-world
validation domain once the native pipeline can run end to end; do not
attempt the whole-basin DEM in an iterative test loop.

`r.watershed.opencl` (`$HOME/dev/r.watershed.opencl`) is a separate,
existing, validated addon -- integrate with it as an addon-to-addon
dependency (its `direction=`/`accumulation=`-equivalent outputs consumed
as this module's own `direction=`/`accumulation=` inputs), not vendored.

## 7. What happens to the existing code in this directory

As of this plan being written, `r.hydro.rri.py` (subprocess-based
driver), `engine/` (vendored `RRI.opencl` copy + the CMake-in-Makefile
build hook), and the associated `tests/`/docs describe the
now-superseded ASCII/subprocess architecture. They are **left in place,
not deleted**, because:

- `engine/`'s physics-core files (§1's "reusable unchanged" rows) are
  exactly the source this rewrite reuses -- deleting them would just
  mean re-copying them from `RRI.opencl` again.
- `r.hydro.rri.py`'s reverse-engineered format/behavior knowledge (the
  `r.watershed` direction-convention reclass table, the outlet
  auto-detection heuristic, the rain-unit-conversion logic, the config
  field defaults) is directly reusable *design* even though the
  mechanism (write-ASCII-then-shell-out) is being replaced -- it is
  useful prior art for `io_grass.c`'s design, not dead weight.
- Nothing in this plan has been implemented yet against these files, so
  there is nothing working to break by leaving them as reference.

**This state should not persist past the next implementation pass**:
either the native module supersedes and this directory's root-level
Python driver + `engine/`'s Makefile hook are removed once the C module
does the same job, or (if a hybrid is ever deliberately chosen) that
decision gets documented explicitly rather than left as accidental
leftover clutter. Flagging this explicitly so it isn't mistaken for the
current, intended architecture by a future reader.
