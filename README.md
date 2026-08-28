# r.hydro.rri

A GRASS GIS addon that prepares inputs for the RRI (Rainfall-Runoff-
Inundation) distributed hydrology model from GRASS rasters and space-time
raster datasets (STRDS) -- with an emphasis on **satellite/reanalysis
raster forcing** (e.g. `t.in.era5` precipitation and potential
evapotranspiration) -- and, optionally, runs the model and imports its
results back into GRASS as native rasters/STRDS/DB tables.

## Two architectures currently coexist in this repo -- read this first

This repo is mid-transition. **`main.c` (built with plain `gcc`, or via
the Makefile once it's switched over -- see "Status" below) is the real,
current, actively-developed architecture**: a compiled GRASS C module
with zero ASCII intermediate files anywhere in the pipeline -- GRASS
rasters and STRDS in, GRASS rasters/STRDS/DB tables out, the RRI physics
engine linked in directly (not shelled out to as a separate binary).

`r.hydro.rri.py` (a Python script) plus `engine/` (a vendored, frozen
copy of `RRI.opencl`) are the **superseded, first-draft architecture**:
write ASCII `RRI_Input.txt`/grid files, shell out to a separate compiled
`rri_cpu` binary, import only the hydrograph back via `db.in.ogr`. Kept
around only because it is still genuinely useful as a second, independent
implementation to cross-check the native path's numbers against (see
"Validation" below) -- **not** because it's a supported way to use this
module going forward. See `NATIVE_GRASS_PLAN.md` for the full rationale,
increment-by-increment history, and the plan for retiring it once the
native path's coverage is confirmed equivalent.

**If you're just trying to use this module: use the compiled `main.c`
module (see "Quick start" below), not `r.hydro.rri.py`.**

## Status (NATIVE_GRASS_PLAN.md "Progress", summarized)

| Increment | What | Status |
|---|---|---|
| 1 | Static input (elevation/drainage/accumulation) + index setting | Done, validated bit-for-bit against the old ASCII path |
| 2 | Single-raster (`rain=`) forcing read/index | Done, validated |
| 3 | STRDS (`rain_strds=`) forcing time-series resolution | Done, validated |
| 4 | Adaptive RK45 time loop (the physics core) | Done, validated to 6+ sig figs against the ASCII engine, 24h stability run |
| 5 | GRASS-native output (hydrograph DB table, `hs_output=` STRDS) | Done, validated |
| 6 | `r.watershed.opencl` auto-invocation when `drainage=`/`accumulation=` omitted | Done, validated |
| 7 | Pixel-based LULC (`landuse=`, per-class parameters) | Done (static raster only), validated |
| 8 | `rain_units=` (mm/day default, matching `t.in.era5`) | Done, validated |
| -- | `landuse_strds=` (time-varying land use) | Not implemented -- see "Known gaps" |
| -- | Retire `r.hydro.rri.py`/`engine/` | Not done -- kept as the cross-check reference for now |
| -- | Groundwater, dams, diversions, boundary conditions, custom cross-sections | Not implemented (matches the vendored engine's own scope) |

## Quick start

```sh
# Build (ad hoc, until the Makefile switches to Module.make -- see below)
export PATH="/usr/local/bin:$PATH"    # or wherever `grass` actually is
gcc -std=c11 -O2 \
    -I "$(grass --config path)/include" -I rri_include \
    main.c rri_setup.c rri_geo.c rri_riv.c rri_slope.c rri_rivslo.c rri_infilt.c rri_rk.c \
    -L "$(grass --config path)/lib" -lgrass_gis.8.6 -lgrass_raster.8.6 -lm \
    -Wl,-rpath,"$(grass --config path)/lib" \
    -o r_hydro_rri_native

# Run (inside a GRASS session, region already set to your DEM):
./r_hydro_rri_native elevation=dem rain_strds=era5_precipitation \
    lasth=240 -r hydrograph_table=my_hydrograph hs_output=my_hs_strds
```

`drainage=`/`accumulation=` are optional -- omit them and the module runs
`r.watershed.opencl` itself (see "Flow direction / accumulation" below).

### Driving it from `t.in.era5` (the primary intended forcing source)

```sh
t.in.era5 variables=precipitation start=2026-01-01 end=2026-01-10 \
    output_prefix=era5

./r_hydro_rri_native elevation=dem rain_strds=era5_precipitation \
    lasth=240 -r hydrograph_table=my_hydrograph
```

**No manual unit conversion needed.** `t.in.era5`'s `precipitation` (and
`potential_evaporation`) outputs are *always* daily sums in mm -- checked
directly against `t.in.era5.py`'s source: both are registered with the
description `"... daily sum (mm/d)"`, and `t.in.era5` has no hourly-output
mode, so this is not a maybe-the-user-configured-it-differently case.
`rain_units=` therefore **defaults to `mm_per_day`**, exactly matching
`t.in.era5`'s one and only convention -- point `rain_strds=` straight at
a `t.in.era5`-produced STRDS and it just works. This *does* mean the
model runs on a uniform hourly rate derived from each day's total (a
real, documented simplification -- a daily sum cannot recover sub-daily
rainfall intensity variation, only its correct daily mean), not
instantaneous ERA5 sub-daily values (`t.in.era5` doesn't offer those).
If your own STRDS genuinely is already hourly-rate data, pass
`rain_units=mm_per_hour` explicitly. Getting this backwards silently
produces a 24x error in either direction -- this exact mixup happened
once already in this module's own test harness during development (see
`NATIVE_GRASS_PLAN.md`), which is why the default was chosen this
carefully rather than left as an arbitrary guess.

## What this bridges

RRI is not a GRASS-native model. Its physics core -- adaptive Cash-Karp
RK45 river/hillslope routing, river<->slope exchange, Green-Ampt
infiltration -- is a C library vendored from `RRI.opencl` (`engine/`,
frozen copy, see `engine/VENDORED.md`; original at `$HOME/dev/RRI.opencl`,
itself validated against the compiled Fortran RRI reference on a real
watershed). That physics-core code (`rri_riv.c`, `rri_slope.c`,
`rri_rivslo.c`, `rri_infilt.c`, `rri_rk.c`, `include/rri/kernels.h`,
`include/rri/rri.h`) has **zero file-I/O coupling** -- confirmed by
inspection, not assumed -- so `main.c` links it directly and drives it
from GRASS rasters/STRDS, with GRASS raster/STRDS/DB-table output, and no
ASCII files anywhere in between.

```
GRASS raster (DEM) ────────────────┐
GRASS raster (drainage, optional) ─┤  (auto-derived via r.watershed.opencl if omitted)
GRASS raster (accumulation, opt.) ─┤
GRASS raster (landuse, optional) ──┤
STRDS (t.in.era5 precipitation) ───┼──► main.c ──► [linked-in RRI physics core, RK45 loop]
GRASS raster (rain=, alternative) ─┘         │
                                              ▼ (-r)
                          GRASS raster STRDS (hs_output=)  +  DB table (hydrograph_table=)
```

## Flow direction / accumulation

`drainage=`/`accumulation=` follow `r.watershed`'s convention (drainage:
8 directions counter-clockwise from north-east=1, negative at domain
edges; accumulation: magnitude used, sign is r.watershed's own per-cell
reliability flag, not part of RRI's model). Reclassed internally to
RRI's own D8 bitmask (`drainage_to_rri_dir` in `main.c`) -- smoke-tested
but not independently cross-validated against a real watershed with
known flow paths, see "Known gaps".

Omit both to auto-derive via **`r.watershed.opencl`** (this user's own
GPU-accelerated addon, `$HOME/dev/r.watershed.opencl` -- a drop-in
replacement for `r.watershed`, whose single-threaded core is
impractically slow at basin scale). `watershed_threshold=` controls its
own `threshold=` (default 1 -- this module never reads its `stream=`/
`basin=` outputs, so the finest-grained default has no effect on
`drainage=`/`accumulation=` themselves). Auto-derived temp rasters are
cleaned up after the run. Pass `drainage=`/`accumulation=` explicitly
instead when you already have them (e.g. reused across repeated
experiments on the same DEM) to skip recomputing -- on a basin-scale
DEM this is the difference between minutes and well over an hour,
mirroring the `drainage_input=`/`accumulation_input=` passthrough
pattern already established in this user's `r.hydro.hbv.basins` addon.

## Land use / land cover

`landuse=` is a classified GRASS raster. **The raster's cell VALUES
(integer categories, read via `Rast_get_c_row`) are what the physics
reads -- category LABEL text (`r.category`/`r.support`) is metadata this
module never looks at.** Categories must be `1..num_of_landuse`
contiguous; `num_of_landuse` is taken as the raster's own maximum
category value. Each category is a distinct combination of hydraulic
parameters (`ns_slope=`, `soildepth=`, `gammaa=`, `ksv=`, `faif=`, `ka=`,
`gammam=`, `beta=`) -- not a semantic land-cover label GRASS has any
opinion about; category 3 being "urban" or "forest" is entirely up to
how you built the raster and what parameter values you supply for it.

Pass each of those 8 options as `num_of_landuse` comma-separated values,
in category order (`ns_slope=0.4,0.1,0.02` for 3 classes), or a single
value to apply uniformly to every class. Passing the wrong count is a
hard error (`ns_slope: got 2 value(s), need exactly 3 ...`), not a
silent reuse of the last value or a default fill-in.

Omit `landuse=` entirely for a single uniform class over the whole
domain (`num_of_landuse=1`).

**Worked example**, classifying a real source into a 1..N raster:

```sh
# ESA WorldCover / MODIS MCD12Q1 / any classified LULC product, imported
# via whatever standard GRASS import path fits your source (r.in.gdal
# for a pre-classified GeoTIFF, r.in.modis for MCD12Q1, etc.) -- ends up
# as some_lulc_raw with the SOURCE product's own category scheme.
r.reclass input=some_lulc_raw output=landuse_3class rules=- << 'EOF'
10 20 30 = 1 forest
40 = 2 cropland
50 = 3 urban
* = 1
EOF

./r_hydro_rri_native elevation=dem rain_strds=era5_precipitation \
    landuse=landuse_3class \
    ns_slope=0.4,0.15,0.02 soildepth=1.2,0.8,0.1 gammaa=0.5,0.45,0.3 \
    lasth=240 -r hydrograph_table=my_hydrograph
```

**Time-varying land use is not implemented** (`landuse_strds=`, mirroring
`rain_strds=`'s pattern -- resolve a registered land-cover-change STRDS's
maps by valid time and re-feed the read/reclass/index path per
timestep). If your run needs land use to change mid-simulation (a
multi-year run crossing a land-cover-change epoch, distinct wet/dry-season
classifications), that's not supported yet -- flagged as the clear next
step in `NATIVE_GRASS_PLAN.md`, not silently ignored.

## Output

* `hydrograph_table=<name>`: a DB table (`time_s`, `discharge_cms`) of
  discharge summed over the domain's outlet river cell(s)
  (`domain==2 && riv==1`), one row per outer timestep.
* `hs_output=<name>` (+ `hs_interval=`, default every timestep): a GRASS
  STRDS of hillslope water depth [m], one map every `hs_interval` outer
  timesteps, registered with `t.register`.

Both are written via a one-shot shelled-out `db.execute`/`t.register`
call (not a subprocess per row/map -- GRASS's temporal framework and
DBMI have no stable C API from a compiled `Module.make` addon, so a
single batched call was chosen over linking either directly; see
`main.c`'s `write_hydrograph_table`/`register_hs_strds` for the exact
approach). Timestamps for `hs_output=` are `time(NULL)` (module start
time) plus each timestep's elapsed seconds -- a reasonable placeholder,
not necessarily meaningful in real-world calendar time; a real
deployment threading the forcing STRDS's own actual start date through
would be a natural improvement, not done this pass.

## Known gaps and judgment calls

* **Flow-direction convention conversion** is smoke-tested but not
  independently cross-validated against a real watershed with
  known flow paths. Verify on your own data before trusting it blindly.
* **Single-outlet assumption.** Hydrograph discharge is summed over
  every `domain==2 && riv==1` cell, which in practice is usually one
  cell for a single-outlet watershed -- wrong (under-counts) for a
  genuinely multi-outlet domain (deltas, etc.), though the summation
  itself would still be correct if there happen to be several.
* **`landuse_strds=` (time-varying land use) is not implemented** -- see
  "Land use / land cover" above.
* **Groundwater, dams, diversions, boundary conditions, custom
  cross-sections** are not implemented -- matches the vendored engine's
  own scope (`engine/VENDORED.md`), not a native-rewrite regression.
* **River geometry** always comes from RRI's internal power-law estimate
  (`width_param_*`/`depth_param_*`) from flow accumulation -- supplying
  measured channel width/depth/height directly is not exposed.
* **`r.hydro.rri.py`/`engine/` are not retired yet** despite being fully
  superseded in scope by the native `main.c` path -- kept only as the
  cross-check reference; see `NATIVE_GRASS_PLAN.md` section 7 for why
  this is flagged as a real "should not persist" state, not the intended
  end architecture.

None of these are silent -- every one fails loudly or is documented,
following this user's standing GRASS convention of failing loudly rather
than silently guessing.

## Validation

See `NATIVE_GRASS_PLAN.md` "Progress" for the full increment-by-increment
validation writeup (each one checked against either the old ASCII
engine's own output on an identical synthetic domain, or a physically
checkable diagnostic, before the next increment began). Run the full
suite:

```sh
export PATH="/usr/local/bin:$PATH"
export PYTHONPATH="$(grass --config python_path):${PYTHONPATH}"
export LD_LIBRARY_PATH="$(grass --config path)/lib:${LD_LIBRARY_PATH}"
cd $HOME/dev/r.hydro.rri
python3 -m pytest tests/ -v
```

15 tests as of this writing, all passing, covering: index-setting
bit-for-bit agreement with the ASCII path; forcing read/index/STRDS
resolution correctness; the RK45 loop's mass balance and 6+ significant
figure agreement with the ASCII engine; a 24h stability run; GRASS-native
output correctness (including a real cross-checked finding -- the tiny
synthetic test domain's outlet reports exactly zero discharge in BOTH
engines, a genuine domain artifact, not a bug); `r.watershed.opencl`
auto-invocation and cleanup; `rain_units=` conversion correctness; and
LULC per-class parameterization actually changing routing physics
differently per class (not just accepted syntactically).

## Install / development layout (old architecture -- see status above)

The rest of this section describes `r.hydro.rri.py`'s installation as a
GRASS script addon, kept for reference while it's still present:

```sh
ln -s $HOME/dev/r.hydro.rri $HOME/dev/grass-addons/src/raster/r.hydro.rri
make MODULE_TOPDIR=$HOME/dev/grass
```

Running the vendored `engine/build/rri_cpu` binary (used by
`r.hydro.rri.py` and by the native path's own cross-check tests) requires
building it once:

```sh
cd $HOME/dev/r.hydro.rri && cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
cmake --build engine/build -j
```
