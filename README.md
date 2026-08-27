# r.hydro.rri

A GRASS GIS addon that prepares inputs for the RRI (Rainfall-Runoff-
Inundation) distributed hydrology model from GRASS rasters and space-time
raster datasets (STRDS) -- with an emphasis on **satellite/reanalysis
raster forcing** (e.g. `t.in.era5` precipitation and potential
evapotranspiration) -- and, optionally, runs the model and imports its
outlet hydrograph back into GRASS.

See [`r.hydro.rri.md`](r.hydro.rri.md) for the full module documentation
(description, notes on format/convention conversions, known gaps,
examples). This file covers install/dev/test setup.

## What this bridges

RRI is not a GRASS module. The actual hydrology engine lives at
`$HOME/dev/RRI.opencl` -- a C11/OpenMP/OpenCL port of the original
Fortran RRI model (`$HOME/dev/RRI_1.4.2.7_Linux`), validated against the
compiled Fortran reference on a real watershed. That engine reads its own
plain-text config (`RRI_Input.txt`) and ESRI-ASCII-grid inputs
(`topo/*.txt`, `rain/rain.dat`, ...), with no knowledge of GRASS at all.

`r.hydro.rri` is the GIS-side data-preparation layer: it turns GRASS
rasters (DEM, land use) and STRDS (rainfall, PET) into that input format,
generates the config, and -- with `-r` -- shells out to the compiled
`rri_cpu` binary and imports its outlet-discharge time series back into
GRASS as a DB table.

```
GRASS raster (DEM)  ─┐
GRASS raster (land use) ─┤
STRDS (t.in.era5 precipitation) ─┼─► r.hydro.rri ─► RRI_Input.txt + topo/*.txt
STRDS (t.in.era5 PET, optional) ─┘        + rain/rain.dat [+ evp/pet.dat]
                                              │
                                              ▼ (-r)
                                        rri_cpu / rri_cl
                                              │
                                              ▼
                                  hydro.txt ─► GRASS DB table (hydrograph)
```

## Install / development layout

Standalone at `$HOME/dev/r.hydro.rri` (this directory); symlink into
`$HOME/dev/grass-addons/src/raster/r.hydro.rri` to build via the addon
tree, following this user's established convention for `r.hydro.hbv`/
`r.hydro.hbv.basins`/`r.hydro.hbv.forcing` (all in the same
`grass-addons/src/raster/` directory).

```sh
ln -s $HOME/dev/r.hydro.rri $HOME/dev/grass-addons/src/raster/r.hydro.rri
```

Build/install as a Python script addon (`Makefile` uses `Script.make`,
matching `r.hydro.hbv.forcing`'s pattern -- no compiled C in this module):

```sh
make MODULE_TOPDIR=$HOME/dev/grass
```

Running `RRI.opencl`'s `rri_cpu` binary requires it to already be built
separately:

```sh
cd $HOME/dev/RRI.opencl && cmake -B build && cmake --build build
```

`r.hydro.rri` defaults to `~/dev/RRI.opencl/build/rri_cpu`; override with
`rri_bin=` if built elsewhere.

## Running the tests

```sh
export PATH="/usr/local/bin:$PATH"   # or wherever `grass` actually is
export PYTHONPATH="$(grass --config python_path):${PYTHONPATH}"
export LD_LIBRARY_PATH="$(grass --config path)/lib:${LD_LIBRARY_PATH}"
cd $HOME/dev/r.hydro.rri
python3 -m pytest tests/ -v
```

Four tests, all exercising a small (10x10) synthetic domain built inside
each test (no external dataset dependency):

* `test_writes_rri_input_txt` -- config and static-grid files exist and
  are well-formed.
* `test_rain_dat_format_and_units` -- the rain forcing file's block
  structure, elapsed-time headers, and mm/day-to-mm/h conversion are
  correct.
* `test_direction_reclass_has_no_invalid_codes` -- every cell in the
  RRI-convention direction grid is a valid D8 bitmask code or an outlet.
* `test_full_run_against_compiled_rri_binary` -- **the real validation**:
  runs the actual compiled `rri_cpu` against everything this module
  wrote, and checks it produces a non-empty outlet hydrograph, imported
  successfully into a GRASS DB table. Skips itself if `RRI.opencl` hasn't
  been built yet (rather than failing), since that's a separate,
  optional dependency for this test only.

All four passed during development, including the real end-to-end run
against `RRI.opencl`'s compiled binary -- this is genuine evidence the
forcing/config format this module writes is accepted by RRI, not just
self-consistent.

## Known gaps and judgment calls

See `r.hydro.rri.md`'s "KNOWN GAPS" and "NOTES" sections for the full,
current list -- summarized here:

* **Single land-use class only.** Multi-class per-cell parameterization
  (RRI supports it; this module doesn't expose it yet) is the most
  likely next real limitation someone hits on a real watershed.
* **Flow-direction convention conversion** (`r.watershed`'s
  counter-clockwise-from-NE numbering -> RRI's D8 bitmask) is a manual
  cross-reference between two format specs, smoke-tested but not
  cross-validated against a real watershed with independently-known flow
  paths. Verify it on your own data before trusting it blindly on a
  domain where a wrong direction code would silently misroute flow.
* **Single-outlet assumption.** The hydrograph station is
  auto-placed at the accumulation raster's single maximum cell -- wrong
  for a multi-outlet domain (deltas, etc.).
* **`final_storage=` is not implemented.** RRI's periodic per-cell output
  grids are dumped in the model's own compressed sparse indexing, not a
  simple full-grid file re-importable as-is; this needs its own small
  reprojection step, not built in this pass. Only the outlet hydrograph
  (a DB table) is currently imported back into GRASS.
* **River geometry, dams, diversions, and boundary conditions** (all
  real RRI features) are not exposed by this module -- it relies on
  RRI's own internal power-law river-geometry estimate from flow
  accumulation instead of letting the user supply measured channel
  geometry.

None of these are silent — every one fails loudly or is documented,
following this user's standing GRASS convention of failing loudly rather
than silently guessing.
