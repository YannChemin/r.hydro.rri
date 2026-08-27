# r.hydro.rri

## DESCRIPTION

*r.hydro.rri* prepares model inputs for RRI (Rainfall-Runoff-Inundation),
a fully-distributed rainfall-runoff/flood-inundation model, from GRASS
rasters and space-time raster datasets (STRDS), and optionally runs the
model and imports its outlet hydrograph back into GRASS.

RRI itself is not a GRASS module -- it is a separate C engine
(`RRI.opencl`, at `$HOME/dev/RRI.opencl`, built from `RRI_1.4.2.7_Linux`)
with its own text-file/ESRI-ASCII-grid input format
(`RRI_Input.txt`/`topo/*.txt`/`rain/rain.dat`). *r.hydro.rri* is a GIS
data-preparation bridge: it exports **elevation**, derives flow
**direction**/**accumulation** (via *r.watershed*) when not supplied, and
converts a precipitation (and optionally potential evapotranspiration)
**STRDS** -- for example, `t.in.era5`'s `<prefix>_precipitation` -- into
RRI's own per-timestep grid format. With the **-r** flag it also invokes
the compiled `rri_cpu` binary and imports the resulting outlet hydrograph
into a GRASS DB table.

### Satellite/reanalysis forcing

`rain_strds=` is expected to come from a reanalysis or satellite-derived
precipitation product already imported as a GRASS STRDS -- `t.in.era5`'s
`precipitation` variable is the primary target (its daily mm/d output
matches `rain_units=mm_per_day`, the default). Any other STRDS with the
same interval-registered (start, end) structure works too, including one
built from *r.in.sentinel*/*r.in.modis*/*r.in.landsat* derived products or
registered manually with *t.register*. `pet_strds=` follows the same
pattern for potential evapotranspiration, e.g. `t.in.era5`'s
`potential_evaporation` output, or a series of *i.evapo.pm*/*i.evapo.pt*/
*i.evapo.mh* outputs registered into a STRDS.

Every map in a forcing STRDS must be registered as an **interval** (an
explicit start AND end time), not an instantaneous point -- RRI's rain/PET
file format is a step function of elapsed time and needs both. A STRDS
built by *t.in.era5* already satisfies this.

### RRI-side simplifications

* Single land-use class only (`num_of_landuse=1`) -- the `landuse=`
  option, if given, is exported for RRI's own internal use, but this
  module's `ns_slope=`/`soildepth=`/etc. options apply the *same* scalar
  value to the whole domain rather than one value per class. See "Known
  gaps" below.
* River channel geometry (width/depth/height) is not exported; RRI's own
  internal power-law estimate from flow accumulation is used instead
  (`rivfile_switch=0` in the generated `RRI_Input.txt`).
* The model's own D8 flow-direction convention differs from
  *r.watershed*'s `drainage` output -- see NOTES.

## NOTES

### Flow direction convention

*r.watershed*'s `drainage` output numbers 8 directions counter-clockwise
from 1=north-east (`r.watershed.md`); RRI's own `dir` grid instead uses an
8-bit D8 bitmask (east=1, doubling clockwise: E=1, SE=2, S=4, SW=8, W=16,
NW=32, N=64, NE=128 -- documented in prose only in `RRI_Break.f90`'s
comment block). *r.hydro.rri* reclasses one into the other. This mapping
has been smoke-tested (see `tests/`) but not cross-validated against a
real watershed with known flow paths -- treat it as a reasonable,
documented judgment call, not a verified-against-ground-truth conversion.

A cell whose flow leaves the computational region gets a *negative*
`drainage` value from *r.watershed*; such cells become RRI outlet cells
(`dir=0`). *r.watershed*'s `accumulation` output separately uses a
negative sign to flag a per-cell reliability caveat (not a direction) --
this module always exports `abs(accumulation)`, since RRI only ever
compares accumulation's magnitude against `riv_thresh=`.

### Outlet/hydrograph station

*r.hydro.rri* auto-places a single hydrograph station, named `outlet`, at
the accumulation raster's maximum-magnitude cell. This is a reasonable
default for a single-outlet watershed but is wrong for a domain with more
than one drainage outlet (e.g. a delta) -- not handled by this module.

### Rain/PET file format

RRI's `rain.dat`/PET file is a sequence of timestep blocks: a header line
`t nx ny` (`t` = elapsed seconds since the forcing series' own first map's
start) followed by `ny` rows of `nx` whitespace-separated values in mm/h,
preceded by a `t=0` all-zero block (used only as the lower bound of RRI's
own internal time-interpolation search, never applied itself). Values are
converted from `rain_units=`/`pet_units=` to mm/h before writing. The
forcing grid's own georeference is written into `RRI_Input.txt`'s
`xllcorner_rain`/etc. fields -- it does not need to align with
`elevation=`'s grid; RRI does its own nearest-cell lookup at runtime.

## KNOWN GAPS

* Single land-use class only; multi-class per-cell parameterization is not
  implemented.
* River width/depth/height override rasters, dam operation, flow
  diversion, and boundary-condition files (all separate RRI features) are
  not exposed by this module.
* `final_storage=` (importing RRI's periodic per-cell output grids back as
  a GRASS raster) is not yet implemented -- RRI's `out/hs_*`/`out/hr_*`
  files are per-cell-index dumps in the model's own compressed indexing,
  not directly re-importable as a full grid; wiring this up needs a small
  reprojection step not built in this pass.
* Only the first hydrograph station's discharge column is imported into
  the output DB table, even if `location.txt` ever listed more than one
  (it currently never does -- see "Outlet/hydrograph station" above).

## EXAMPLES

Prepare RRI inputs from a DEM and an ERA5 precipitation STRDS, without
running the model:

```sh
t.in.era5 variables=precipitation start=2026-01-01 end=2026-01-10 \
  output_prefix=era5
r.hydro.rri elevation=dem rain_strds=era5_precipitation \
  project_dir=/data/rri_project lasth=240
```

Also run the model (OpenMP/CPU backend) and import the outlet hydrograph:

```sh
r.hydro.rri elevation=dem rain_strds=era5_precipitation \
  project_dir=/data/rri_project lasth=240 -r
```

With PET forcing and the GPU/OpenCL backend:

```sh
t.in.era5 variables=precipitation,potential_evaporation \
  start=2026-01-01 end=2026-01-10 output_prefix=era5
r.hydro.rri elevation=dem rain_strds=era5_precipitation \
  pet_strds=era5_potential_evaporation \
  project_dir=/data/rri_project lasth=240 -r -g
```

## SEE ALSO

*[i.evapo.mh](i.evapo.mh.md), [i.evapo.pm](i.evapo.pm.md),
[i.evapo.pt](i.evapo.pt.md), [r.hydro.hbv](r.hydro.hbv.md),
[r.hydro.hbv.basins](r.hydro.hbv.basins.md),
[r.hydro.hbv.forcing](r.hydro.hbv.forcing.md), [r.in.landsat](r.in.landsat.md),
[r.in.modis](r.in.modis.md), [r.in.sentinel](r.in.sentinel.md),
[r.watershed](r.watershed.md), [t.in.era5](t.in.era5.md),
[t.rast.list](t.rast.list.md)*

RRI model: T. Sayama et al., "Rainfall-Runoff-Inundation (RRI) Model"
(<https://github.com/YannChemin/RRI_1.4.2.7_Linux>). The C/OpenMP/OpenCL
engine this module drives: `$HOME/dev/RRI.opencl` (this user's own port,
see its README for physics/validation details).

## AUTHORS

Yann Chemin
