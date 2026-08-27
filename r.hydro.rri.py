#!/usr/bin/env python3
############################################################################
#
# MODULE:       r.hydro.rri
# AUTHOR:       Yann Chemin
# PURPOSE:      Prepares RRI (Rainfall-Runoff-Inundation) model inputs --
#               topography, land use, and satellite/reanalysis raster
#               forcing (precipitation and potential evapotranspiration,
#               e.g. from t.in.era5) -- from GRASS rasters and space-time
#               raster datasets, following the RRI_Input.txt/ESRI-ASCII-
#               grid format used by the RRI.opencl and RRIpy engines.
#               Optionally runs the model (rri_cpu/rri_cl) and imports its
#               outlet hydrograph and final storage back into GRASS.
# COPYRIGHT:    (C) 2026 by Yann Chemin
#               Released into the public domain -- see LICENSE (Unlicense).
#
############################################################################

# %module
# % description: Prepares RRI model inputs (topography, land use, satellite/reanalysis rainfall and PET forcing) from GRASS rasters and STRDS, and optionally runs the model and imports its outputs.
# % keyword: raster
# % keyword: temporal
# % keyword: hydrology
# % keyword: RRI
# %end
# %option G_OPT_R_ELEV
# % key: elevation
# % description: Input DEM; the current region (g.region) defines the model domain and must match it
# %end
# %option G_OPT_R_INPUT
# % key: direction
# % required: no
# % description: D8 flow direction raster, r.watershed 'drainage' convention (1-8 counter-clockwise from north-east, negative at region edges). Derived from elevation via 'r.watershed -s' if omitted.
# %end
# %option G_OPT_R_INPUT
# % key: accumulation
# % required: no
# % description: Flow accumulation raster, r.watershed 'accumulation' convention (cell counts). Derived from elevation via 'r.watershed -s' if omitted.
# %end
# %option G_OPT_R_INPUT
# % key: landuse
# % required: no
# % description: Land-use/land-cover raster, integer categories 1..num_of_landuse. Omitted -> a single uniform land class covers the whole domain (num_of_landuse=1); RRI's per-landuse hydraulic parameters (ns_slope=, soildepth=, etc. below) are then used as a single scalar each. Multi-landuse parameterization is not yet implemented -- see README "Known gaps".
# %end
# %option G_OPT_STRDS_INPUT
# % key: rain_strds
# % description: Precipitation space-time raster dataset, e.g. t.in.era5's <output_prefix>_precipitation
# %end
# %option
# % key: rain_units
# % type: string
# % options: mm_per_day,mm_per_hour
# % answer: mm_per_day
# % description: Units of rain_strds's raster cell values. t.in.era5's precipitation output is mm_per_day (its daily total); use mm_per_hour for a series that is already an hourly (or sub-daily) rate.
# %end
# %option G_OPT_STRDS_INPUT
# % key: pet_strds
# % required: no
# % description: Potential evapotranspiration space-time raster dataset, e.g. t.in.era5's <output_prefix>_potential_evaporation, or an i.evapo.* output series registered into a STRDS with t.register. Omit to run RRI without PET (evp_switch=0).
# %end
# %option
# % key: pet_units
# % type: string
# % options: mm_per_day,mm_per_hour
# % answer: mm_per_day
# % description: Units of pet_strds's raster cell values
# %end
# %option G_OPT_M_DIR
# % key: project_dir
# % description: Directory to write the RRI project into (RRI_Input.txt, topo/, rain/, evp/). Created if it does not exist.
# %end
# %option
# % key: lasth
# % type: integer
# % required: yes
# % description: Simulation length (hours)
# %end
# %option
# % key: dt
# % type: integer
# % answer: 600
# % description: Slope/outer timestep (seconds)
# %end
# %option
# % key: dt_riv
# % type: integer
# % answer: 60
# % description: Initial river adaptive-RK45 sub-timestep (seconds)
# %end
# %option
# % key: riv_thresh
# % type: integer
# % answer: 100
# % description: Minimum flow accumulation (cell count) for a cell to be treated as a river channel rather than hillslope
# %end
# %option
# % key: ns_river
# % type: double
# % answer: 0.03
# % description: River Manning's roughness coefficient
# %end
# %option
# % key: ns_slope
# % type: double
# % answer: 0.4
# % description: Hillslope Manning's roughness coefficient
# %end
# %option
# % key: soildepth
# % type: double
# % answer: 1.0
# % description: Soil depth (m), for Green-Ampt infiltration and groundwater storage capacity
# %end
# %option
# % key: gammaa
# % type: double
# % answer: 0.475
# % description: Soil porosity (effective saturated water content), 0-1
# %end
# %option
# % key: faif
# % type: double
# % answer: 0.316
# % description: Green-Ampt wetting front suction head Sf (m)
# %end
# %option
# % key: ksv
# % type: double
# % answer: 0.0
# % description: Green-Ampt vertical saturated hydraulic conductivity (m/s). 0 disables Green-Ampt infiltration for this land class (mutually exclusive with ka; RRI errors if both are nonzero).
# %end
# %option
# % key: beta
# % type: double
# % answer: 8.0
# % description: Hillslope kinematic/diffusive-wave exponent (unsaturated subsurface flow)
# %end
# %flag
# % key: r
# % description: Also run the model after writing its inputs, and import the outlet hydrograph and final storage back into GRASS
# %end
# %flag
# % key: g
# % description: With -r, run the OpenCL/GPU backend (rri_cpu --gpu) instead of plain OpenMP/CPU. See RRI.opencl's README "Choosing a backend" -- only worth it on domains much larger than this module has been tested with.
# %end
# %option
# % key: rri_bin
# % type: string
# % required: no
# % description: Path to the compiled rri_cpu binary (default $HOME/dev/RRI.opencl/build/rri_cpu)
# %end
# %option
# % key: hydrograph_table
# % type: string
# % required: no
# % description: With -r, name for the output outlet-hydrograph DB table (time, discharge_cms). Default <project_dir basename>_hydrograph.
# %end
# %option G_OPT_R_OUTPUT
# % key: final_storage
# % required: no
# % description: With -r, name for an output raster of final per-cell water storage depth (m), reprojected back onto elevation's grid. Omit to skip importing it.
# %end

import atexit
import os
import subprocess
import sys

import numpy as np

import grass.script as gs
import grass.script.array as garray

TMP_RASTERS = []
TMP_REGIONS = []

DEFAULT_RRI_BIN = os.path.expanduser("~/dev/RRI.opencl/build/rri_cpu")

# r.watershed's 'drainage' output: 8 directions numbered counter-clockwise
# from 1 = north-east (see r.watershed.md), i.e. compass bearing
# 45 * value degrees measured counter-clockwise from east. RRI's own 'dir'
# grid instead uses an 8-bit D8 bitmask (east=1, doubling clockwise) --
# see RRI_Break.f90's comment block, the only place in the Fortran source
# that documents it in prose. This table is a manual cross-reference of
# the two conventions, not something read from either source directly --
# flagged here as a judgment call to verify (e.g. against a known real
# watershed's flow pattern) before trusting it on a domain where getting
# flow direction wrong would silently misroute water rather than crash.
_DRAINAGE_TO_RRI_DIR = {
    1: 128,  # NE
    2: 64,  # N
    3: 32,  # NW
    4: 16,  # W
    5: 8,  # SW
    6: 4,  # S
    7: 2,  # SE
    8: 1,  # E
}


def cleanup():
    for name in TMP_RASTERS:
        gs.run_command(
            "g.remove", flags="f", type="raster", name=name, quiet=True,
        )
    for region in TMP_REGIONS:
        gs.run_command(
            "g.remove", flags="f", type="region", name=region, quiet=True,
        )


def tmp_raster(prefix):
    name = gs.append_uuid(gs.append_node_pid(prefix))
    TMP_RASTERS.append(name)
    return name


def derive_flow_direction_and_accumulation(elevation, direction, accumulation):
    """Returns (direction_map, accumulation_map), deriving whichever of
    the two was not supplied via a single 'r.watershed -s' pass (forced
    single-flow-direction/D8, since RRI's routing is strictly D8 -- MFD's
    fractional multi-direction accumulation has no RRI equivalent)."""
    if direction and accumulation:
        return direction, accumulation
    # r.watershed computes drainage and accumulation together in one
    # pass -- if only one of the two options was supplied, the other
    # still has to be derived, but r.watershed's drainage=/accumulation=
    # are OUTPUT map names it will happily overwrite. Always write both
    # to fresh temporary names here, then use the user-supplied map
    # (unmodified) in place of whichever one they actually gave, rather
    # than risking r.watershed silently clobbering an existing raster
    # the user passed in as direction= or accumulation=.
    drainage_raw = tmp_raster("rri_drainage_raw")
    accum_raw = tmp_raster("rri_accum_raw")
    gs.run_command(
        "r.watershed",
        elevation=elevation,
        drainage=drainage_raw,
        accumulation=accum_raw,
        flags="s",
    )
    drainage_out = direction or drainage_raw
    if accumulation:
        # Trust a user-supplied accumulation raster's values as-is (it
        # is not r.watershed's own output, so the sign convention below
        # does not apply to it).
        return drainage_out, accumulation
    # r.watershed's accumulation sign flags a per-cell reliability
    # caveat (negative = "likely an underestimate", typically because
    # the cell's contributing area may extend past the mapped region's
    # edge -- see r.watershed.md's -a flag description), not a real
    # direction of flow the way drainage's sign does. RRI only ever
    # compares |accumulation| against riv_thresh (RRI_Sub.f90 declares
    # it a plain magnitude), so the sign must be dropped here -- on a
    # small domain (e.g. this module's own tests) most or all cells can
    # carry the negative flag simultaneously, since every flow path
    # exits through a nearby edge; keeping the sign would then treat
    # nearly the whole domain as a mass of "unreliable" river cells.
    accum_out = tmp_raster("rri_accum")
    gs.run_command(
        "r.mapcalc", expression=f"{accum_out} = abs({accum_raw})", overwrite=True
    )
    return drainage_out, accum_out


def reclass_direction_to_rri(drainage_map):
    """Builds an RRI-convention 'dir' raster (1/2/4/8/16/32/64/128 D8
    bitmask, 0 = outlet/boundary) from an r.watershed 'drainage' raster.

    r.watershed marks a cell whose flow leaves the computational region
    with a negative direction value (see r.watershed.md); RRI instead
    marks such a cell dir=0 (checked in RRI.f90's domain-setting loop,
    where dir==0 or dir==-1 -> domain=2, i.e. outlet). Treating every
    r.watershed edge/negative cell as an RRI outlet is a reasonable but
    unverified mapping between the two models' conventions -- it has not
    been cross-checked against a real watershed with known outlet
    locations, only smoke-tested on a synthetic domain (see tests/).
    """
    rri_dir = tmp_raster("rri_dir")
    cases = " ".join(
        f"if(abs({drainage_map})=={code},{rri},"
        for code, rri in _DRAINAGE_TO_RRI_DIR.items()
    )
    closing = ")" * len(_DRAINAGE_TO_RRI_DIR)
    expr = (
        f"{rri_dir} = if(isnull({drainage_map}) || {drainage_map} < 0, 0, "
        f"{cases}0{closing})"
    )
    gs.run_command("r.mapcalc", expression=expr, overwrite=True)
    return rri_dir


def write_ascii_grid(mapname, path, integer=False):
    """Exports mapname to path in ESRI ASCII grid format (ncols/nrows/
    xllcorner/yllcorner/cellsize/NODATA_value header + row-major values),
    which is exactly the format RRI's read_gis_real/read_gis_int expect
    (RRI_Sub.f90). r.out.ascii's *default* header (north:/south:/east:/
    west:/rows:/cols:) is GRASS's own format, not this one -- flags='l'
    (r.out.ascii's "LISFLOOD" mode) is what actually produces the
    ncols/nrows/xllcorner/yllcorner/cellsize/NODATA_value layout RRI
    needs (see r.out.ascii's formspecific.c:writeLISFLOODheader);
    confirmed by running r.out.ascii both ways and diffing the header,
    not assumed from the flag's LISFLOOD-branded name."""
    kwargs = {
        "input": mapname,
        "output": path,
        "null_value": "-9999",
        "flags": "l",
        "overwrite": True,
    }
    if integer:
        kwargs["flags"] += "i"
    gs.run_command("r.out.ascii", **kwargs)


def strds_maps_chronological(strds):
    """Returns [(map_name, end_time), ...] for strds, sorted by start
    time -- the raster list order t.rast.list itself returns is not
    documented as guaranteed-chronological, so this sorts explicitly."""
    rows = gs.read_command(
        "t.rast.list", input=strds, columns="name,start_time,end_time",
        separator="|",
    ).strip()
    entries = []
    for line in rows.splitlines()[1:] if rows else []:
        name, start, end = line.split("|")
        entries.append((name, start, end))
    entries.sort(key=lambda e: e[1])
    return entries


def write_forcing_series(strds, units, out_path, label):
    """Writes an RRI-format forcing file (rain_.dat/PET.txt) for strds:
    a t=0 zero block (RRI's t_rain(0)/t_evp(0) convention -- see RRI.f90's
    rainfall-lookup loop, which only ever uses this block as the *lower*
    bound of its bracket search, never as an applied value itself) followed
    by one block per map in chronological order, each timestamped at the
    elapsed seconds since the series' own first map's start (so day k's
    map lands at t=k*86400 for a daily series), holding that map's value
    converted to mm/h (RRI.f90 line ~555: 'qp = qp / 3600.d0 / 1000.d0'
    converts an mm/h file to m/s internally, so the file itself must be
    mm/h -- t.in.era5's own daily-total mm/day output needs /24 first).

    Returns a dict of the forcing grid's own georeference (xllcorner,
    yllcorner, cellsize_x, cellsize_y) for RRI_Input.txt -- read from
    whichever raster of the series r.out.ascii itself, since RRI does its
    own nearest-cell lookup against these coordinates at runtime and does
    not require this grid to align with the DEM's.
    """
    import datetime

    entries = strds_maps_chronological(strds)
    if not entries:
        gs.fatal(f"{label}: STRDS <{strds}> has no registered raster maps")
    if any(not end for _, _, end in entries):
        gs.fatal(
            f"{label}: STRDS <{strds}> has one or more maps registered "
            "as an instant (no end_time) rather than an interval. RRI's "
            "rain/PET file format is a step function of elapsed time, so "
            "every map needs a definite (start, end) interval -- a real "
            "t.in.era5-produced STRDS already has this; an instant-point "
            "STRDS needs re-registering with explicit end times first "
            "(t.register's 'name|start|end' file format)."
        )

    first_map = entries[0][0]
    region = None
    with gs.RegionManager(raster=first_map):
        region = gs.region()
    ncols, nrows = int(region["cols"]), int(region["rows"])
    cellsize_x, cellsize_y = region["ewres"], region["nsres"]
    xllcorner, yllcorner = region["w"], region["s"]

    def parse_time(t):
        return datetime.datetime.strptime(t.split(".")[0], "%Y-%m-%d %H:%M:%S")

    t0 = parse_time(entries[0][1])

    with open(out_path, "w") as f:
        f.write(f"{0:>15d}{ncols:>6d}{nrows:>6d}\n")
        for _ in range(nrows):
            f.write((" " * 8 + "0.000") * ncols + "\n")

        for name, start, end in entries:
            elapsed_s = int(round((parse_time(end) - t0).total_seconds()))
            with gs.RegionManager(raster=name):
                arr = garray.array(mapname=name)
            values = arr.astype(float)
            if units == "mm_per_day":
                values = values / 24.0
            f.write(f"{elapsed_s:>15d}{ncols:>6d}{nrows:>6d}\n")
            for row in values:
                f.write("".join(f"{v:13.3f}" for v in row) + "\n")

    gs.verbose(
        f"{label}: wrote {len(entries)} timestep(s) to {out_path} "
        f"(t=0 .. {elapsed_s}s, mm/h)"
    )
    return {
        "xllcorner": xllcorner,
        "yllcorner": yllcorner,
        "cellsize_x": cellsize_x,
        "cellsize_y": cellsize_y,
    }


def write_location_txt(accumulation_map, path):
    """Writes RRI's location.txt (outlet-hydrograph station list, format
    '<label> <i> <j>' with 1-based (row, col) indices -- RRI.f90's
    "hydro file" section, RRI_Read.f90 line ~278). One auto-detected
    station named 'outlet', placed at the accumulation raster's maximum
    cell -- a reasonable default for a single-outlet domain (matches how
    r.hydro.rri's own direction reclass treats the highest-accumulation
    edge cell as the domain's outlet), but wrong for a multi-outlet
    domain (e.g. a delta or a region with more than one edge drain
    point) -- not handled by this module; see README 'Known gaps'.
    """
    with gs.RegionManager(raster=accumulation_map):
        arr = garray.array(mapname=accumulation_map)
    row, col = (int(i) for i in np.unravel_index(np.argmax(arr), arr.shape))
    with open(path, "w") as f:
        f.write(f"outlet {row + 1} {col + 1}\n")


def write_rri_input_txt(path, cfg):
    """Writes RRI_Input.txt in RRI's exact field order (RRI_Read.f90,
    cross-checked against RRI.opencl's src/rri_io.c parser and the real
    example at RRI_1.4.2.7_Linux/solo30s/RRI_Input.txt). Single land-use
    class only (num_of_landuse=1) -- see module docstring/README for why.
    River geometry is left to RRI's own internal derivation from
    accumulation (rivfile_switch=0), not supplied here.
    """
    lines = [
        "RRI_Input_Format_Ver1_4_2",
        "",
        "./rain/rain.dat",
        "./topo/dem.txt",
        "./topo/acc.txt",
        "./topo/dir.txt",
        "",
        "0    # utm(1) or latlon(0)",
        "1    # 4-direction (0), 8-direction(1)",
        f"{cfg['lasth']}    # lasth(hour)",
        f"{cfg['dt']}    # dt(second)",
        f"{cfg['dt_riv']}    # dt_riv",
        "24    # outnum [-]",
        f"{cfg['rain_xllcorner']:.6f}   # xllcorner_rain",
        f"{cfg['rain_yllcorner']:.6f}    # yllcorner_rain",
        f"{cfg['rain_cellsize_x']:.10f} {cfg['rain_cellsize_y']:.10f}    # cellsize_rain",
        "",
        f"{cfg['ns_river']:.3f}     # ns_river",
        "1    # num_of_landuse",
        "1    # diffusion(1) or kinematic(0)",
        f"{cfg['ns_slope']:.3f}     # ns_slope",
        f"{cfg['soildepth']:.3f}     # soildepth",
        f"{cfg['gammaa']:.3f}     # gammaa",
        "",
        f"{cfg['ksv']:.3f}     # ksv",
        f"{cfg['faif']:.3f}     # faif",
        "",
        "0.000     # ka",
        "0.000     # gammam",
        f"{cfg['beta']:.3f}     # beta",
        "",
        "0.000     # kgv",
        "0.400     # gammag",
        "0.00050     # tg",
        "0.030     # fpg",
        "0.500     # init_cond_gw",
        "",
        f"{cfg['riv_thresh']}      # riv_thresh",
        "5.000      # width_param_c (2.5)",
        "0.350      # width_param_s (0.4)",
        "0.950      # depth_param_c (0.1)",
        "0.200      # depth_param_s (0.4)",
        "0.000      # height_param",
        "20       # height_limit_param",
        "",
        "0",
        "./riv/width.txt",
        "./riv/depth.txt",
        "./riv/height.txt",
        "",
        "0  0  0  0",
        "./init/hs_init_dummy.out",
        "./init/hr_init_dummy.out",
        "./init/hg_init_dummy.out",
        "./init/gamptff_init_dummy.out",
        "",
        "0  0",
        "./bound/hs_bound.txt",
        "./bound/hr_bound.txt",
        "",
        "0  0",
        "./bound/qs_bound.txt",
        "./bound/qr_bound.txt",
        "",
        "1" if cfg["landuse"] else "0",
        "./topo/landuse.txt",
        "",
        "0",
        "./dam.txt",
        "",
        "0",
        "./div.txt",
        "",
        ("1" if cfg["pet"] else "0"),
        "./evp/pet.dat",
        f"{cfg.get('pet_xllcorner', 0.0):.6f}      # xllcorner_evp",
        f"{cfg.get('pet_yllcorner', 0.0):.6f}      # yllcorner_evp",
        f"{cfg.get('pet_cellsize_x', 0.01):.10f}  {cfg.get('pet_cellsize_y', 0.01):.10f}     # cellsize",
        "",
        "0",
        "./riv/length.txt",
        "",
        "0",
        "./riv/sec_map.txt",
        "./riv/section/sec_",
        "",
        "1  1  0  1  0  0  0  0  0  1",
        "./out/hs_",
        "./out/hr_",
        "./out/hg_",
        "./out/qr_",
        "./out/qu_",
        "./out/qv_",
        "./out/gu_",
        "./out/gv_",
        "./out/gampt_ff_",
        "./out/storage.dat",
        "",
        "1",
        "./location.txt",
        "",
    ]
    with open(path, "w") as f:
        f.write("\n".join(lines))


def prepare_project(options):
    if float(options["ksv"]) > 0.0:
        gs.warning(
            "ksv > 0 with RRI's default ka=0.0 -- Green-Ampt infiltration "
            "is on. RRI itself fatals if both ksv and ka are nonzero for "
            "the same land class (see RRI_Input.py's own parameter-check "
            "comment); this module never sets ka nonzero, so no conflict "
            "is possible here."
        )

    project_dir = options["project_dir"]
    for sub in ("topo", "rain", "evp", "riv", "init", "bound", "out"):
        os.makedirs(os.path.join(project_dir, sub), exist_ok=True)

    elevation = options["elevation"]
    direction, accumulation = derive_flow_direction_and_accumulation(
        elevation, options["direction"], options["accumulation"]
    )
    rri_dir = reclass_direction_to_rri(direction)

    write_ascii_grid(elevation, os.path.join(project_dir, "topo", "dem.txt"))
    write_ascii_grid(
        accumulation, os.path.join(project_dir, "topo", "acc.txt"), integer=True
    )
    write_ascii_grid(
        rri_dir, os.path.join(project_dir, "topo", "dir.txt"), integer=True
    )
    if options["landuse"]:
        write_ascii_grid(
            options["landuse"],
            os.path.join(project_dir, "topo", "landuse.txt"),
            integer=True,
        )
    write_location_txt(accumulation, os.path.join(project_dir, "location.txt"))

    rain_geo = write_forcing_series(
        options["rain_strds"],
        options["rain_units"],
        os.path.join(project_dir, "rain", "rain.dat"),
        "rain_strds",
    )

    pet_geo = None
    if options["pet_strds"]:
        pet_geo = write_forcing_series(
            options["pet_strds"],
            options["pet_units"],
            os.path.join(project_dir, "evp", "pet.dat"),
            "pet_strds",
        )

    cfg = {
        "lasth": int(options["lasth"]),
        "dt": int(options["dt"]),
        "dt_riv": int(options["dt_riv"]),
        "riv_thresh": int(options["riv_thresh"]),
        "ns_river": float(options["ns_river"]),
        "ns_slope": float(options["ns_slope"]),
        "soildepth": float(options["soildepth"]),
        "gammaa": float(options["gammaa"]),
        "ksv": float(options["ksv"]),
        "faif": float(options["faif"]),
        "beta": float(options["beta"]),
        "landuse": bool(options["landuse"]),
        "pet": bool(options["pet_strds"]),
        "rain_xllcorner": rain_geo["xllcorner"],
        "rain_yllcorner": rain_geo["yllcorner"],
        "rain_cellsize_x": rain_geo["cellsize_x"],
        "rain_cellsize_y": rain_geo["cellsize_y"],
    }
    if pet_geo:
        cfg.update(
            pet_xllcorner=pet_geo["xllcorner"],
            pet_yllcorner=pet_geo["yllcorner"],
            pet_cellsize_x=pet_geo["cellsize_x"],
            pet_cellsize_y=pet_geo["cellsize_y"],
        )

    write_rri_input_txt(os.path.join(project_dir, "RRI_Input.txt"), cfg)
    gs.message(f"Wrote RRI project to {project_dir}")


def run_model_and_import(options, flags):
    project_dir = options["project_dir"]
    rri_bin = options["rri_bin"] or DEFAULT_RRI_BIN
    if not os.path.isfile(rri_bin):
        gs.fatal(
            f"rri_bin <{rri_bin}> not found -- build RRI.opencl first "
            "(cmake -B build && cmake --build build in $HOME/dev/RRI.opencl), "
            "or pass rri_bin= explicitly"
        )

    cmd = [rri_bin]
    if flags["g"]:
        cmd.append("--gpu")
    cmd.append(project_dir + os.sep)

    gs.message(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=project_dir, capture_output=True, text=True)
    if result.returncode != 0:
        gs.fatal(
            f"rri_cpu exited with status {result.returncode}\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    if result.stdout:
        gs.verbose(result.stdout)

    # rri_cpu writes hydro.txt/hydro_hr.txt directly under the project
    # root (datadir), not under outfile_storage's "./out/" prefix like
    # storage.dat -- see RRI.opencl's src/main.c, which builds this path
    # as "%shydro.txt" % datadir regardless of the RRI_Input.txt
    # hydro_file field's own "./out/" style default value. Confirmed by
    # running rri_cpu directly and checking where it actually landed,
    # not assumed from the config field's path.
    hydro_path = os.path.join(project_dir, "hydro.txt")
    table = options["hydrograph_table"] or (
        os.path.basename(os.path.normpath(project_dir)) + "_hydrograph"
    )
    if os.path.isfile(hydro_path):
        import csv
        import io

        tmp_stem = gs.tempfile()
        csv_path, csvt_path = tmp_stem + ".csv", tmp_stem + ".csvt"
        n_rows = 0
        with open(hydro_path) as src, open(csv_path, "w", newline="") as dst:
            writer = csv.writer(dst)
            writer.writerow(["time_s", "discharge_cms"])
            for line in src:
                # RRI.opencl's rri_cpu writes hydro.txt comma-separated
                # ("%.2f,%.5f\n" per station -- src/main.c), unlike the
                # original Fortran binary's whitespace-separated
                # "hydro.txt" -- confirmed by inspecting an actual run's
                # output, not assumed from either source's format.
                # Only the first station column is imported here; a
                # multi-station location.txt (this module currently only
                # ever writes one, "outlet") would need every column.
                fields = line.strip().split(",")
                if len(fields) < 2:
                    continue
                writer.writerow([fields[0], fields[1]])
                n_rows += 1
        with open(csvt_path, "w") as f:
            f.write("Real,Real\n")
        gs.run_command("db.in.ogr", input=csv_path, output=table, overwrite=True)
        gs.message(f"Imported {n_rows}-row outlet hydrograph into table <{table}>")
    else:
        gs.warning(f"{hydro_path} was not produced -- hydro_switch may be off")

    if options["final_storage"]:
        gs.warning(
            "final_storage= is not yet implemented -- RRI's periodic "
            "output grids (out/hs_*, out/hr_*) are per-cell-index binary/"
            "ascii dumps in the model's own compressed indexing, not a "
            "full-grid ESRI ASCII file re-importable via r.in.gdal as-is; "
            "wiring this up needs a small reprojection step not built in "
            "this pass. See README 'Known gaps'."
        )


def main():
    options, flags = gs.parser()

    prepare_project(options)

    if flags["r"]:
        run_model_and_import(options, flags)
    elif flags["g"]:
        gs.warning("-g has no effect without -r (nothing was run)")


if __name__ == "__main__":
    atexit.register(cleanup)
    sys.exit(main())
