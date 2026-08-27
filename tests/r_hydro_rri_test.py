"""Tests for r.hydro.rri: prepares RRI model inputs from GRASS rasters
and a synthetic precipitation STRDS, and checks the written files are
well-formed and (when a built RRI.opencl binary is present) actually
accepted by the model."""

import datetime
import os
import subprocess
import sys

import grass.script as gs
from grass.tools import Tools

MODULE_PATH = os.path.join(os.path.dirname(__file__), "..", "r.hydro.rri.py")
RRI_BIN = os.path.expanduser("~/dev/RRI.opencl/build/rri_cpu")


def _make_domain(session, tools, dem_name):
    """A small 10x10 synthetic sloped domain: elevation drops toward the
    south-east corner, so r.watershed drains everything there -- a
    single, unambiguous outlet for a smoke test. The row/col weights
    (1.7/1.3, not 1/1) are deliberately not equal: an exactly-planar
    "100 - row - col" surface makes every diagonal-vs-orthogonal flow
    comparison an exact tie, which sent r.watershed's D8 accumulation
    into a degenerate all-cells-flagged-unreliable state during this
    module's own development (see write_ascii_grid's abs(accumulation)
    handling) -- asymmetric weights avoid exact ties without needing a
    fully synthetic (e.g. r.surf.fractal) terrain."""
    tools.g_region(n=10, s=0, e=10, w=0, res=1)
    tools.r_mapcalc(
        expression=f"{dem_name} = 100.0 - row() * 1.7 - col() * 1.3",
        overwrite=True,
    )


def _make_rain_strds(session, tools, strds_name, n_days, mm_per_day):
    tools.t_create(
        output=strds_name,
        type="strds",
        temporaltype="absolute",
        title="synthetic rain",
        description="synthetic rain",
        overwrite=True,
    )
    maps = []
    start = datetime.date(2026, 1, 1)
    for day in range(n_days):
        name = f"{strds_name}_day_{day}"
        tools.r_mapcalc(expression=f"{name} = {mm_per_day}", overwrite=True)
        day_start = start + datetime.timedelta(days=day)
        day_end = day_start + datetime.timedelta(days=1)
        maps.append((name, day_start.isoformat(), day_end.isoformat()))
    # Interval (start|end) registration, matching a real t.in.era5-produced
    # daily STRDS -- not instant-point registration, which would leave
    # end_time unset and break write_forcing_series's elapsed-time math.
    register_input = "\n".join(f"{name}|{s}|{e}" for name, s, e in maps)
    register_file = gs.tempfile(env=session.env)
    with open(register_file, "w") as f:
        f.write(register_input)
    gs.run_command(
        "t.register",
        input=strds_name,
        type="raster",
        file=register_file,
        env=session.env,
    )
    return strds_name


def _run_module(session, args):
    cmd = [sys.executable, MODULE_PATH] + args
    result = subprocess.run(
        cmd, env=session.env, capture_output=True, text=True,
    )
    assert result.returncode == 0, (
        f"r.hydro.rri failed:\nstdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
    return result


def test_writes_rri_input_txt(session, tmp_path):
    tools = Tools(session=session)
    _make_domain(session, tools, "dem1")
    _make_rain_strds(session, tools, "rain_strds1", n_days=3, mm_per_day=24.0)

    project_dir = str(tmp_path / "rri_project")
    _run_module(
        session,
        [
            "elevation=dem1",
            "rain_strds=rain_strds1",
            f"project_dir={project_dir}",
            "lasth=72",
        ],
    )

    config_path = os.path.join(project_dir, "RRI_Input.txt")
    assert os.path.isfile(config_path)
    with open(config_path) as f:
        lines = f.read().splitlines()
    assert lines[0] == "RRI_Input_Format_Ver1_4_2"
    assert lines[9].split()[0] == "72"  # lasth

    for grid in ("dem.txt", "acc.txt", "dir.txt"):
        assert os.path.isfile(os.path.join(project_dir, "topo", grid))

    dem_grid = os.path.join(project_dir, "topo", "dem.txt")
    with open(dem_grid) as f:
        header = [f.readline() for _ in range(6)]
    assert header[0].split()[0] == "ncols"
    assert int(header[0].split()[1]) == 10
    assert int(header[1].split()[1]) == 10


def test_rain_dat_format_and_units(session, tmp_path):
    """1.0 mm_per_day should be written as 1/24 mm/h, in RRI's
    "t nx ny" + row-block layout, with a leading t=0 zero block."""
    tools = Tools(session=session)
    _make_domain(session, tools, "dem2")
    _make_rain_strds(session, tools, "rain_strds2", n_days=2, mm_per_day=24.0)

    project_dir = str(tmp_path / "rri_project")
    _run_module(
        session,
        [
            "elevation=dem2",
            "rain_strds=rain_strds2",
            "rain_units=mm_per_day",
            f"project_dir={project_dir}",
            "lasth=48",
        ],
    )

    rain_path = os.path.join(project_dir, "rain", "rain.dat")
    with open(rain_path) as f:
        lines = f.read().splitlines()

    header0 = lines[0].split()
    assert header0 == ["0", "10", "10"]
    for row in lines[1:11]:
        assert set(row.split()) <= {"0.000"}

    header1 = lines[11].split()
    assert header1[0] == "86400"
    assert header1[1:] == ["10", "10"]
    first_data_row = lines[12].split()
    assert abs(float(first_data_row[0]) - 1.0) < 1e-6  # 24 mm/d -> 1 mm/h

    header2 = lines[22].split()
    assert header2[0] == "172800"


def test_direction_reclass_has_no_invalid_codes(session, tmp_path):
    """Every non-outlet cell in the RRI-convention direction grid must
    be one of the 8 valid D8 bitmask codes (or 0 for an outlet/edge
    cell) -- catches a broken reclass expression before it ever reaches
    the model, where a wrong code would silently misroute flow instead
    of erroring."""
    tools = Tools(session=session)
    _make_domain(session, tools, "dem3")
    _make_rain_strds(session, tools, "rain_strds3", n_days=1, mm_per_day=10.0)

    project_dir = str(tmp_path / "rri_project")
    _run_module(
        session,
        [
            "elevation=dem3",
            "rain_strds=rain_strds3",
            f"project_dir={project_dir}",
            "lasth=24",
        ],
    )

    dir_path = os.path.join(project_dir, "topo", "dir.txt")
    with open(dir_path) as f:
        header = [f.readline() for _ in range(6)]
        rows = f.readlines()
    valid = {0, 1, 2, 4, 8, 16, 32, 64, 128}
    for row in rows:
        for token in row.split():
            assert int(token) in valid


def test_full_run_against_compiled_rri_binary(session, tmp_path):
    """If RRI.opencl has been built, run the actual model against the
    grids/config this module writes -- the cheapest real proof the
    forcing format is accepted by RRI, not just self-consistent."""
    if not os.path.isfile(RRI_BIN):
        import pytest

        pytest.skip(f"{RRI_BIN} not built -- skipping end-to-end run")

    tools = Tools(session=session)
    _make_domain(session, tools, "dem4")
    _make_rain_strds(session, tools, "rain_strds4", n_days=1, mm_per_day=48.0)

    project_dir = str(tmp_path / "rri_project")
    _run_module(
        session,
        [
            "elevation=dem4",
            "rain_strds=rain_strds4",
            f"project_dir={project_dir}",
            "lasth=6",
            "dt=600",
            "riv_thresh=5",  # low enough for a 10x10 test domain to form a channel
            "-r",
        ],
    )

    # rri_cpu writes hydro.txt under the project root, not out/ -- see
    # r.hydro.rri.py's run_model_and_import for why.
    hydro_path = os.path.join(project_dir, "hydro.txt")
    assert os.path.isfile(hydro_path)
    with open(hydro_path) as f:
        content = f.read().strip()
    assert content, "hydro.txt was produced but is empty"

    table = os.path.basename(project_dir) + "_hydrograph"
    tools = Tools(session=session)
    rows = tools.db_select(sql=f"select * from {table}").text
    assert rows.strip(), f"hydrograph table <{table}> is empty"
