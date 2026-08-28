"""Validates increment 5 (NATIVE_GRASS_PLAN.md 'Progress'): GRASS-native
output -- an outlet hydrograph DB table and a periodic hillslope-depth
STRDS, written directly (no ASCII intermediate) during the RK45 loop.

Cross-checked against the vendored ASCII engine's own hydro.txt on the
identical domain/config: a real finding surfaced while building this
test -- the synthetic 10x10 domain's single classified outlet river
cell reports EXACTLY ZERO discharge for 24 simulated hours in BOTH
engines (confirmed by running engine/build/rri_cpu with the equivalent
config and reading its hydro.txt directly), a genuine property of this
domain (huge ~110km cells given its degree-based synthetic region, so a
parametric channel sized from a 9-cell contributing area is
physically negligible relative to the domain's hillslope storage) --
not a bug in either engine. Documented here rather than assumed silently
correct, since "always zero" is exactly the kind of result that could
also indicate a real bug (e.g. reading the wrong cell) if it hadn't been
cross-checked.
"""

import os
import subprocess

import pytest

MODULE_DIR = os.path.join(os.path.dirname(__file__), "..")
ENGINE_BIN = os.path.join(MODULE_DIR, "engine", "build", "rri_cpu")


def _grass_config(key):
    result = subprocess.run(["grass", "--config", key], capture_output=True, text=True)
    return result.stdout.strip()


@pytest.fixture(scope="module")
def native_binary(tmp_path_factory):
    grass_prefix = _grass_config("path")
    if not grass_prefix:
        pytest.skip("`grass` not found on PATH -- cannot resolve GRASS headers/libs")
    out = str(tmp_path_factory.mktemp("build") / "r_hydro_rri_native_out")
    sources = [
        "main.c", "rri_setup.c", "rri_geo.c",
        "rri_riv.c", "rri_slope.c", "rri_rivslo.c", "rri_infilt.c", "rri_rk.c",
    ]
    cmd = (
        ["gcc", "-std=c11", "-O2",
         "-I", os.path.join(grass_prefix, "include"),
         "-I", os.path.join(MODULE_DIR, "rri_include")]
        + [os.path.join(MODULE_DIR, s) for s in sources]
        + ["-L", os.path.join(grass_prefix, "lib"),
           "-lgrass_gis.8.6", "-lgrass_raster.8.6", "-lm",
           "-Wl,-rpath," + os.path.join(grass_prefix, "lib"),
           "-o", out]
    )
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"native module failed to compile:\n{result.stderr}"
    return out


def test_hydrograph_table_and_hs_strds_written(session, native_binary):
    from grass.tools import Tools

    tools = Tools(session=session)
    tools.g_region(n=10, s=0, e=10, w=0, res=1)
    tools.r_mapcalc(
        expression="dem_output_test = 100.0 - row() * 1.7 - col() * 1.3",
        overwrite=True,
    )
    tools.r_watershed(
        elevation="dem_output_test", drainage="drain_output_test",
        accumulation="acc_output_test", flags="s",
    )
    tools.r_mapcalc(expression="rain_output_test = 10.0", overwrite=True)

    result = subprocess.run(
        [
            native_binary,
            "elevation=dem_output_test",
            "drainage=drain_output_test",
            "accumulation=acc_output_test",
            "riv_thresh=5",
            "rain=rain_output_test",
            "lasth=1", "dt=600", "dt_riv=60", "rain_units=mm_per_hour",
            "-r", "--v",
            "hs_output=output_test_hs",
            "hydrograph_table=output_test_hydro",
        ],
        env=session.env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    # -- hydrograph DB table: right shape, right row count, and the
    # cross-checked (not just assumed) all-zero discharge on this domain --
    rows = tools.db_select(sql="select * from output_test_hydro").text.strip().splitlines()
    assert rows[0] == "time_s|discharge_cms"
    data_rows = rows[1:]
    assert len(data_rows) == 6, data_rows
    times = [float(r.split("|")[0]) for r in data_rows]
    assert times == [600.0, 1200.0, 1800.0, 2400.0, 3000.0, 3600.0]
    discharges = [float(r.split("|")[1]) for r in data_rows]
    assert all(d == 0.0 for d in discharges), (
        "expected all-zero discharge on this domain (see module docstring) "
        f"-- got {discharges}"
    )

    # -- hs STRDS: registered, right map count, chronological --
    info = tools.t_info(input="output_test_hs", type="strds", flags="g").keyval
    assert int(info["number_of_maps"]) == 6, info

    maps = tools.t_rast_list(
        input="output_test_hs", columns="name,start_time", separator="|"
    ).text.strip().splitlines()[1:]
    assert len(maps) == 6


def test_hydro_zero_matches_ascii_engine_cross_check(session, tmp_path):
    """Same domain/config through the OLD (still-present, still useful
    for exactly this) ASCII engine -- confirms the native path's
    all-zero discharge isn't a native-only artifact."""
    if not os.path.isfile(ENGINE_BIN):
        pytest.skip(f"{ENGINE_BIN} not built")

    import datetime

    from grass.tools import Tools

    import grass.script as gs

    tools = Tools(session=session)
    tools.g_region(n=10, s=0, e=10, w=0, res=1)
    tools.r_mapcalc(
        expression="dem_output_xcheck = 100.0 - row() * 1.7 - col() * 1.3",
        overwrite=True,
    )
    tools.t_create(
        output="rain_output_xcheck_strds", type="strds", temporaltype="absolute",
        title="x", description="x", overwrite=True,
    )
    tools.r_mapcalc(expression="rain_output_xcheck = 10.0", overwrite=True)
    d0 = datetime.date(2026, 1, 1)
    d1 = d0 + datetime.timedelta(days=1)
    register_file = gs.tempfile(env=session.env)
    with open(register_file, "w") as f:
        f.write(f"rain_output_xcheck|{d0.isoformat()}|{d1.isoformat()}")
    gs.run_command(
        "t.register", input="rain_output_xcheck_strds", type="raster",
        file=register_file, env=session.env,
    )

    project_dir = str(tmp_path / "ascii_output_xcheck")
    old_driver = os.path.join(MODULE_DIR, "r.hydro.rri.py")
    result = subprocess.run(
        [
            "python3", old_driver,
            "elevation=dem_output_xcheck",
            "rain_strds=rain_output_xcheck_strds",
            "rain_units=mm_per_hour",
            f"project_dir={project_dir}",
            "lasth=24", "dt=600", "dt_riv=60", "riv_thresh=5",
        ],
        env=session.env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr

    result = subprocess.run([ENGINE_BIN, project_dir + os.sep], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr

    with open(os.path.join(project_dir, "hydro.txt")) as f:
        lines = [l for l in f if l.strip()]
    assert lines, "hydro.txt was empty"
    for line in lines:
        discharge = float(line.strip().split(",")[1])
        assert discharge == 0.0, line
