"""Validates rain_units= (mm_per_day default, matching t.in.era5's one
and only output convention -- see NATIVE_GRASS_PLAN.md 'Progress' and
main.c's opt.rain_units description for why that default was chosen
over mm_per_hour). A raster of value 240.0 with rain_units=mm_per_day
must drive the RK45 loop identically to a raster of value 10.0 with
rain_units=mm_per_hour (240/24 == 10) -- not just report a converted
diagnostic number, but actually change what the physics loop sees."""

import os
import subprocess

import pytest

MODULE_DIR = os.path.join(os.path.dirname(__file__), "..")


def _grass_config(key):
    result = subprocess.run(["grass", "--config", key], capture_output=True, text=True)
    return result.stdout.strip()


@pytest.fixture(scope="module")
def native_binary(tmp_path_factory):
    grass_prefix = _grass_config("path")
    if not grass_prefix:
        pytest.skip("`grass` not found on PATH")
    out = str(tmp_path_factory.mktemp("build") / "r_hydro_rri_native_units")
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
    assert result.returncode == 0, result.stderr
    return out


def _final_rain_sum(stdout_stderr):
    import re
    matches = re.findall(r"rain_sum=([0-9.eE+-]+)", stdout_stderr)
    assert matches, stdout_stderr
    return float(matches[-1])


def test_mm_per_day_default_matches_equivalent_mm_per_hour_run(session, native_binary):
    from grass.tools import Tools

    tools = Tools(session=session)
    tools.g_region(n=10, s=0, e=10, w=0, res=1)
    tools.r_mapcalc(
        expression="dem_units_test = 100.0 - row() * 1.7 - col() * 1.3",
        overwrite=True,
    )
    tools.r_watershed(
        elevation="dem_units_test", drainage="drain_units_test",
        accumulation="acc_units_test", flags="s",
    )
    tools.r_mapcalc(expression="rain_units_test_hourly = 10.0", overwrite=True)
    tools.r_mapcalc(expression="rain_units_test_daily = 240.0", overwrite=True)

    common = [
        "elevation=dem_units_test", "drainage=drain_units_test",
        "accumulation=acc_units_test", "riv_thresh=5",
        "lasth=1", "dt=600", "dt_riv=60", "-r", "--v",
    ]

    hourly = subprocess.run(
        [native_binary] + common
        + ["rain=rain_units_test_hourly", "rain_units=mm_per_hour"],
        env=session.env, capture_output=True, text=True,
    )
    assert hourly.returncode == 0, hourly.stdout + hourly.stderr

    daily_explicit = subprocess.run(
        [native_binary] + common
        + ["rain=rain_units_test_daily", "rain_units=mm_per_day"],
        env=session.env, capture_output=True, text=True,
    )
    assert daily_explicit.returncode == 0, daily_explicit.stdout + daily_explicit.stderr

    # Default (no rain_units= given at all) must be mm_per_day too.
    daily_default = subprocess.run(
        [native_binary] + common + ["rain=rain_units_test_daily"],
        env=session.env, capture_output=True, text=True,
    )
    assert daily_default.returncode == 0, daily_default.stdout + daily_default.stderr

    rs_hourly = _final_rain_sum(hourly.stdout + hourly.stderr)
    rs_daily_explicit = _final_rain_sum(daily_explicit.stdout + daily_explicit.stderr)
    rs_daily_default = _final_rain_sum(daily_default.stdout + daily_default.stderr)

    assert rs_hourly == rs_daily_explicit == rs_daily_default
