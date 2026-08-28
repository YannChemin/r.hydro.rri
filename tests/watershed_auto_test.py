"""Validates increment 6 (NATIVE_GRASS_PLAN.md 'Progress'): auto-invoking
r.watershed.opencl when drainage=/accumulation= are omitted, mirroring
r.hydro.hbv.basins' drainage_input=/accumulation_input= passthrough
pattern -- default path derives them, explicit options skip that and
reuse what the caller already has."""

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
    out = str(tmp_path_factory.mktemp("build") / "r_hydro_rri_native_ws")
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


def test_auto_derives_drainage_accumulation_and_cleans_up(session, native_binary):
    from grass.tools import Tools

    tools = Tools(session=session)
    tools.g_region(n=10, s=0, e=10, w=0, res=1)
    tools.r_mapcalc(
        expression="dem_watershed_auto = 100.0 - row() * 1.7 - col() * 1.3",
        overwrite=True,
    )

    before = set(tools.g_list(type="raster").text.split())

    result = subprocess.run(
        [native_binary, "elevation=dem_watershed_auto", "riv_thresh=5", "--v"],
        env=session.env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "riv_count=16 slo_count=100" in (result.stdout + result.stderr)

    after = set(tools.g_list(type="raster").text.split())
    assert after == before, f"auto-derived temp rasters were not cleaned up: {after - before}"


def test_explicit_drainage_accumulation_skip_auto_derivation(session, native_binary):
    """Passing drainage=/accumulation= explicitly must not invoke
    r.watershed.opencl at all -- the passthrough path r.hydro.hbv.basins
    established this pattern for."""
    from grass.tools import Tools

    tools = Tools(session=session)
    tools.g_region(n=10, s=0, e=10, w=0, res=1)
    tools.r_mapcalc(
        expression="dem_watershed_explicit = 100.0 - row() * 1.7 - col() * 1.3",
        overwrite=True,
    )
    tools.r_watershed(
        elevation="dem_watershed_explicit", drainage="drain_watershed_explicit",
        accumulation="acc_watershed_explicit", flags="s",
    )

    result = subprocess.run(
        [
            native_binary,
            "elevation=dem_watershed_explicit",
            "drainage=drain_watershed_explicit",
            "accumulation=acc_watershed_explicit",
            "riv_thresh=5", "--v",
        ],
        env=session.env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "not given -- running" not in (result.stdout + result.stderr)
    assert "riv_count=16 slo_count=100" in (result.stdout + result.stderr)
