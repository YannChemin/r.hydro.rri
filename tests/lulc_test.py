"""Validates increment 7 (NATIVE_GRASS_PLAN.md 'Progress'): pixel-based
LULC via landuse= (static raster, category value = class, matching
RRI's own 1..num_of_landuse per-class parameter-array indexing) --
per-class ns_slope=/soildepth=/etc, not just a single scalar for the
whole domain.

Real check, not just "compiles and reads a raster": num_of_landuse is
correctly taken as the raster's max category value, per-class options
require an exact count (fail loudly, not silently reused/defaulted),
and -- the actual physics check -- giving two classes genuinely
different ns_slope actually changes routing behavior differently on
each half of a split domain, not just accepted syntactically."""

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
    out = str(tmp_path_factory.mktemp("build") / "r_hydro_rri_native_lulc")
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


def _setup_domain(tools, suffix):
    tools.g_region(n=10, s=0, e=10, w=0, res=1)
    tools.r_mapcalc(
        expression=f"dem_lulc_{suffix} = 100.0 - row() * 1.7 - col() * 1.3",
        overwrite=True,
    )
    tools.r_watershed(
        elevation=f"dem_lulc_{suffix}", drainage=f"drain_lulc_{suffix}",
        accumulation=f"acc_lulc_{suffix}", flags="s",
    )
    tools.r_mapcalc(expression=f"rain_lulc_{suffix} = 10.0", overwrite=True)


def test_wrong_value_count_fails_loudly(session, native_binary):
    from grass.tools import Tools

    tools = Tools(session=session)
    _setup_domain(tools, "count")
    tools.r_mapcalc(expression="landuse_lulc_count = if(col() <= 5, 1, 2)", overwrite=True)

    result = subprocess.run(
        [
            native_binary,
            "elevation=dem_lulc_count", "drainage=drain_lulc_count",
            "accumulation=acc_lulc_count", "riv_thresh=5",
            "landuse=landuse_lulc_count",
            "ns_slope=0.4,0.1,0.02",  # 3 values for a 2-class raster -- must fail
        ],
        env=session.env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert "ns_slope" in (result.stdout + result.stderr)


def test_two_classes_change_routing_differently(session, native_binary):
    from grass.tools import Tools

    tools = Tools(session=session)
    _setup_domain(tools, "physics")
    # West half (col<=5) = class 1 (very rough, slow), east half = class 2
    # (very smooth, fast) -- both halves get identical rain, so any
    # difference in resulting hillslope depth is attributable to the
    # per-class ns_slope actually being used, not noise.
    tools.r_mapcalc(
        expression="landuse_lulc_physics = if(col() <= 5, 1, 2)", overwrite=True,
    )

    result = subprocess.run(
        [
            native_binary,
            "elevation=dem_lulc_physics", "drainage=drain_lulc_physics",
            "accumulation=acc_lulc_physics", "riv_thresh=5",
            "landuse=landuse_lulc_physics",
            "ns_slope=1.0,0.01",  # class 1 much rougher than class 2
            "rain=rain_lulc_physics",
            "lasth=1", "dt=600", "dt_riv=60", "rain_units=mm_per_hour",
            "-r", "--v",
            "hs_output=lulc_hs_out",
        ],
        env=session.env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "landuse=<landuse_lulc_physics>: 2 class(es)" in (result.stdout + result.stderr)

    # Compare final hs between a west-half (class 1) and east-half
    # (class 2) cell -- rougher class 1 should retain MORE standing
    # water (slower drainage) than smooth class 2, under identical
    # uniform rain.
    import grass.script.array as garray

    maps = tools.t_rast_list(
        input="lulc_hs_out", columns="name", separator="|"
    ).text.strip().splitlines()[1:]
    last_map = maps[-1]
    arr = garray.array(mapname=last_map, env=session.env)
    hs_west = float(arr[5, 2])   # row 5, col 2 -> class 1 (col() <= 5, 1-based)
    hs_east = float(arr[5, 8])   # row 5, col 8 -> class 2
    assert hs_west > hs_east, (hs_west, hs_east)
