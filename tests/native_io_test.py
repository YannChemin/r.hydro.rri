"""Validates the native GRASS-C I/O layer (main.c, rri_setup.c, rri_geo.c
-- NATIVE_GRASS_PLAN.md's first increment: static input + index setting,
no forcing/RK45/output yet) against the OLD, already-validated ASCII-file
architecture's known-good numbers on the identical synthetic domain.

This is compiled ad hoc here (plain gcc against GRASS's own libs/headers,
via `grass --config`) rather than through the project's real Makefile,
because the Makefile transition from Script.make (old r.hydro.rri.py
driver) to Module.make (this native module) is not yet done -- see
NATIVE_GRASS_PLAN.md section 7. Once that transition happens this test
should build through the normal addon Makefile instead.
"""

import os
import subprocess
import sys

import pytest

MODULE_DIR = os.path.join(os.path.dirname(__file__), "..")


def _grass_config(key):
    result = subprocess.run(["grass", "--config", key], capture_output=True, text=True)
    return result.stdout.strip()


@pytest.fixture(scope="module")
def native_binary(tmp_path_factory):
    grass_prefix = _grass_config("path")
    if not grass_prefix:
        pytest.skip("`grass` not found on PATH -- cannot resolve GRASS headers/libs")
    out = str(tmp_path_factory.mktemp("build") / "r_hydro_rri_native")
    cmd = [
        "gcc", "-std=c11", "-O2",
        "-I", os.path.join(grass_prefix, "include"),
        "-I", os.path.join(MODULE_DIR, "rri_include"),
        os.path.join(MODULE_DIR, "main.c"),
        os.path.join(MODULE_DIR, "rri_setup.c"),
        os.path.join(MODULE_DIR, "rri_geo.c"),
        "-L", os.path.join(grass_prefix, "lib"),
        "-lgrass_gis.8.6", "-lgrass_raster.8.6", "-lm",
        "-Wl,-rpath," + os.path.join(grass_prefix, "lib"),
        "-o", out,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    assert result.returncode == 0, f"native module failed to compile:\n{result.stderr}"
    return out


def test_matches_ascii_path_known_good_numbers(session, native_binary, tmp_path):
    """Same synthetic 10x10 domain (asymmetric-weight slope, see
    r_hydro_rri_test.py's _make_domain docstring for why the weights are
    asymmetric) the old ASCII-path pytest suite already validated end to
    end against the compiled RRI.opencl binary. riv_count=16, slo_count=100,
    dx=110479.427, dy=110582.711 are the exact numbers that run produced
    (tests/r_hydro_rri_test.py + this project's own manual cross-check
    during development, both against r.watershed -s output on this same
    DEM) -- if the native I/O layer disagrees with these, it has a real
    bug, not just a different-but-equally-valid interpretation."""
    from grass.tools import Tools

    tools = Tools(session=session)
    tools.g_region(n=10, s=0, e=10, w=0, res=1)
    tools.r_mapcalc(
        expression="dem_native_test = 100.0 - row() * 1.7 - col() * 1.3",
        overwrite=True,
    )
    tools.r_watershed(
        elevation="dem_native_test",
        drainage="drain_native_test",
        accumulation="acc_native_test",
        flags="s",
    )

    result = subprocess.run(
        [
            native_binary,
            "elevation=dem_native_test",
            "drainage=drain_native_test",
            "accumulation=acc_native_test",
            "riv_thresh=5",
            "--v",
        ],
        env=session.env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        f"native module run failed:\nstdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    output = result.stdout + result.stderr

    assert "riv_count=16 slo_count=100" in output, output
    assert "dx=110479.427 dy=110582.711" in output, output


def test_rain_read_and_index_matches_raster_mean(session, native_binary):
    """Increment 2 (see NATIVE_GRASS_PLAN.md 'Progress'): rain=
    reads/converts/indexes a raster into slope-idx space, but is not yet
    wired into a time loop. A spatially-VARYING raster (not uniform --
    a uniform raster would pass even with a broken index mapping, as
    long as the count is right) whose domain-wide mean is known via
    r.univar must produce the identical mean across qp_t_idx, proving
    both the read and the ij->idx conversion are correct, not just the
    cell count."""
    from grass.tools import Tools

    tools = Tools(session=session)
    tools.g_region(n=10, s=0, e=10, w=0, res=1)
    tools.r_mapcalc(
        expression="dem_native_rain_test = 100.0 - row() * 1.7 - col() * 1.3",
        overwrite=True,
    )
    tools.r_watershed(
        elevation="dem_native_rain_test",
        drainage="drain_native_rain_test",
        accumulation="acc_native_rain_test",
        flags="s",
    )
    tools.r_mapcalc(expression="rain_grad_native_test = col()", overwrite=True)
    expected_mean = float(
        tools.r_univar(map="rain_grad_native_test", flags="g").keyval["mean"]
    )

    result = subprocess.run(
        [
            native_binary,
            "elevation=dem_native_rain_test",
            "drainage=drain_native_rain_test",
            "accumulation=acc_native_rain_test",
            "riv_thresh=5",
            "rain=rain_grad_native_test",
            "--v",
        ],
        env=session.env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    output = result.stdout + result.stderr

    assert "slo_count=100" in output, output
    got_mean = None
    for line in output.splitlines():
        if "mean qp_t_idx" in line:
            got_mean = float(line.split("=")[-1].split()[0])
    assert got_mean is not None, output
    assert abs(got_mean - expected_mean) < 1e-6, (got_mean, expected_mean)
