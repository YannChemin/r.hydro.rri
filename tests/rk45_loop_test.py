"""Validates increment 4 (NATIVE_GRASS_PLAN.md 'Progress'): the native
RK45 time loop (run_rk45_loop in main.c), cross-checked against the
vendored, already-Fortran-validated ASCII engine (engine/build/rri_cpu)
on the identical synthetic 10x10 domain -- the same standard increment 1
used for index setting, extended here to the full coupled physics loop.

This is the highest-risk piece of the native rewrite (adaptive Cash-Karp
RK45, signed-error-norm accept/reject, river<->slope coupling) -- see
this project's own history of subtle bugs here (zb/zb_riv conflation,
signed-vs-fabs error norm). Real bugs were found and fixed while
building this exact test: a NULL groundwater array crashed
rri_storage_calc (it dereferences hg unconditionally, the vendored
engine always allocates a real zeroed array even with groundwater
disabled), and a missing mm/h -> m/s unit conversion on the forcing
input made rain_sum wrong by orders of magnitude (the vendored engine's
load_rain() applies this once at load time; this port's preload step
does not, so the conversion has to happen at point of use in the loop
instead -- see run_rk45_loop's comment on this).
"""

import os
import subprocess

import pytest

import grass.script as gs

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
    out = str(tmp_path_factory.mktemp("build") / "r_hydro_rri_native_rk45")
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


def _parse_storage_line(text):
    """Pulls the LAST 't=N/M ...' diagnostic line's numeric fields into a
    dict, e.g. {'t': 6, 'rain_sum': 1.22e10, 'sout': ..., ...}.
    G_message word-wraps long lines (observed: the wrap point differs by
    timestep, so newlines are not a reliable field separator on their
    own) -- normalize newlines to spaces first, then anchor each record
    with a regex on 't=<int>/<int>' specifically, NOT a naive split on
    the substring "t=" (which also matches inside "dt=", "maxt=",
    "elapsed_s=", etc. -- an actual bug caught while writing this
    parser, not a hypothetical one)."""
    import re

    flat = text.replace("\n", " ")
    records = list(re.finditer(r"t=(\d+)/(\d+)\s", flat))
    assert records, text
    start = records[-1].start()
    end = records[-1].end() if len(records) == 1 else len(flat)
    segment = flat[start:end]

    fields = {"t": float(records[-1].group(1))}
    for tok in segment.replace(",", " ").split():
        if tok.startswith("t="):
            continue
        if "=" in tok:
            k, v = tok.split("=", 1)
            try:
                fields[k] = float(v)
            except ValueError:
                pass
    return fields


def test_rk45_loop_matches_ascii_engine_on_synthetic_domain(session, native_binary, tmp_path):
    if not os.path.isfile(ENGINE_BIN):
        pytest.skip(f"{ENGINE_BIN} not built -- run cmake/cmake --build in engine/ first")

    from grass.tools import Tools

    tools = Tools(session=session)
    tools.g_region(n=10, s=0, e=10, w=0, res=1)
    tools.r_mapcalc(
        expression="dem_rk45_xcheck = 100.0 - row() * 1.7 - col() * 1.3",
        overwrite=True,
    )
    tools.r_watershed(
        elevation="dem_rk45_xcheck", drainage="drain_rk45_xcheck",
        accumulation="acc_rk45_xcheck", flags="s",
    )
    tools.r_mapcalc(expression="rain_rk45_xcheck = 10.0", overwrite=True)

    # -- native path --
    result = subprocess.run(
        [
            native_binary,
            "elevation=dem_rk45_xcheck",
            "drainage=drain_rk45_xcheck",
            "accumulation=acc_rk45_xcheck",
            "riv_thresh=5",
            "rain=rain_rk45_xcheck",
            "lasth=1", "dt=600", "dt_riv=60", "rain_units=mm_per_hour",
            "-r", "--v",
        ],
        env=session.env, capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    native = _parse_storage_line(result.stdout + result.stderr)

    # -- old ASCII-path engine (vendored, kept only for this cross-check
    # -- see tests/ascii_reference.py), same domain, same 10 mm/h --
    from ascii_reference import write_reference_project

    project_dir = str(tmp_path / "ascii_xcheck")
    write_reference_project(
        tools, session, project_dir,
        elevation="dem_rk45_xcheck", accumulation="acc_rk45_xcheck",
        direction="drain_rk45_xcheck", rain_mm_per_hour=10.0,
        lasth=1, dt=600, dt_riv=60, riv_thresh=5,
    )

    result = subprocess.run([ENGINE_BIN, project_dir + os.sep], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    with open(os.path.join(project_dir, "out", "storage.dat")) as f:
        last_line = [l for l in f if l.strip()][-1]
    cols = [float(x) for x in last_line.split(",")]
    ascii_rain_sum, ascii_sout, ascii_storage, ascii_balance = cols[0], cols[3], cols[4], cols[5]

    # Relative agreement, not exact -- both accumulate floating-point sums
    # over the same physics in a different order (native: sc-idx loop;
    # ASCII: full ny*nx grid loop with a domain!=0 guard) and are compiled
    # by different toolchains/optimization settings.
    def rel(a, b):
        return abs(a - b) / max(abs(a), abs(b), 1e-30)

    assert rel(native["rain_sum"], ascii_rain_sum) < 1e-4
    assert rel(native["sout"], ascii_sout) < 1e-3
    assert rel(native["storage"], ascii_storage) < 1e-3
    # balance is itself a small residual (near-zero) -- compare as an
    # absolute fraction of rain_sum instead of relative to itself.
    assert abs(native["balance"]) / ascii_rain_sum < 1e-6
    assert abs(ascii_balance) / ascii_rain_sum < 1e-6
