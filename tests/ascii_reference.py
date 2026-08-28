"""Builds a minimal RRI ASCII-format project (RRI_Input.txt + topo/*.txt
+ rain/rain.dat) for the vendored engine/build/rri_cpu binary, used ONLY
by this test suite's cross-checks against the native module (main.c).

This is deliberately NOT r.hydro.rri.py reborn -- that Python
subprocess-wrapper driver was retired (NATIVE_GRASS_PLAN.md section 7)
because the native compiled module supersedes it as a real, user-facing
way to run RRI from GRASS. What's here is a small, test-only ASCII
writer whose only job is producing inputs for engine/build/rri_cpu so
tests can keep comparing the native module's numbers against that
independent (Fortran-validated) reference -- the same value the retired
driver provided for cross-checking, without resurrecting it as
production code.
"""

import os


def write_grid(tools, mapname, path, integer=False):
    kwargs = {"input": mapname, "output": path, "null_value": "-9999", "flags": "l"}
    if integer:
        kwargs["flags"] += "i"
    tools.r_out_ascii(**kwargs)


def write_reference_project(tools, session, project_dir, elevation, accumulation,
                             direction, rain_mm_per_hour, lasth, dt, dt_riv, riv_thresh):
    """Writes a one-landuse-class, constant-rain-for-the-whole-run RRI
    ASCII project to project_dir, matching this test suite's synthetic
    10x10 domains and the field order/defaults RRI_Input.txt needs
    (cross-checked against RRI_1.4.2.7_Linux/solo30s/RRI_Input.txt
    during this module's original development -- see README.md
    "Validation")."""
    for sub in ("topo", "rain", "riv", "init", "bound", "out"):
        os.makedirs(os.path.join(project_dir, sub), exist_ok=True)

    write_grid(tools, elevation, os.path.join(project_dir, "topo", "dem.txt"))
    write_grid(tools, accumulation, os.path.join(project_dir, "topo", "acc.txt"), integer=True)

    # Reclass r.watershed's drainage convention to RRI's D8 bitmask --
    # duplicated here (not imported from the retired r.hydro.rri.py)
    # since this is now purely test-fixture code, not production logic.
    rri_dir = "ascii_ref_dir_tmp"
    codes = {1: 128, 2: 64, 3: 32, 4: 16, 5: 8, 6: 4, 7: 2, 8: 1}
    cases = " ".join(f"if(abs({direction})=={c},{r}," for c, r in codes.items())
    expr = f"{rri_dir} = if(isnull({direction}) || {direction} < 0, 0, {cases}0{')' * len(codes)})"
    tools.r_mapcalc(expression=expr, overwrite=True)
    write_grid(tools, rri_dir, os.path.join(project_dir, "topo", "dir.txt"), integer=True)

    region = tools.g_region(flags="pg").keyval
    ncols, nrows = int(region["cols"]), int(region["rows"])
    with open(os.path.join(project_dir, "rain", "rain.dat"), "w") as f:
        f.write(f"{0:>15d}{ncols:>6d}{nrows:>6d}\n")
        for _ in range(nrows):
            f.write((" " * 8 + "0.000") * ncols + "\n")
        elapsed_s = int(lasth * 3600) + 1
        f.write(f"{elapsed_s:>15d}{ncols:>6d}{nrows:>6d}\n")
        for _ in range(nrows):
            f.write("".join(f"{rain_mm_per_hour:13.3f}" for _ in range(ncols)) + "\n")

    lines = [
        "RRI_Input_Format_Ver1_4_2", "",
        "./rain/rain.dat", "./topo/dem.txt", "./topo/acc.txt", "./topo/dir.txt", "",
        "0    # utm(1) or latlon(0)",
        "1    # 4-direction (0), 8-direction(1)",
        f"{lasth}    # lasth(hour)",
        f"{dt}    # dt(second)",
        f"{dt_riv}    # dt_riv",
        "24    # outnum [-]",
        f"{float(region['w']):.6f}   # xllcorner_rain",
        f"{float(region['s']):.6f}    # yllcorner_rain",
        f"{float(region['ewres']):.10f} {float(region['nsres']):.10f}    # cellsize_rain", "",
        "0.030     # ns_river",
        "1    # num_of_landuse",
        "1    # diffusion(1) or kinematic(0)",
        "0.400     # ns_slope", "1.000     # soildepth", "0.475     # gammaa", "",
        "0.000     # ksv", "0.316     # faif", "",
        "0.000     # ka", "0.000     # gammam", "8.000     # beta", "",
        "0.000     # kgv", "0.400     # gammag", "0.00050     # tg", "0.030     # fpg",
        "0.500     # init_cond_gw", "",
        f"{riv_thresh}      # riv_thresh",
        "5.000      # width_param_c (2.5)", "0.350      # width_param_s (0.4)",
        "0.950      # depth_param_c (0.1)", "0.200      # depth_param_s (0.4)",
        "0.000      # height_param", "20       # height_limit_param", "",
        "0", "./riv/width.txt", "./riv/depth.txt", "./riv/height.txt", "",
        "0  0  0  0",
        "./init/hs_init_dummy.out", "./init/hr_init_dummy.out",
        "./init/hg_init_dummy.out", "./init/gamptff_init_dummy.out", "",
        "0  0", "./bound/hs_bound.txt", "./bound/hr_bound.txt", "",
        "0  0", "./bound/qs_bound.txt", "./bound/qr_bound.txt", "",
        "0", "./topo/landuse.txt", "",
        "0", "./dam.txt", "",
        "0", "./div.txt", "",
        "0", "./evp/pet.dat",
        "0.000000      # xllcorner_evp", "0.000000      # yllcorner_evp",
        "0.0100000000  0.0100000000     # cellsize", "",
        "0", "./riv/length.txt", "",
        "0", "./riv/sec_map.txt", "./riv/section/sec_", "",
        "1  1  0  1  0  0  0  0  0  1",
        "./out/hs_", "./out/hr_", "./out/hg_", "./out/qr_", "./out/qu_", "./out/qv_",
        "./out/gu_", "./out/gv_", "./out/gampt_ff_", "./out/storage.dat", "",
        "1", "./location.txt", "",
    ]
    with open(os.path.join(project_dir, "RRI_Input.txt"), "w") as f:
        f.write("\n".join(lines))

    import numpy as np

    import grass.script.array as garray

    acc_arr = garray.array(mapname=accumulation, env=session.env)
    row, col = (int(i) for i in np.unravel_index(np.argmax(np.abs(np.asarray(acc_arr))), acc_arr.shape))
    with open(os.path.join(project_dir, "location.txt"), "w") as f:
        f.write(f"outlet {row + 1} {col + 1}\n")

    tools.g_remove(flags="f", type="raster", name=rri_dir)
