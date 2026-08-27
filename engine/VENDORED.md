# Vendored from RRI.opencl

This directory is a **frozen copy** of `$HOME/dev/RRI.opencl`'s `src/`,
`include/`, `cl/`, and `CMakeLists.txt`, copied in (not symlinked) so
`r.hydro.rri` is fully self-contained -- a `g.extension` install of this
addon alone builds the hydrology engine, with no assumption that a
sibling `RRI.opencl` checkout exists on the target machine.

**Source and status at copy time**: `RRI.opencl`, validated end-to-end
against the compiled Fortran RRI reference (`RRI_1.4.2.7_Linux`) on the
real solo30s watershed dataset over its full 360-hour run: storage.dat
0.073% max relative error, outlet discharge (hydro.txt) 0.40% max / 0.16%
mean relative error, on both the CPU/OpenMP backend and the OpenCL
backend (confirmed on a real GPU, AMD Polaris10 via Mesa Clover). See
`RRI.opencl/README.md` for the full validation writeup, physics overview,
and data model this engine implements.

**One deliberate change from the original**, in `src/rri_opencl.c`: the
OpenCL kernel source files (`include/rri/kernels.h`, `cl/rri_kernels.cl`)
are read from disk at runtime, and the original code located them via an
absolute path baked in at compile time
(`${CMAKE_SOURCE_DIR}/include/rri/kernels.h`, etc.) -- correct for
RRI.opencl's own build-and-run-in-place development workflow, but wrong
once vendored: this addon's `Makefile` builds the engine in
`engine/build/` and then installs only the *binary* (plus copies of
those two kernel-source files) to `$(ETC)/r.hydro.rri/bin/`, and the
build directory is not guaranteed to still exist by the time the
installed binary actually runs. `rri_opencl.c` now checks a
`RRI_ENGINE_SHARE_DIR` environment variable first (set by
`r.hydro.rri.py` to wherever it finds the installed `kernels.h`/
`rri_kernels.cl`) and only falls back to the original compile-time path
if that variable is unset -- see that file's comment for the full
rationale. No other functional change was made to the vendored source.

**Not a live dependency.** If `RRI.opencl` is later bugfixed or extended,
that does not automatically propagate here -- re-vendoring (re-copying
`src/`/`include/`/`cl/`/`CMakeLists.txt` and re-applying the
`RRI_ENGINE_SHARE_DIR` change above) is a deliberate, manual step, not
something wired up to track the source repository automatically.

**Do not edit `$HOME/dev/RRI.opencl` from this addon's development, and
do not treat this copy as the place to develop new RRI.opencl features**
-- fixes belong upstream in `RRI.opencl` first, then re-vendored here.
