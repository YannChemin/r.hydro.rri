MODULE_TOPDIR = ../..

PGM = r.hydro.rri

LIBES = $(GISLIB) $(RASTERLIB) $(MATHLIB)
DEPENDENCIES = $(GISDEP) $(RASTERDEP)
EXTRA_INC = -Irri_include

# main.c + the physics-core sources reused unchanged from the vendored
# RRI.opencl copy (engine/, see engine/VENDORED.md) -- rri_setup.c,
# rri_geo.c, rri_riv.c, rri_slope.c, rri_rivslo.c, rri_infilt.c, rri_rk.c
# are symlinks into engine/src/, picked up by Module.make's normal *.c
# auto-discovery in this directory; rri_include/rri is a symlink to
# engine/include/rri for the same reason. engine/'s own CMake build
# (engine/build/rri_cpu, the now-retired ASCII/subprocess architecture's
# binary) is NOT built by this Makefile -- it is no longer a runtime
# dependency of this addon. It remains buildable manually, for this
# project's own test suite to cross-check the native module's numbers
# against (see tests/ascii_reference.py and README.md "Validation") --
# see NATIVE_GRASS_PLAN.md section 7 for the full history of why engine/
# is still here as a vendored source reference despite the subprocess
# architecture it originally existed to support being retired.

include $(MODULE_TOPDIR)/include/Make/Module.make

default: cmd
