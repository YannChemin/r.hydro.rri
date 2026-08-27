PGM = r.hydro.rri

include $(MODULE_TOPDIR)/include/Make/Script.make

default: script engine

# Builds the vendored RRI.opencl engine (engine/, a frozen copy -- see
# engine/VENDORED.md for provenance/validation status) with CMake, and
# installs the resulting rri_cpu/rri_cl binaries plus the two OpenCL
# kernel-source files they read at runtime (kernels.h, rri_kernels.cl --
# see engine/src/rri_opencl.c's RRI_ENGINE_SHARE_DIR handling for why
# those two travel with the binary rather than staying baked to the
# build directory) into $(ETC)/$(PGM)/bin/. Module.make's own ETCFILES
# mechanism only handles plain data files copied verbatim, not a
# CMake build step, so this is a custom rule -- same idea as
# r.hydro.hbv's own install-data rule for its bundled CSVs, but building
# (not just copying) a compiled component. $(ETC)/$(PGM) is picked up
# automatically by Script.make's own `install:` rule
# (`cp -RL $(ETC)/$(PGM) $(INST_DIR)/etc/`), so nothing else has to
# change for `g.extension`/`make install` to ship it.
engine:
	cmake -S engine -B engine/build -DCMAKE_BUILD_TYPE=Release
	cmake --build engine/build -j
	$(MKDIR) $(ETC)/$(PGM)/bin
	$(INSTALL) engine/build/rri_cpu $(ETC)/$(PGM)/bin/
	$(INSTALL_DATA) engine/include/rri/kernels.h $(ETC)/$(PGM)/bin/
	$(INSTALL_DATA) engine/cl/rri_kernels.cl $(ETC)/$(PGM)/bin/

.PHONY: engine
