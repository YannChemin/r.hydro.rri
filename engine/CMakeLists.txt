cmake_minimum_required(VERSION 3.16)
project(rri_opencl C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_C_FLAGS_RELEASE "-O2")

find_package(OpenMP)
find_package(OpenCL)

set(RRI_CORE_SRCS
  src/rri_io.c
  src/rri_setup.c
  src/rri_rk.c
  src/rri_riv.c
  src/rri_slope.c
  src/rri_gw.c
  src/rri_infilt.c
  src/rri_rivslo.c
)

add_library(rri_core STATIC ${RRI_CORE_SRCS})
target_include_directories(rri_core PUBLIC include)
target_link_libraries(rri_core PUBLIC m)
if(OpenMP_C_FOUND)
  target_link_libraries(rri_core PUBLIC OpenMP::OpenMP_C)
endif()

# OpenCL backend (PLAN.md milestone 8): src/rri_opencl.c dispatches to
# cl/rri_kernels.cl, which #includes kernels.h's math bodies verbatim at
# runtime (see that file's header comment) -- so the kernel source paths
# are baked in as compile definitions rather than embedded at build time;
# the source tree (including cl/ and include/rri/kernels.h) must be
# present alongside the built binary, which is why the remote-GPU
# validation procedure in README.md rsyncs the whole tree rather than
# just a binary.
#
# rri_cpu (below) is built against rri_opencl whenever OpenCL is found --
# main.c's `--gpu` flag (rri_cl_backend_init et al.) is compiled in
# unconditionally, so the symbols must always be linkable; only whether
# `--gpu` is actually passed at runtime decides if OpenCL is touched.
if(OpenCL_FOUND)
  add_library(rri_opencl STATIC src/rri_opencl.c)
  target_include_directories(rri_opencl PUBLIC include ${OpenCL_INCLUDE_DIRS})
  target_link_libraries(rri_opencl PUBLIC rri_core ${OpenCL_LIBRARIES})
  target_compile_definitions(rri_opencl PRIVATE
    RRI_KERNELS_H_PATH="${CMAKE_SOURCE_DIR}/include/rri/kernels.h"
    RRI_CL_SRC_PATH="${CMAKE_SOURCE_DIR}/cl/rri_kernels.cl")
  message(STATUS "OpenCL found (${OpenCL_LIBRARIES}) -- rri_opencl backend built, rri_cpu supports --gpu")
else()
  message(FATAL_ERROR "OpenCL not found -- main.c requires it unconditionally for the --gpu code path (see src/main.c). Install an OpenCL ICD loader + headers, or ask for a CPU-only build variant if this needs to be optional again.")
endif()

add_executable(rri_cpu src/main.c)
target_link_libraries(rri_cpu PRIVATE rri_opencl)

# RRI.opencl's own tests/ (ctest unit tests, cross-backend validation)
# were deliberately not vendored into r.hydro.rri/engine/ -- this addon
# only needs the built rri_cpu binary at runtime, and r.hydro.rri has
# its own separate pytest suite validating the GRASS-facing behavior.
# The engine's own correctness is RRI.opencl's responsibility upstream;
# see engine/VENDORED.md for that validation's status as of the copy.
