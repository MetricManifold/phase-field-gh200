cmake_minimum_required(VERSION 3.24)

project(phase_field_gh200 VERSION 0.1.0 LANGUAGES CXX CUDA)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_EXTENSIONS OFF)
set(CMAKE_CUDA_SEPARABLE_COMPILATION OFF)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 90)
endif()

# Linux/GH200 release builds use the static CUDA runtime. MSVC's default /MD
# runtime instead requires shared CUDA runtime linkage.
if(MSVC)
  set(PF_CUDA_RUNTIME Shared)
else()
  set(PF_CUDA_RUNTIME Static)
endif()

# Fatal integrity checks remain enabled. This option adds a boundary-contact
# counter for fields whose detected support reaches an active-window edge.
option(PF_DIAGNOSTIC_ALARMS
       "Enable the high-frequency support-edge counter" OFF)
option(PF_BUILD_TOOLS "Build initializer tools" ON)
option(PF_BUILD_3D "Build the separate three-dimensional simulator" ON)
option(PF_ENABLE_CHECKPOINT_IO_TESTS
  "Register checkpoint I/O tests that stage through any CUDA device without launching kernels" OFF)
option(PF_ENABLE_GPU_TESTS "Register the public CUDA smoke tests" OFF)

add_library(pf_shape_config INTERFACE)
if(PF_DIAGNOSTIC_ALARMS)
  target_compile_definitions(pf_shape_config INTERFACE PF_ALARMS=1)
endif()

# Preserve the specified arithmetic and FMA-contraction policy. Fast-math can
# change the diffuse-interface profile and reductions, so it is omitted. The
# ptxas report exposes register spills at build time.
set(PF_CUDA_FLAGS
  --expt-relaxed-constexpr
  -fmad=true
  -lineinfo
  -Xptxas=-v)

function(pf_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:/W4>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/W4>)
  elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    target_compile_options(${target} PRIVATE
      $<$<COMPILE_LANGUAGE:CXX>:-Wall>
      $<$<COMPILE_LANGUAGE:CXX>:-Wextra>
      $<$<COMPILE_LANGUAGE:CXX>:-Wpedantic>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wall>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wextra>
      $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=-Wpedantic>)
  endif()
endfunction()

add_library(pf_core STATIC
  src/kernels.cu
  src/sim.cu
  src/checkpoint.cu
  src/trajectory.cpp)
target_link_libraries(pf_core PUBLIC pf_shape_config)
target_include_directories(pf_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${CMAKE_CURRENT_SOURCE_DIR}/common
  ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
target_compile_options(pf_core PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
if(MSVC)
  # The portable C stdio calls are intentional; the suggested *_s variants
  # are specific to Microsoft's runtime.
  target_compile_definitions(pf_core PRIVATE _CRT_SECURE_NO_WARNINGS)
endif()
set_target_properties(pf_core PROPERTIES
  CUDA_RUNTIME_LIBRARY ${PF_CUDA_RUNTIME})
pf_enable_warnings(pf_core)

add_executable(cell_gh200 src/main.cu)
target_link_libraries(cell_gh200 PRIVATE pf_core)
target_compile_options(cell_gh200 PRIVATE
  $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
set_target_properties(cell_gh200 PROPERTIES
  CUDA_RUNTIME_LIBRARY ${PF_CUDA_RUNTIME})
pf_enable_warnings(cell_gh200)

if(PF_BUILD_3D)
  add_library(pf3d_core STATIC
    src/pf3d/kernels.cu
    src/pf3d/kernels_periodic_slab_walk.cu
    src/pf3d/sim.cu
    src/pf3d/checkpoint.cu)
  target_link_libraries(pf3d_core PUBLIC pf_shape_config)
  target_include_directories(pf3d_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common)
  target_compile_options(pf3d_core PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
  set_target_properties(pf3d_core PROPERTIES
    CUDA_RUNTIME_LIBRARY ${PF_CUDA_RUNTIME})
  pf_enable_warnings(pf3d_core)

  add_executable(cell_gh200_3d src/pf3d/main.cu)
  target_link_libraries(cell_gh200_3d PRIVATE pf3d_core)
  target_compile_options(cell_gh200_3d PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
  set_target_properties(cell_gh200_3d PROPERTIES
    CUDA_RUNTIME_LIBRARY ${PF_CUDA_RUNTIME})
  pf_enable_warnings(cell_gh200_3d)
endif()

if(CMAKE_CXX_COMPILER_ID STREQUAL "NVHPC")
  set(PF_HOST_NO_CONTRACTION -Mnofma)
  set(PF_HOST_NO_CONTRACTION_KNOWN TRUE)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
  set(PF_HOST_NO_CONTRACTION -ffp-contract=off)
  set(PF_HOST_NO_CONTRACTION_KNOWN TRUE)
elseif(MSVC)
  set(PF_HOST_NO_CONTRACTION /fp:strict)
  set(PF_HOST_NO_CONTRACTION_KNOWN TRUE)
else()
  set(PF_HOST_NO_CONTRACTION_KNOWN FALSE)
endif()

# nvcc may use a different host compiler than CMake uses for C++ sources.
# Select its flag independently so, for example, GCC never receives -Mnofma.
set(PF_CUDA_HOST_ID "${CMAKE_CUDA_HOST_COMPILER_ID}")
if(NOT PF_CUDA_HOST_ID AND CMAKE_CUDA_SIMULATE_ID)
  set(PF_CUDA_HOST_ID "${CMAKE_CUDA_SIMULATE_ID}")
endif()
if(NOT PF_CUDA_HOST_ID)
  set(PF_CUDA_HOST_ID "${CMAKE_CXX_COMPILER_ID}")
endif()
if(PF_CUDA_HOST_ID STREQUAL "NVHPC")
  set(PF_CUDA_HOST_NO_CONTRACTION -Mnofma)
  set(PF_CUDA_HOST_NO_CONTRACTION_KNOWN TRUE)
elseif(PF_CUDA_HOST_ID MATCHES "^(GNU|Clang|AppleClang)$")
  set(PF_CUDA_HOST_NO_CONTRACTION -ffp-contract=off)
  set(PF_CUDA_HOST_NO_CONTRACTION_KNOWN TRUE)
elseif(PF_CUDA_HOST_ID STREQUAL "MSVC")
  set(PF_CUDA_HOST_NO_CONTRACTION /fp:strict)
  set(PF_CUDA_HOST_NO_CONTRACTION_KNOWN TRUE)
else()
  set(PF_CUDA_HOST_NO_CONTRACTION_KNOWN FALSE)
endif()

# Fresh 3D runs execute the same inline placement routine as the standalone
# centre-table generator. Apply the verified host contraction policy to CUDA
# translation units so both paths make identical acceptance decisions.
if(PF_BUILD_3D)
  if(NOT PF_HOST_NO_CONTRACTION_KNOWN OR
     NOT PF_CUDA_HOST_NO_CONTRACTION_KNOWN)
    message(FATAL_ERROR
      "No verified no-contraction flags for C++ compiler "
      "${CMAKE_CXX_COMPILER_ID} and CUDA host compiler ${PF_CUDA_HOST_ID}")
  endif()
  target_compile_options(pf3d_core PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${PF_HOST_NO_CONTRACTION}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${PF_CUDA_HOST_NO_CONTRACTION}>)
  target_compile_options(cell_gh200_3d PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:${PF_HOST_NO_CONTRACTION}>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${PF_CUDA_HOST_NO_CONTRACTION}>)
endif()

if(PF_BUILD_TOOLS)
  if(PF_HOST_NO_CONTRACTION_KNOWN)
    add_executable(palmieri_centres tools/palmieri_centres.cpp)
    target_include_directories(palmieri_centres PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    target_compile_options(palmieri_centres PRIVATE ${PF_HOST_NO_CONTRACTION})
    if(PF_BUILD_3D)
      add_executable(palmieri_centres_3d tools/palmieri_centres_3d.cpp)
      target_include_directories(palmieri_centres_3d PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/common)
      target_compile_options(palmieri_centres_3d PRIVATE
        ${PF_HOST_NO_CONTRACTION})
    endif()
  else()
    message(STATUS
      "palmieri_centres disabled: no verified no-contraction host flag for "
      "${CMAKE_CXX_COMPILER_ID}")
  endif()
endif()

include(CTest)
if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)

  add_executable(common_primitives_cpu_test
    tests/common_primitives_cpu_test.cpp)
  target_include_directories(common_primitives_cpu_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common)
  pf_enable_warnings(common_primitives_cpu_test)
  add_test(NAME common_primitives_cpu COMMAND common_primitives_cpu_test)

  add_executable(pf2d_trajectory_contract_test
    tests/trajectory_contract_test.cpp
    src/trajectory.cpp)
  target_include_directories(pf2d_trajectory_contract_test PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/common
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
  if(MSVC)
    target_compile_definitions(pf2d_trajectory_contract_test
      PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
  pf_enable_warnings(pf2d_trajectory_contract_test)
  add_test(NAME pf2d_trajectory_contract
    COMMAND pf2d_trajectory_contract_test)

  add_executable(pf2d_checkpoint_fixture
    tests/checkpoint_fixture.cpp)
  target_include_directories(pf2d_checkpoint_fixture PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/common)
  if(MSVC)
    target_compile_definitions(pf2d_checkpoint_fixture
      PRIVATE _CRT_SECURE_NO_WARNINGS)
  endif()
  pf_enable_warnings(pf2d_checkpoint_fixture)
  add_test(NAME pf2d_exporter_contract
    COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/exporter_contract_test.py
      --fixture-writer $<TARGET_FILE:pf2d_checkpoint_fixture>
      --exporter
        ${CMAKE_CURRENT_SOURCE_DIR}/tools/relaxed_2d_to_slab_centres.py)

  if(PF_ENABLE_CHECKPOINT_IO_TESTS)
    add_executable(pf2d_checkpoint_seed_test
      tests/checkpoint_seed_test.cu)
    target_link_libraries(pf2d_checkpoint_seed_test PRIVATE pf_core)
    target_compile_options(pf2d_checkpoint_seed_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
    set_target_properties(pf2d_checkpoint_seed_test PROPERTIES
      CUDA_RUNTIME_LIBRARY ${PF_CUDA_RUNTIME})
    pf_enable_warnings(pf2d_checkpoint_seed_test)
    add_test(NAME pf2d_checkpoint_seed COMMAND pf2d_checkpoint_seed_test)
    set_tests_properties(pf2d_checkpoint_seed PROPERTIES SKIP_RETURN_CODE 77)
  endif()

  if(PF_BUILD_3D)
    add_executable(pf3d_one_step_cpu_test
      tests/3d/model_one_step_cpu_test.cpp)
    target_include_directories(pf3d_one_step_cpu_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include)
    pf_enable_warnings(pf3d_one_step_cpu_test)
    if(PF_HOST_NO_CONTRACTION_KNOWN)
      target_compile_options(pf3d_one_step_cpu_test PRIVATE
        ${PF_HOST_NO_CONTRACTION})
    endif()
    add_test(NAME pf3d_one_step_cpu COMMAND pf3d_one_step_cpu_test)

    add_executable(pf3d_measure_shards_cpu_test
      tests/3d/measure_shards_cpu_test.cpp)
    target_include_directories(pf3d_measure_shards_cpu_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/common)
    pf_enable_warnings(pf3d_measure_shards_cpu_test)
    add_test(NAME pf3d_measure_shards_cpu
      COMMAND pf3d_measure_shards_cpu_test)

    add_executable(pf3d_wall_field_contract_test
      tests/3d/wall_field_contract_test.cu)
    target_include_directories(pf3d_wall_field_contract_test PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/common)
    target_compile_options(pf3d_wall_field_contract_test PRIVATE
      $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
    set_target_properties(pf3d_wall_field_contract_test PROPERTIES
      CUDA_RUNTIME_LIBRARY ${PF_CUDA_RUNTIME})
    pf_enable_warnings(pf3d_wall_field_contract_test)
    add_test(NAME pf3d_wall_field_contract
      COMMAND pf3d_wall_field_contract_test)

    add_executable(pf3d_checkpoint_header_fixture
      tests/3d/checkpoint_header_fixture.cpp)
    target_include_directories(pf3d_checkpoint_header_fixture PRIVATE
      ${CMAKE_CURRENT_SOURCE_DIR}/common)
    pf_enable_warnings(pf3d_checkpoint_header_fixture)
    add_test(NAME pf3d_checkpoint_header_python
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/3d/checkpoint_header_python_test.py
        --fixture-writer $<TARGET_FILE:pf3d_checkpoint_header_fixture>)

    # Executable-level option parsing; all cases stop before device setup.
    set(pf3d_cli_runner
      ${CMAKE_CURRENT_SOURCE_DIR}/tests/3d/run_cli_case.cmake)
    foreach(pf3d_ms_bad -2 65 eleven)
      add_test(NAME pf3d_measure_shards_cli_invalid_${pf3d_ms_bad}
        COMMAND ${CMAKE_COMMAND}
          -DEXE=$<TARGET_FILE:cell_gh200_3d>
          "-DCASE_ARGS=--measure-shards;${pf3d_ms_bad};--N;8"
          -DEXPECT_CODE=2
          "-DEXPECT_LINE=[3d] --measure-shards expects an integer in [-1, 64], got '${pf3d_ms_bad}'"
          -P ${pf3d_cli_runner})
    endforeach()
    foreach(pf3d_ms_value -1 0 1 64)
      add_test(NAME pf3d_measure_shards_cli_valid_${pf3d_ms_value}
        COMMAND ${CMAKE_COMMAND}
          -DEXE=$<TARGET_FILE:cell_gh200_3d>
          "-DCASE_ARGS=--measure-shards;${pf3d_ms_value};--memory-fraction;0.1"
          -DEXPECT_CODE=2
          "-DEXPECT_LINE=[3d] --memory-fraction expects a finite number in [0.5, 0.99], got '0.1'"
          -P ${pf3d_cli_runner})
    endforeach()

    if(PF_ENABLE_CHECKPOINT_IO_TESTS)
      add_executable(pf3d_measure_shards_checkpoint_test
        tests/3d/measure_shards_checkpoint_test.cu)
      target_link_libraries(pf3d_measure_shards_checkpoint_test
        PRIVATE pf3d_core)
      target_compile_options(pf3d_measure_shards_checkpoint_test PRIVATE
        $<$<COMPILE_LANGUAGE:CUDA>:${PF_CUDA_FLAGS}>)
      set_target_properties(pf3d_measure_shards_checkpoint_test PROPERTIES
        CUDA_RUNTIME_LIBRARY ${PF_CUDA_RUNTIME})
      pf_enable_warnings(pf3d_measure_shards_checkpoint_test)
      add_test(NAME pf3d_measure_shards_checkpoint
        COMMAND pf3d_measure_shards_checkpoint_test)
      set_tests_properties(pf3d_measure_shards_checkpoint PROPERTIES
        SKIP_RETURN_CODE 77)
    endif()
  endif()

  if(PF_ENABLE_GPU_TESTS)
    add_test(NAME gh200_smoke
      COMMAND ${Python3_EXECUTABLE}
        ${CMAKE_CURRENT_SOURCE_DIR}/tests/run_gpu_smoke.py
        --executable $<TARGET_FILE:cell_gh200>)
    set_tests_properties(gh200_smoke PROPERTIES TIMEOUT 180)
    if(PF_BUILD_3D)
      add_test(NAME gh200_3d_smoke
        COMMAND ${Python3_EXECUTABLE}
          ${CMAKE_CURRENT_SOURCE_DIR}/examples/3d/run_example.py
          --executable $<TARGET_FILE:cell_gh200_3d>
          --output-dir ${CMAKE_CURRENT_BINARY_DIR}/gh200-3d-smoke-output)
      set_tests_properties(gh200_3d_smoke PROPERTIES TIMEOUT 300)
    endif()
  endif()
endif()

install(TARGETS cell_gh200 RUNTIME DESTINATION bin)
if(TARGET cell_gh200_3d)
  install(TARGETS cell_gh200_3d RUNTIME DESTINATION bin)
endif()
if(TARGET palmieri_centres)
  install(TARGETS palmieri_centres RUNTIME DESTINATION bin)
endif()
if(TARGET palmieri_centres_3d)
  install(TARGETS palmieri_centres_3d RUNTIME DESTINATION bin)
endif()
if(EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/tools/relaxed_2d_to_slab_centres.py)
  install(PROGRAMS tools/relaxed_2d_to_slab_centres.py DESTINATION bin)
endif()
install(PROGRAMS tools/render_pf3d_slab.py DESTINATION bin)
install(FILES tools/pf3d_checkpoint_header.py DESTINATION bin)
install(FILES
  README.md
  CITATION.cff
  LICENSE.txt
  COPYING
  DESTINATION share/doc/phase_field_gh200)
install(DIRECTORY docs/
  DESTINATION share/doc/phase_field_gh200/docs
  FILES_MATCHING PATTERN "*.md")
install(DIRECTORY examples/
  DESTINATION share/doc/phase_field_gh200/examples
  FILES_MATCHING
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*.py"
    PATTERN "*.args")
install(FILES
  tools/render_pf3d_slab.py
  tools/pf3d_checkpoint_header.py
  tools/relaxed_2d_to_slab_centres.py
  tools/requirements-visualization.txt
  DESTINATION share/doc/phase_field_gh200/tools)

message(STATUS "phase_field_gh200: CUDA arch ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "phase_field_gh200: 2D support layout 288/224/286")
message(STATUS
  "phase_field_gh200: support-edge counter ${PF_DIAGNOSTIC_ALARMS}")
message(STATUS "phase_field_gh200: 3D solver ${PF_BUILD_3D}")
