cmake_minimum_required(VERSION 3.20)
project(DB25PhysicalPlan VERSION 0.1.0 LANGUAGES CXX)

# Match the rest of the DB25 stack.
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Supported toolchain: GCC 14+ (the CI baseline). Surface an older GCC as a
# clear, actionable message rather than a deep C++23 template error.
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14)
    message(WARNING
        "DB25 targets GCC 14+ (the supported, CI-tested toolchain); "
        "detected GCC ${CMAKE_CXX_COMPILER_VERSION}, which is older and unsupported.")
endif()

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# Namespaced so the upstream add_subdirectory (which does
# `set(BUILD_TESTS OFF CACHE BOOL "" FORCE)` to silence the parser's own tests)
# cannot clobber our choice to build the physical-plan tests.
option(DB25_PHYSICAL_BUILD_TESTS "Build db25-physical-plan tests" ON)

# ---------------------------------------------------------------------------
# Upstream: db25-logical-plan (git submodule). Consumed exactly as the umbrella
# consumes it - add_subdirectory builds the db25logicalplan target, whose
# include dirs are PUBLIC and which links db25parser transitively, so linking it
# alone brings in db25/plan, db25/semantic, db25/ast, db25/parser + tokenizer.
#
# Guarded on the target: in a super-build (the umbrella) that has already added
# db25-logical-plan, reuse that target instead of adding a second, colliding copy.
# ---------------------------------------------------------------------------
if(NOT TARGET db25logicalplan)
    if(NOT EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/external/db25-logical-plan/CMakeLists.txt)
        message(FATAL_ERROR
            "external/db25-logical-plan is empty. Run:\n"
            "  git submodule update --init --recursive")
    endif()
    add_subdirectory(external/db25-logical-plan)
endif()

# ---------------------------------------------------------------------------
# The physical planner library. The physical IR + Cascades memo + s-expr
# rendering land here (Unit 0.2); the cost model, spec IDL, and lowering follow
# in later units.
# ---------------------------------------------------------------------------
add_library(db25physicalplan STATIC
    src/physical_plan.cpp
    src/memo.cpp
    src/sexpr.cpp
    src/spec.cpp)
target_include_directories(db25physicalplan PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(db25physicalplan PUBLIC db25logicalplan)
target_compile_options(db25physicalplan PRIVATE -fno-exceptions -Wall -Wextra)

if(DB25_PHYSICAL_BUILD_TESTS)
    enable_testing()
    add_executable(physical_tests tests/test_smoke.cpp)
    target_link_libraries(physical_tests PRIVATE db25physicalplan)
    target_compile_options(physical_tests PRIVATE -fno-exceptions -Wall -Wextra)
    add_test(NAME physical_tests COMMAND physical_tests)

    add_executable(physical_ir_tests tests/test_physical_ir.cpp)
    target_link_libraries(physical_ir_tests PRIVATE db25physicalplan)
    target_compile_options(physical_ir_tests PRIVATE -fno-exceptions -Wall -Wextra)
    add_test(NAME physical_ir_tests COMMAND physical_ir_tests)

    add_executable(spec_tests tests/test_spec.cpp)
    target_link_libraries(spec_tests PRIVATE db25physicalplan)
    target_compile_options(spec_tests PRIVATE -fno-exceptions -Wall -Wextra)
    # The conformance test loads the shipped spec from the source tree.
    target_compile_definitions(spec_tests PRIVATE
        DB25_PHYSICAL_SPEC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/spec")
    add_test(NAME spec_tests COMMAND spec_tests)
endif()
