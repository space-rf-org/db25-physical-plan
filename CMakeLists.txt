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
    src/sexpr_read.cpp
    src/spec.cpp
    src/cost.cpp
    src/properties.cpp
    src/lowering.cpp
    src/structural_key.cpp)
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

    add_executable(cost_tests tests/test_cost.cpp)
    target_link_libraries(cost_tests PRIVATE db25physicalplan)
    target_compile_options(cost_tests PRIVATE -fno-exceptions -Wall -Wextra)
    # The cost test loads the pinned-lab calibration profile from the source tree.
    target_compile_definitions(cost_tests PRIVATE
        DB25_PHYSICAL_SPEC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/spec")
    add_test(NAME cost_tests COMMAND cost_tests)

    add_executable(property_tests tests/test_properties.cpp)
    target_link_libraries(property_tests PRIVATE db25physicalplan)
    target_compile_options(property_tests PRIVATE -fno-exceptions -Wall -Wextra)
    add_test(NAME property_tests COMMAND property_tests)

    add_executable(lowering_tests tests/test_lowering.cpp)
    target_link_libraries(lowering_tests PRIVATE db25physicalplan)
    target_compile_options(lowering_tests PRIVATE -fno-exceptions -Wall -Wextra)
    target_compile_definitions(lowering_tests PRIVATE
        DB25_PHYSICAL_SPEC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/spec")
    add_test(NAME lowering_tests COMMAND lowering_tests)

    # The applicability sweep. Iterates kAllPhysicalOps, so an operator added
    # without a decision about its preconditions, its cost, its derivation or its
    # name fails HERE rather than in a query months later.
    add_executable(applicability_tests tests/test_applicability.cpp)
    target_link_libraries(applicability_tests PRIVATE db25physicalplan)
    target_compile_options(applicability_tests PRIVATE -fno-exceptions -Wall -Wextra)
    target_compile_definitions(applicability_tests PRIVATE
        DB25_PHYSICAL_SPEC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/spec")
    add_test(NAME applicability_tests COMMAND applicability_tests)

    # The allocation budget replaces global operator new to count, which does not
    # coexist with the sanitizers' own allocator interception - so it is built and
    # run only in an unsanitized build. The metric is deterministic, so one
    # configuration measuring it is enough.
    if(NOT CMAKE_CXX_FLAGS MATCHES "sanitize")
        add_executable(allocation_budget_tests tests/test_allocation_budget.cpp)
        target_link_libraries(allocation_budget_tests PRIVATE db25physicalplan)
        target_compile_options(allocation_budget_tests PRIVATE -fno-exceptions -Wall -Wextra)
        target_compile_definitions(allocation_budget_tests PRIVATE
            DB25_PHYSICAL_SPEC_DIR="${CMAKE_CURRENT_SOURCE_DIR}/spec")
        add_test(NAME allocation_budget_tests COMMAND allocation_budget_tests)
    endif()
endif()
