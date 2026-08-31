# db25-physical-plan

The DB25 **physical planner**: a spec-driven, Cascades-based `logical → physical`
planner, deliberately decoupled from any execution engine.

It consumes the logical plan produced by
[`db25-logical-plan`](https://github.com/space-rf-org/db25-logical-plan) and emits
a physical plan (an executable operator DAG: access paths, join algorithms, data
layout, parallelism, storage substrate). It emits a plan; it never runs one.

The design — decisions D1–D10, the time envelope, and the increment plan — lives
in the umbrella repo at
[`docs/design/physical-planner.md`](https://github.com/space-rf-org/db25/blob/main/docs/design/physical-planner.md).
In one line: a pure, deterministic function of
`(logical_plan, catalog_stats, execution_capability_profile, runtime_profile?)`,
one physical IR and one cost model, spec expressed as an external versioned IDL,
native plan interchange (a validation-only Substrait export stays off the
execution critical path).

## Status

Increment 0 — building the seams-complete skeleton. This bootstrap establishes the
repo, build, CI, and the upstream link; the physical IR, memo, cost model, and
lowering land in the units that follow.

## Building

`db25-logical-plan` is vendored as a recursive git submodule:

```sh
git clone --recursive https://github.com/space-rf-org/db25-physical-plan
# or, after a plain clone:
git submodule update --init --recursive

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER=g++-14
cmake --build build -j"$(nproc)"
ctest --test-dir build --output-on-failure
```

Supported toolchain: GCC 14+ (the CI baseline), C++23, `-fno-exceptions`.
