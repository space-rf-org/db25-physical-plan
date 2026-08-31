// Placeholder translation unit for db25-physical-plan.
//
// Unit 0.1 establishes the repo, build, CI, and the db25-logical-plan link.
// The real physical planner - IR, Cascades memo, cost model, and lowering -
// lands in the Increment 0 units that follow (see the design doc in the
// umbrella: docs/design/physical-planner.md). This TU exists only so the
// library target is non-empty until then.
namespace db25::physical {
// A trivial, ABI-stable anchor so the static library is non-empty on every
// platform's archiver (some warn on an empty .a).
int abi_anchor() { return 0; }
}  // namespace db25::physical
