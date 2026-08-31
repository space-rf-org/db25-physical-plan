// Unit 0.1 smoke test: prove the repo builds, links db25physicalplan, and that
// the vendored db25-logical-plan is reachable through it (the include + link
// path). A real test harness and physical-plan goldens arrive in later
// Increment 0 units.
#include "db25/physical/version.hpp"
// expr_ir.hpp completes db25::plan::Expr (it forward-declares in
// logical_plan.hpp), which LogicalNode's owning unique_ptr<Expr> members need
// before the header can be compiled - the intended consumer contract, the same
// one db25-logical-plan's own tests follow (via binder.hpp).
#include "db25/plan/expr_ir.hpp"
#include "db25/plan/logical_plan.hpp"  // from the vendored db25-logical-plan

#include <cstdio>

int main() {
    // Touch an upstream enumerator so the include + transitive link is genuinely
    // exercised (not dead-stripped), without asserting a brittle ordinal.
    std::printf("db25-physical-plan smoke ok (LogicalOp::Scan=%d, v%d.%d.%d)\n",
                static_cast<int>(db25::plan::LogicalOp::Scan),
                db25::physical::kVersionMajor, db25::physical::kVersionMinor,
                db25::physical::kVersionPatch);
    return 0;
}
