#pragma once
// Canonical s-expression rendering of a physical plan.
//
// One-way and deterministic: the same plan always renders to the same text, and
// any structural change to the plan changes the text (the falsifiability
// contract the DB25 staged goldens rely on). This is the golden form used by the
// physical-plan tests here and, later, by the umbrella staged harness (Unit 0.7,
// which also adds the parse-back / plan-injection round-trip, exactly as the
// logical-plan stage does).
#include "db25/physical/physical_plan.hpp"

#include <string>

namespace db25::physical {

[[nodiscard]] std::string physical_to_sexpr(const PhysicalNode& node);

}  // namespace db25::physical
