#pragma once
// Structural identity for memo groups (unit 2.4).
//
// Two logical subtrees that are STRUCTURALLY identical produce the same physical
// alternatives at the same cost, so they should share one memo group rather than
// be explored and optimized twice. Establishing that identity needs an exact
// comparison, not a hash: a hash collision that merged two DIFFERENT subtrees
// would silently plan one query as another. So hashing is used only to bucket
// candidates, and every apparent match is verified field by field.
//
// The comparison is deliberately CONSERVATIVE. plan::Expr carries a large payload
// - window specs, aggregate FILTER and ORDER BY, cast type modifiers, an owned
// subquery plan - and an equality that forgot one field would merge expressions
// that are not equivalent. Rather than track that surface, only the expression
// kinds whose payload is compared EXHAUSTIVELY here are eligible to match;
// everything else is treated as equal to nothing, including itself. A subtree
// containing one is simply never shared. That costs a missed dedup, which is
// free; the alternative costs a wrong plan.
#include "db25/plan/expr_ir.hpp"
#include "db25/plan/logical_plan.hpp"

#include <cstdint>
#include <string>
#include <vector>

namespace db25::physical {

// Is every node of `e` an expression kind this file compares exhaustively?
// Null counts as comparable (it is the absent payload).
[[nodiscard]] bool expr_is_comparable(const plan::Expr* e) noexcept;

// Exact structural equality over the comparable kinds. Returns FALSE whenever
// either side contains a kind outside that set, so an uncomparable expression
// never matches - not even itself.
[[nodiscard]] bool expr_structurally_equal(const plan::Expr* a, const plan::Expr* b) noexcept;

// A bucketing hash consistent with expr_structurally_equal: equal expressions
// hash equally. Unequal expressions MAY collide, which is why every match is
// verified.
[[nodiscard]] std::uint64_t expr_structural_hash(const plan::Expr* e) noexcept;

[[nodiscard]] bool schema_equal(const plan::Schema& a, const plan::Schema& b) noexcept;
[[nodiscard]] std::uint64_t schema_hash(const plan::Schema& s) noexcept;

}  // namespace db25::physical
