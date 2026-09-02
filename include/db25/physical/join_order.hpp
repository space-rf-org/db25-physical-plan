#pragma once
// Join reordering: the region analysis and the index bookkeeping it needs.
//
// (A join B) join C and A join (B join C) return the same rows and can cost very
// differently, so which one the planner builds should be a cost decision. Making
// it one runs into a property of DB25's IR: a ColumnRef is a POSITIONAL
// `input_index` into `child0.output ++ child1.output`, so changing the shape of a
// join tree changes what every index above it means.
//
// The way through is to translate indices at EXTRACTION rather than during search.
// A region is analysed once into leaves and conjuncts, both addressed in one
// canonical numbering; the search then reasons about SUBSETS of leaves and never
// touches an expression; and only the winning plan has its residual predicates
// rebuilt with translated indices, into an arena the result owns. A search that
// considers a thousand orders clones nothing; a plan that uses one clones the
// conjuncts of that one.
//
// Only INNER and CROSS joins are reordered. An outer join is not associative -
// moving it changes which rows are null-extended - and a LATERAL join's right
// input depends on its left, so it cannot be moved at all. Either one ENDS the
// region rather than being reordered within it.
#include "db25/physical/physical_plan.hpp"
#include "db25/plan/logical_plan.hpp"

#include <cstdint>
#include <deque>
#include <vector>

namespace db25::physical {

// One conjunct of a region's join predicates.
//
// `base` is what makes this work without rewriting anything. A conjunct sitting on
// a join partway down the region has indices relative to THAT join's inputs, not
// to the region; and because an in-order traversal gives every subtree a
// contiguous range of leaves, the difference is a single offset. So the region
// position of a column is `base + expr->input_index`, and the expression itself is
// left exactly as the logical plan wrote it.
struct RegionConjunct {
    const plan::Expr* expr = nullptr;
    std::uint32_t base = 0;
    std::uint64_t leaf_mask = 0;  // which leaves it references
};

// A maximal connected subtree of INNER / CROSS joins, flattened.
//
// `leaves` are in the order an in-order traversal visits them, which is also the
// order their columns appear in the region's output - so the FULL leaf set's
// canonical column order is the region's original one, and nothing above the
// region has to change however the inside is rearranged.
struct JoinRegion {
    std::vector<const plan::LogicalNode*> leaves;
    std::vector<std::uint32_t> leaf_offset;  // canonical start column of each leaf
    std::vector<std::uint32_t> leaf_width;   // how many columns each leaf produces
    std::vector<RegionConjunct> conjuncts;
    const plan::Schema* output = nullptr;    // the region root's output schema

    [[nodiscard]] std::size_t leaf_count() const { return leaves.size(); }
    [[nodiscard]] std::uint32_t width() const {
        return leaf_offset.empty() ? 0 : leaf_offset.back() + leaf_width.back();
    }
};

// Collect the region rooted at `n`, which must be an INNER or CROSS join. Returns
// false when `n` roots no region worth reordering - fewer than three leaves, or
// more than 64 (the leaf-set bitmask's width, and far past any budget).
[[nodiscard]] bool collect_join_region(const plan::LogicalNode& n, JoinRegion& out);

// Which columns of the region a subset of leaves produces, in canonical order, and
// where each region column lands within it. `slot[i]` is the position of region
// column i inside the subset's own output, or kNoSlot when the subset does not
// produce it.
inline constexpr std::uint32_t kNoSlot = 0xFFFFFFFFu;
[[nodiscard]] std::vector<std::uint32_t> subset_slots(const JoinRegion& region,
                                                      std::uint64_t subset);

// Storage the extracted plan owns: schemas synthesized for subsets that no logical
// node produced, and predicates rebuilt with translated indices. A deque because
// the memo and the plan hold POINTERS into it and it grows while they do.
struct LoweringArena {
    // A deque: the memo stores POINTERS to these schemas and keeps them while more
    // are appended, which a vector's reallocation would invalidate.
    std::deque<plan::Schema> schemas;
    // Each entry OWNS a cloned conjunct and its whole subtree, because
    // plan::Expr's children are unique_ptr and cannot be owned twice. Only the
    // root pointer is handed out, and a vector never moves the pointed-to Expr.
    std::vector<plan::ExprPtr> exprs;
};

// Rebuild `c` with every ColumnRef's index translated from region numbering into
// the numbering of a join over (left, right), and return the clone. Nodes are
// appended to `arena`, which must outlive the returned expression.
//
// This is the ONLY place the physical planner constructs an expression, and it
// does not change what any expression computes - it re-addresses columns whose
// position moved because the planner chose a different join order.
[[nodiscard]] const plan::Expr* translate_conjunct(const RegionConjunct& c,
                                                   const JoinRegion& region,
                                                   std::uint64_t left, std::uint64_t right,
                                                   LoweringArena& arena);

}  // namespace db25::physical
