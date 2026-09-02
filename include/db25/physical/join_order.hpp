#pragma once
// Join reordering: the region analysis and the index arithmetic it rests on.
//
// (A join B) join C and A join (B join C) return the same rows and can cost very
// differently, so which one the planner builds should be a cost decision. Making
// it one runs into a property of DB25's IR: a ColumnRef is a POSITIONAL
// `input_index` into `child0.output ++ child1.output`, so changing the shape of a
// join tree changes what every index above it means.
//
// The way through is an observation about traversal order. An in-order walk gives
// every subtree a CONTIGUOUS range of leaves, and a region's output columns are
// its leaves' columns concatenated in that same order. So if the planner
// re-associates a join tree without reordering its leaves - (A B) C into A (B C) -
// then every subtree still covers a contiguous range of the region's columns, in
// the region's own order. A join over ranges [i,k) and [k,j) produces exactly the
// region's columns [off_i, off_j), already in the right order. No permutation is
// needed anywhere, and translating a column index is one subtraction.
//
// WHAT THIS DOES AND DOES NOT EXPLORE. It re-associates; it does not permute. For
// a three-way join it finds both trees, which is all of them. For four or more it
// finds the re-associations that keep leaf order, not the orderings that swap
// distant leaves - so `A B C D` will consider (A B)(C D) but not (A C)(B D).
// Permuting leaves changes the output column order and therefore needs each join
// to carry an output permutation, or a Project above it to restore the order; the
// arena below is what that would be built on. Deliberately not in this unit: it is
// a larger change and this one is provably free of permutation bugs because it
// contains no permutations.
//
// Only INNER and CROSS joins are re-associated. An outer join is not associative -
// moving it changes which rows are null-extended - and a LATERAL join's right
// input reads its left, so it cannot move at all. Either one ENDS the region and
// becomes one of its leaves, which is how reordering stays sound around them
// rather than refusing queries that contain them.
#include "db25/physical/physical_plan.hpp"
#include "db25/plan/logical_plan.hpp"

#include <cstdint>
#include <memory>
#include <vector>

namespace db25::physical {

// One conjunct of a region's join predicates.
//
// `base` is what lets the search reason about a conjunct without rewriting it. A
// conjunct sitting on a join partway down the region has indices relative to THAT
// join's inputs; because every subtree covers a contiguous leaf range, the
// difference from region numbering is a single offset. So a column's region
// position is `base + expr->input_index`, and the expression is left exactly as
// the logical plan wrote it.
struct RegionConjunct {
    const plan::Expr* expr = nullptr;
    std::uint32_t base = 0;
    std::uint64_t leaf_mask = 0;  // which leaves it references
};

// A maximal connected subtree of INNER / CROSS joins, flattened. `leaves` are in
// in-order traversal order, which is also the order their columns appear in the
// region's output.
struct JoinRegion {
    std::vector<const plan::LogicalNode*> leaves;
    std::vector<std::uint32_t> leaf_offset;  // region column where each leaf starts
    std::vector<std::uint32_t> leaf_width;
    std::vector<RegionConjunct> conjuncts;
    const plan::Schema* output = nullptr;

    [[nodiscard]] std::size_t leaf_count() const { return leaves.size(); }
    // Region column just past leaf `last - 1`.
    [[nodiscard]] std::uint32_t end_of(std::size_t last) const {
        return last == 0 ? 0 : leaf_offset[last - 1] + leaf_width[last - 1];
    }
};

// Collect the region rooted at `n`, which must be an INNER or CROSS join. Returns
// false when there is nothing to re-associate (fewer than three leaves - the
// two-way case is what build-side selection already covers), when a conjunct
// cannot be translated, or when the region exceeds 64 leaves.
[[nodiscard]] bool collect_join_region(const plan::LogicalNode& n, JoinRegion& out);

// All leaves in [first, last) as one mask.
[[nodiscard]] std::uint64_t range_mask(std::size_t first, std::size_t last);

// Is `c` placed on the join over [first, split) and [split, last)? A conjunct
// belongs to the LOWEST join whose subtree contains every leaf it references, so
// it is placed here when it fits in this range and cannot be pushed into either
// child - either because it spans the split, or because the child that contains
// it is a single leaf and so is not a join at all.
[[nodiscard]] bool placed_here(const RegionConjunct& c, std::size_t first,
                               std::size_t split, std::size_t last);

// Storage the extracted plan owns: schemas synthesized for ranges no logical node
// produced, and predicates rebuilt with translated indices.
struct LoweringArena {
    // A vector of unique_ptr rather than a deque, even though the memo holds
    // POINTERS to these schemas while more are appended. A deque would keep them
    // stable directly - but an empty libstdc++ deque allocates its map and a node
    // at CONSTRUCTION, and this arena is empty on every query that reorders
    // nothing, which is most of them. An empty vector allocates nothing, and a
    // unique_ptr keeps the pointed-to schema still when the vector grows. Measured
    // as two allocations per lower() on the budget query, which reorders nothing.
    std::vector<std::unique_ptr<plan::Schema>> schemas;
    // Each entry OWNS a cloned conjunct and its whole subtree, because
    // plan::Expr's children are unique_ptr and cannot be owned twice. Only the
    // root pointer is handed out, and a vector never moves the pointed-to Expr.
    std::vector<plan::ExprPtr> exprs;
};

// The conjunct as the join over [first, last) sees it: every ColumnRef's index
// shifted from region numbering into that join's own input numbering.
//
// Returns the BORROWED original when no shift is needed, which is the common case
// - a conjunct already sitting on a join with this left edge needs no translation
// at all, so a plan that keeps the original association clones nothing and costs
// no allocations. Otherwise a clone is appended to `arena`, which must outlive it.
//
// This is the only place the physical planner constructs an expression, and it
// does not change what any expression computes: it re-addresses columns whose
// position moved because the planner chose a different association.
[[nodiscard]] const plan::Expr* translate_conjunct(const RegionConjunct& c,
                                                   const JoinRegion& region,
                                                   std::size_t first,
                                                   LoweringArena& arena);

}  // namespace db25::physical
