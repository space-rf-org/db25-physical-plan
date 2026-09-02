#pragma once
// Increment 5.1: the plan cache, and the key that makes one safe.
//
// Planning the same query twice is waste, and the same query arrives constantly:
// an application issues a handful of statement shapes over and over. A cache
// turns T6 into a lookup for every repeat.
//
// TWO THINGS MAKE THIS HARDER THAN A HASH MAP.
//
// FIRST, OWNERSHIP. A physical plan BORROWS its expressions from the logical
// plan (see physical_plan.hpp) - it selects algorithms, it does not rewrite
// predicates. So a cache that stored only the physical plan would hand back a
// plan pointing at a logical plan the caller had since destroyed. The cache
// therefore owns BOTH, and hands out a reference into itself; an entry lives as
// long as the cache does. That is not an implementation detail to tidy away
// later, it is the borrowing contract followed to its conclusion.
//
// SECOND, WHAT COUNTS AS "THE SAME QUERY". Not the SQL text - two spellings of
// one query should share a plan, and one spelling under two catalogs must not.
// The key is the logical plan's STRUCTURE, plus every declared input that could
// change what the planner chooses: the calibration coefficients, the cardinality
// model, the storage and distribution catalogs, the required output properties,
// and the search flags. A key that omitted one of those would return a plan
// chosen under a different cost model and call it a hit.
//
// PARAMETERS ARE ALREADY SEPARATE, and this is the part that needs no machinery.
// A prepared statement's values are ExprKind::Parameter nodes, not literals - the
// values are not in the plan at all, they arrive at execution. So a parameterized
// query's key is naturally independent of what is bound to it, and its plan is
// reused across executions. A LITERAL, by contrast, is part of the key and must
// be: the physical plan RENDERS it, so reusing a plan built for `x > 10` on a
// query saying `x > 20` would emit a plan that filters on the wrong constant.
// Abstracting literals is what a system does when its plans carry parameter slots
// instead; DB25's IR already distinguishes the two, so nothing has to be
// abstracted and nothing can be abstracted wrongly.
//
// WHICH FIELDS ARE IN THE KEY is not a judgement call made once. Every member of
// plan::Expr, plan::LogicalNode, plan::ColumnSchema and the nested IR structs is
// swept in tests/test_plan_cache.cpp, and the sweep reads the upstream headers to
// check its own coverage - so a field added upstream fails the build's tests
// until somebody decides whether it belongs in the key. There is exactly ONE
// deliberate exclusion, Expr::source: it is a borrowed pointer into the AST, so
// two independently parsed copies of the same query carry different values and
// keying it would turn every lookup into a miss. The consequence is stated rather
// than hidden - a hit hands back expressions whose `source` points at the AST of
// the query that was planned FIRST. That is fine for a diagnostic echo and would
// not be fine for anything semantic, which is why nothing semantic reads it.
#include "db25/physical/lowering.hpp"
#include "db25/physical/physical_plan.hpp"

#include "db25/plan/logical_plan.hpp"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <vector>

namespace db25::physical {

// What makes two planning requests the same request.
//
// `structure` digests the logical plan; `inputs` digests everything else the
// planner reads. Two digests rather than one because they answer different
// questions and a diagnostic wants to know which of them moved.
struct PlanCacheKey {
    std::uint64_t structure = 0;
    std::uint64_t inputs = 0;

    [[nodiscard]] bool operator==(const PlanCacheKey& o) const noexcept {
        return structure == o.structure && inputs == o.inputs;
    }
};

// The key for lowering `root` under `ctx`.
//
// A DIGEST, not an exact comparison - and unlike the memo's group key, which
// verifies every apparent match field by field, this one cannot: the thing it
// would have to compare against is a logical plan the cache would then also have
// to keep in a comparable form. So a collision here would return the wrong plan,
// and the digest is built to make that vanishingly unlikely rather than merely
// unlikely: every field of every node, every expression, every schema column, and
// every calibration coefficient goes in, positionally.
//
// That is a real difference in kind from the memo's key and it is stated rather
// than glossed: the memo trades a hash for a verification because it can, and the
// cache cannot.
[[nodiscard]] PlanCacheKey plan_cache_key(const plan::LogicalNode& root,
                                          const LoweringContext& ctx);

// A plan the cache owns, and the query it was built from.
//
// `logical` is here because `plan` borrows from it. Nothing outside the cache
// owns either, and a reference handed out by `get` is valid until the entry is
// evicted or the cache dies.
struct CachedPlan {
    plan::LogicalNodePtr logical;
    LoweringResult result;
};

class PlanCache {
public:
    // The plan for `key`, or null. A HIT returns a reference into the cache; the
    // caller must not outlive it.
    [[nodiscard]] const CachedPlan* get(const PlanCacheKey& key) const;

    // Take ownership of a logical plan and the physical plan lowered from it.
    // Both, together, because the second borrows from the first.
    void put(PlanCacheKey key, plan::LogicalNodePtr logical, LoweringResult result);

    [[nodiscard]] std::size_t size() const noexcept { return entries_.size(); }
    [[nodiscard]] std::size_t hits() const noexcept { return hits_; }
    [[nodiscard]] std::size_t misses() const noexcept { return misses_; }
    void clear();

private:
    struct KeyHash {
        [[nodiscard]] std::size_t operator()(const PlanCacheKey& k) const noexcept {
            return static_cast<std::size_t>(k.structure ^ (k.inputs * 0x9E3779B97F4A7C15ULL));
        }
    };
    std::unordered_map<PlanCacheKey, std::unique_ptr<CachedPlan>, KeyHash> entries_;
    mutable std::size_t hits_ = 0;
    mutable std::size_t misses_ = 0;
};

}  // namespace db25::physical
