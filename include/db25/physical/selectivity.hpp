#pragma once
// Estimating what fraction of rows a predicate keeps, from column statistics.
//
// Separated from cost.hpp because it is a different kind of thing: operator_rows
// is ARITHMETIC over numbers it is handed, and must stay that way for the "one
// cost model" invariant (D3) to mean anything. Deciding what number to hand it
// for a filter means walking an expression tree and consulting a catalog, which
// is an ESTIMATION. Keeping the two apart is what lets the estimator be tested
// on predicates with no plan in sight, and lets the arithmetic be tested with no
// predicate in sight.
#include "db25/physical/statistics.hpp"
#include "db25/physical/physical_plan.hpp"

#include "db25/plan/expr_ir.hpp"

#include <span>

namespace db25::physical {

struct CardinalityModel;

// The fallback selectivities, used where statistics cannot answer. They are
// SEPARATED BY PREDICATE CLASS rather than being one number, because the classes
// are not equally selective and pretending otherwise is the error this whole
// increment is about. The values are the textbook ones (Selinger et al., and the
// same ones PostgreSQL ships) and they are guesses, which is why they are here,
// named, rather than inline.
struct SelectivityDefaults {
    // `col = <unknown>`: without an NDV this is the one the literature pins
    // lowest, because an equality against an unseen value usually matches little.
    double equality = 0.005;
    // `col < / > <unknown>`: a third. High, and deliberately: an inequality that
    // cannot be estimated is more likely to pass most rows than to pass few, and
    // UNDER-estimating a filter is what makes a planner put a huge relation on
    // the build side of a hash join.
    double inequality = 0.3333;
    // Anything else - a LIKE, a function call, a comparison between two columns.
    // This is the model's old flat `filter_selectivity`, now confined to the case
    // it was always really standing in for: "no idea".
    double unknown = 0.1;
};

// What fraction of its input rows `pred` keeps, in [0, 1].
//
// `stats` may be empty, in which case every answer comes from `defaults` and the
// result is exactly what the flat-constant model produced - which is the
// property that lets this land without moving a single golden plan until
// somebody actually seeds statistics.
//
// A null `pred` keeps everything: a Filter with no predicate filters nothing,
// and answering 0.1 to that would be inventing a restriction the query did not
// write.
[[nodiscard]] double selectivity_of(const plan::Expr* pred, const StatisticsCatalog& stats,
                                    const SelectivityDefaults& defaults);

// How many distinct groups `keys` produce over `input_rows` input rows.
//
// Returns 0 when it cannot say, so the caller can fall back rather than mistake
// a refusal for an estimate. With statistics it is the product of the keys' NDVs
// CAPPED at the input cardinality - the cap is not a safety rail, it is the
// statement that grouping cannot produce more groups than it has rows, which an
// uncapped product of independent NDVs will happily claim.
[[nodiscard]] double distinct_groups(std::span<const plan::Expr* const> keys,
                                     double input_rows, const StatisticsCatalog& stats);

// The distinct counts of a join's KEY columns, looked up through the input
// schemas. A HashKey is a positional index into `left` / `right`, and a
// ColumnSchema carries the catalog ids - so this is the one place the physical
// plan's positional key addressing is turned back into a statistics key.
//
// Writes 0 for a side it cannot resolve, which is the caller's signal to fall
// back to the containment bound. A MULTI-key join resolves only its first key:
// the others narrow through `join_selectivity` as before, because combining
// several keys' distinct counts correctly needs their joint distribution and
// multiplying them as though independent is how a planner convinces itself a
// two-column join returns nothing.
void join_key_ndv(const HashKeyVec& keys, const Schema& left, const Schema& right,
                  const StatisticsCatalog& stats, double& left_ndv, double& right_ndv);

}  // namespace db25::physical
