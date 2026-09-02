#include "db25/physical/properties.hpp"

#include <utility>

#include <cstddef>
#include <span>
#include <utility>

namespace db25::physical {
namespace {

// Exact identity of two keys, every field. This is what memoization needs: a
// requirement is the KEY a group's winners are stored under, so two requirements
// that are not the same thing must not compare equal or one would silently
// answer for the other.
bool identical_key(const SortKey& a, const SortKey& b) {
    return a.column == b.column && a.descending == b.descending &&
           a.nulls_specified == b.nulls_specified &&
           (!a.nulls_specified || a.nulls_first == b.nulls_first);
}

// Does a provided key SATISFY a required one? Asymmetric, like Freshness::Any
// and StorageFormat::Any: a requirement that did not specify where NULLs go is
// satisfied by any placement, while one that did needs exactly that placement.
//
// Getting this wrong in either direction is a real bug. Ignoring the nulls
// ordering entirely (as this did while SortKey had no such field) lets a stream
// sorted NULLS FIRST answer a query that asked for NULLS LAST. Requiring an
// exact match unconditionally makes every enforcer-created key - which has no
// opinion - fail to be satisfied by a scan that happens to have one.
bool satisfies_key(const SortKey& provided, const SortKey& required) {
    if (provided.column != required.column) return false;
    if (provided.descending != required.descending) return false;
    if (!required.nulls_specified) return true;
    return provided.nulls_specified && provided.nulls_first == required.nulls_first;
}

// Is `required` a prefix of `provided`? (provided sorted at least as specifically)
bool sort_prefix(const std::vector<SortKey>& provided, const std::vector<SortKey>& required) {
    if (required.size() > provided.size()) return false;
    for (std::size_t i = 0; i < required.size(); ++i) {
        if (!satisfies_key(provided[i], required[i])) return false;
    }
    return true;
}

}  // namespace

bool operator==(const SortKey& a, const SortKey& b) noexcept {
    return identical_key(a, b);
}

bool operator==(const PhysicalProperties& a, const PhysicalProperties& b) noexcept {
    if (a.format != b.format || a.freshness != b.freshness) return false;
    if (a.sort.size() != b.sort.size()) return false;
    for (std::size_t i = 0; i < a.sort.size(); ++i) {
        if (!identical_key(a.sort[i], b.sort[i])) return false;
    }
    return true;
}

bool satisfies(const PhysicalProperties& provided, const PhysicalProperties& required) {
    // Correctness first: stale data cannot answer a query that must see the
    // latest writes, and no enforcer exists that would change that.
    if (required.freshness == Freshness::Fresh && provided.freshness != Freshness::Fresh) {
        return false;
    }
    if (required.format != StorageFormat::Any && required.format != provided.format) {
        return false;
    }
    if (!distribution_satisfies(provided.distribution, required.distribution)) return false;
    return sort_prefix(provided.sort, required.sort);
}

PhysicalProperties derive_op(PhysicalOp op, const HashKeyVec& keys,
                             StorageFormat scan_format,
                             std::span<const PhysicalProperties> input_props,
                             Freshness scan_freshness, std::span<const SortKey> sort_keys,
                             Distribution scan_distribution, std::uint32_t group_key_count) {
    const auto in = [&](std::size_t i) {
        return i < input_props.size() ? input_props[i] : PhysicalProperties{};
    };
    // Where the OUTPUT rows live. Derived alongside everything else and then
    // applied at the end, so each case below can stay about the one thing it is
    // really deciding.
    //
    // The DEFAULT is the first input's distribution, which is right for every
    // operator that reshapes rows without moving them - a filter, a projection, a
    // limit, a sort within each node. The cases that differ say so.
    const auto row_preserving = [&]() { return in(0).distribution; };
    Distribution dist = input_props.empty() ? Distribution{DistributionKind::Single, {}}
                                            : row_preserving();
    switch (op) {
        case PhysicalOp::SeqScan:
            dist = scan_distribution;
            break;
        case PhysicalOp::ValuesScan:
        case PhysicalOp::WorkingTableScan:
            // Literal rows, and the working table of a recursion this plan is
            // running: both live where the plan runs.
            dist = Distribution{DistributionKind::Single, {}};
            break;
        case PhysicalOp::HashJoin:
        case PhysicalOp::MergeJoin:
        case PhysicalOp::NestedLoopJoin: {
            // A join's inputs arrive co-located on the keys (see
            // required_input_properties), so its output is partitioned on them
            // too - and the LEFT key indices are also output indices, because a
            // join's output is left ++ right.
            if (keys.empty()) {
                // A cross product has no key to partition on. Its inputs were
                // required Single, so its output is.
                dist = Distribution{DistributionKind::Single, {}};
            } else {
                Distribution d{DistributionKind::Hashed, {}};
                for (const HashKey& k : keys) d.keys.push_back(k.left_index);
                dist = d;
            }
            break;
        }
        case PhysicalOp::HashSemiJoin:
        case PhysicalOp::HashAntiJoin:
        case PhysicalOp::NestedLoopSemiJoin:
        case PhysicalOp::NestedLoopAntiJoin:
            // A subset of the LEFT input's rows, so however the left was
            // partitioned, the survivors still are.
            dist = in(0).distribution;
            break;
        case PhysicalOp::HashAggregate:
        case PhysicalOp::StreamingAggregate:
        case PhysicalOp::HashGroupingSets: {
            if (group_key_count == 0) {
                // A scalar aggregate is exactly one row, and one row is in one
                // place. Its input was required Single for the same reason: a
                // partial sum per node is not a sum.
                dist = Distribution{DistributionKind::Single, {}};
            } else {
                // The output is [keys..., aggregates...], so the grouping columns
                // ARE output columns 0..n-1, and the groups landed on the node
                // their key hashed to.
                Distribution d{DistributionKind::Hashed, {}};
                for (std::uint32_t i = 0; i < group_key_count; ++i) d.keys.push_back(i);
                dist = d;
            }
            break;
        }
        case PhysicalOp::HashDistinct:
        case PhysicalOp::StreamingDistinct: {
            // De-duplication is over every output column, and its input arrived
            // partitioned on them, so its output still is.
            Distribution d{DistributionKind::Hashed, {}};
            for (std::uint32_t i = 0; i < in(0).sort.size(); ++i) d.keys.push_back(i);
            if (d.keys.empty()) d = in(0).distribution;
            dist = d;
            break;
        }
        case PhysicalOp::UnionAll:
        case PhysicalOp::HashSetOp:
        case PhysicalOp::RecursiveFixpoint:
        case PhysicalOp::CreateTableAs:
        case PhysicalOp::Insert:
        case PhysicalOp::Update:
        case PhysicalOp::Delete:
            // All required Single of their inputs (see required_input_properties),
            // so all produce Single. Distributed set operations, distributed
            // recursion and distributed writes are each their own increment; the
            // requirement is what makes saying so here honest rather than
            // optimistic.
            dist = Distribution{DistributionKind::Single, {}};
            break;
        case PhysicalOp::Exchange:
            // Filled in by derive() from the node's own target, exactly as
            // FormatConvert's format is - an enforcer's effect is its definition.
            break;
        default:
            break;  // row-preserving: the default above
    }
    // Every `return` below builds the order / format / freshness triple this
    // operator provides; `with` stamps the distribution derived above onto it, so
    // no case has to remember to, and a new case cannot forget.
    const auto with = [&](PhysicalProperties p) {
        p.distribution = dist;
        return p;
    };
    // Staleness propagates upward and never washes out: anything derived from a
    // lagging copy is itself lagging.
    const auto combined_freshness = [&]() {
        Freshness f = Freshness::Fresh;
        for (const PhysicalProperties& p : input_props) {
            if (p.freshness == Freshness::Stale) f = Freshness::Stale;
        }
        return f;
    };
    switch (op) {
        case PhysicalOp::SeqScan:
            // A base scan produces the table's stored format, in no guaranteed
            // order, as fresh as the copy it reads.
            return with(PhysicalProperties{{}, scan_format, scan_freshness});
        case PhysicalOp::Filter:
        case PhysicalOp::Project:
            // Both drop or reshape columns but preserve order and format. (Project
            // preserves conservatively in Increment 1; a later increment tracks
            // whether the projection actually keeps the sort columns.)
            return with(in(0));
        case PhysicalOp::HashJoin:
            // Builds a hash table: output order is not guaranteed.
            return with(PhysicalProperties{{}, StorageFormat::Row, combined_freshness()});
        case PhysicalOp::NestedLoopJoin:
            // Emits pairs in left-outer-loop order. That IS the left input's
            // order, but claiming it would be claiming a property of a specific
            // execution strategy; until the engine guarantees it, derive nothing.
            return with(PhysicalProperties{{}, StorageFormat::Row, combined_freshness()});
        case PhysicalOp::MergeJoin: {
            // Consumes both inputs in key order and emits in that same order, so
            // the join keys' order survives - the property that can save a later
            // Sort and is exactly why it may win over a HashJoin.
            PhysicalProperties p;
            p.sort.reserve(keys.size());
            for (const HashKey& k : keys) p.sort.push_back(SortKey{k.left_index, false});
            p.format = StorageFormat::Row;
            p.freshness = combined_freshness();
            return with(p);
        }
        case PhysicalOp::Sort:
            // Establishes ITS order - the whole point of the operator - and
            // preserves the child's format and freshness.
            return with(PhysicalProperties{{sort_keys.begin(), sort_keys.end()}, in(0).format,
                                      in(0).freshness});
        case PhysicalOp::FormatConvert:
            // Changes the format to its target (filled in by the caller); preserves order.
            return with(PhysicalProperties{in(0).sort, StorageFormat::Any, in(0).freshness});
        case PhysicalOp::HashAggregate:
            // Builds a hash table on the grouping keys: the output comes out in
            // hash-bucket order, which is no order at all. Same as a HashJoin.
            return with(PhysicalProperties{{}, StorageFormat::Row, in(0).freshness});
        case PhysicalOp::StreamingAggregate:
            // Consumes its input in grouping-key order and emits one row per group
            // as each group closes, so that order SURVIVES - the property that can
            // save a later Sort and is exactly why it may win over hashing.
            // `sort_keys` is the grouping order here, on the same footing as it is
            // the established order for a Sort: in both cases it is the ordered key
            // set this operator works in.
            return with(PhysicalProperties{{sort_keys.begin(), sort_keys.end()}, StorageFormat::Row,
                                      in(0).freshness});
        case PhysicalOp::HashSemiJoin:
        case PhysicalOp::HashAntiJoin:
        case PhysicalOp::NestedLoopSemiJoin:
        case PhysicalOp::NestedLoopAntiJoin:
            // Emit a subset of the LEFT input's rows, unchanged. That subset IS in
            // the left input's order for every implementation here - but so is a
            // nested loop join's output, and this planner deliberately claims
            // neither, because both are properties of an execution strategy the
            // engine has not yet promised. Consistency matters more than the one
            // Sort it might save.
            return with(PhysicalProperties{{}, StorageFormat::Row, combined_freshness()});
        case PhysicalOp::HashGroupingSets:
            // One hash table per grouping set, emitted set by set: no order.
            return with(PhysicalProperties{{}, StorageFormat::Row, in(0).freshness});
        case PhysicalOp::HashDistinct:
            // A hash table on every output column: emits in bucket order.
            return with(PhysicalProperties{{}, StorageFormat::Row, in(0).freshness});
        case PhysicalOp::StreamingDistinct:
            // Collapses adjacent duplicates in one pass, so the input's order -
            // which it REQUIRED - survives.
            return with(in(0));
        case PhysicalOp::UnionAll:
        case PhysicalOp::HashSetOp: {
            // Two streams meeting produce no single order, even when both inputs
            // happened to be sorted: a concatenation interleaves nothing and a
            // hash probe emits in probe order at best. Claiming an order here
            // would be claiming a property of an execution strategy.
            Freshness f = Freshness::Fresh;
            for (const PhysicalProperties& p : input_props) {
                if (p.freshness == Freshness::Stale) f = Freshness::Stale;
            }
            return with(PhysicalProperties{{}, StorageFormat::Row, f});
        }
        case PhysicalOp::ValuesScan:
            // Literal rows: in the order written, but that is not an order over
            // any COLUMN, so there is no sort property to claim. Always fresh -
            // there is no stored copy that could lag.
            return with(PhysicalProperties{{}, StorageFormat::Row, Freshness::Fresh});
        case PhysicalOp::Window:
            // Appends columns to rows it emits in the order it received them, so
            // everything its input provided survives. Its input is REQUIRED sorted
            // by (partition ++ order), so that order is what survives - stating it
            // as `in(0)` rather than restating the keys keeps one source of truth.
            return with(in(0));
        case PhysicalOp::Limit:
            // Takes a contiguous window of its input's rows, so everything about
            // them is preserved: still in the same order, same format, same
            // freshness. What it does NOT preserve is WHICH rows - see
            // pushdown_requirement.
            return with(in(0));
        case PhysicalOp::RecursiveFixpoint:
            // The anchor's rows, then each iteration's, concatenated. Even if
            // every iteration emitted its rows in some order, the concatenation of
            // iterations is not sorted on anything - the second iteration's rows
            // follow the first's whatever their values. So: no order, from either
            // input, ever.
            return with(PhysicalProperties{{}, StorageFormat::Row, combined_freshness()});
        case PhysicalOp::WorkingTableScan:
            // Reads rows this same query produced an iteration ago. Always FRESH -
            // not as a convention but as a fact: they were computed here, and there
            // is no stored copy that could lag. No order: the fixpoint above
            // guarantees none, and this reads what it wrote.
            return with(PhysicalProperties{{}, StorageFormat::Row, Freshness::Fresh});
        case PhysicalOp::CreateTableAs:
            // Passes its input's rows through as it writes them, so their order and
            // freshness survive to whatever reads this node's output. The FORMAT is
            // the new table's, and a table this statement is creating is written in
            // the row format; a columnar copy is a storage decision made after the
            // table exists, not one this operator gets to make.
            return with(PhysicalProperties{in(0).sort, StorageFormat::Row, in(0).freshness});
        case PhysicalOp::Exchange:
            // Moves rows between nodes, and NO ORDER SURVIVES THAT. Rows arriving
            // from several senders interleave however the network delivers them,
            // and that is true of a gather as much as of a repartition - one
            // receiver merging many streams is still merging them. An
            // order-preserving gather is a real thing and a later operator; saying
            // this one preserves order would let a MergeJoin read an input it
            // believes is sorted and is not.
            //
            // The FORMAT and the FRESHNESS do survive: moving a row does not
            // rewrite it or make it staler. What the exchange ESTABLISHES is
            // applied by derive(), like FormatConvert's format.
            return with(PhysicalProperties{{}, in(0).format, in(0).freshness});
        case PhysicalOp::Insert:
        case PhysicalOp::Update:
        case PhysicalOp::Delete:
            // The AFFECTED rows, which a RETURNING clause projects. FRESH by
            // construction: these rows are what this statement just wrote, so no
            // stored copy stands between them and the reader. No order, because
            // the order rows are written in is the storage layer's business and
            // claiming it would be claiming a property of an execution strategy -
            // the same line drawn at the nested loop and the semi joins.
            return with(PhysicalProperties{{}, StorageFormat::Row, Freshness::Fresh});
    }
    return with(PhysicalProperties{});
}

bool is_applicable(PhysicalOp op, const HashKeyVec& keys, ast::JoinType join_kind,
                   GroupingSpec grouping, ast::SetOp set_op) {
    // A LATERAL join's right input is CORRELATED: its subtree reads the current
    // left row through an OuterRef, so it has no meaning evaluated on its own.
    // Only a nested-loop join re-evaluates the right input per left row; a hash
    // join builds its table from the right input once, standalone, and a merge
    // join scans it once in sorted order. Neither is executable here.
    //
    // This is not a cost question, so it cannot be left to the cost model: the
    // hash join is CHEAPER, and would win. It has to be ruled out on
    // applicability, exactly like the keyless MergeJoin below it.
    //
    // The bug this closes was silent and shape-dependent. `LEFT JOIN LATERAL (..)
    // ON true` has no equi-key, so it fell to NestedLoopJoin anyway and looked
    // correct; add an equi-condition to the ON clause and the same query lowered
    // to a HashJoin over a correlated right input, with lower() reporting ok.
    if (join_is_lateral(join_kind) && !is_nested_loop(op)) return false;

    // A streaming aggregate needs its input sorted on the grouping keys, and a
    // sort requirement is POSITIONAL - so if any grouping key is not a plain
    // column reference there is no requirement to state, and streaming is not an
    // option. Hashing is, because it hashes the expressions themselves.
    //
    // Applicability, not cost, for the usual reason: a streaming aggregate is the
    // CHEAPER of the two per row, so left to the cost model it would win and emit
    // a plan whose stated input requirement does not describe what it needs.
    if (op == PhysicalOp::StreamingAggregate && !grouping.orderable) return false;

    // A GROUPING SETS node computes SEVERAL key combinations; a plain aggregate
    // computes exactly one. Offering a plain aggregate for the former would return
    // the wrong ROWS while reporting success, and offering the grouping-sets
    // operator for a plain GROUP BY would compute a redundant set. Applicability
    // again, and again because the wrong one is the cheaper: the plain aggregates
    // hash once where this hashes once per set.
    if (is_aggregate_family(op) && computes_grouping_sets(op) != grouping.has_grouping_sets) {
        return false;
    }

    // The two set-operation implementations are not alternatives - they compute
    // different things - so applicability, not cost, decides between them.
    // UnionAll concatenates and compares nothing, which is right for UNION ALL and
    // wrong for everything else; HashSetOp compares, which is needed by everything
    // else and pure waste for UNION ALL. Left to the cost model, UnionAll is the
    // cheaper of the two and would win an INTERSECT.
    if (op == PhysicalOp::UnionAll && set_op != ast::SetOp::UnionAll) return false;
    if (op == PhysicalOp::HashSetOp && set_op == ast::SetOp::UnionAll) return false;

    // Both keyed joins need at least one equi-key: a merge join has nothing to
    // merge on without one, and a hash join has nothing to hash on. Note this
    // cannot be left to costing - with no keys a merge join also requires no
    // sort, so it would incur no enforcement and its cheaper per-row cost would
    // WIN, producing a merge join over a cross product. Applicability has to be
    // decided before cost.
    //
    // A keyless join is not unplannable: it is a NestedLoopJoin, the one join
    // that needs no key. Guarding only MergeJoin (as this did originally) did not
    // close the class - it moved the same unexecutable plan onto HashJoin.
    if (needs_equi_key(op)) return !keys.empty();
    return true;
}

std::optional<PhysicalProperties> pushdown_requirement(PhysicalOp op,
                                                      const PhysicalProperties& required) {
    // Which operators can satisfy a consumer's requirement by REQUIRING IT OF
    // THEIR INPUT rather than having an enforcer stacked on top of them. Only the
    // order- and format-preserving single-input operators can: derive_op says a
    // Filter and a Project pass their input's properties through unchanged, so an
    // input that arrives sorted leaves sorted.
    //
    // This is one half of the enforce-vs-push-down choice; it is not automatically
    // the better half. Pushing a Sort BELOW a Filter sorts rows the Filter is
    // about to discard, so the search costs both routes and takes the cheaper -
    // it does not assume that earlier is better.
    //
    // A Limit is deliberately NOT in this list even though derive_op says it
    // preserves order. Pushing an order requirement INTO a Limit's input changes
    // which rows the Limit keeps: `Sort_x(Limit_10(child))` takes ten rows in the
    // child's own order and then sorts those ten, while `Limit_10(Sort_x(child))`
    // takes the ten smallest by x. Those are different result SETS, not two
    // spellings of one plan. The order a Limit's input arrives in is fixed by
    // whatever the logical plan put underneath it, and the search is not free to
    // change it. Enforcing above the Limit is always correct; not pushing down is
    // at worst a missed optimization, and there is no version of this that trades
    // correctness for one.
    switch (op) {
        case PhysicalOp::Filter:
        case PhysicalOp::Project:
            return required;
        default:
            // A hash join and a nested loop destroy order; a merge join produces
            // its own; a scan is a leaf. None can pass a requirement downward.
            return std::nullopt;
    }
}

std::vector<PhysicalProperties> required_input_properties(PhysicalOp op,
                                                          const HashKeyVec& keys,
                                                          std::span<const SortKey> group_sort,
                                                          std::uint32_t group_key_count) {
    std::vector<PhysicalProperties> reqs(expected_arity(op));

    // WHERE THE ROWS HAVE TO BE. Same machinery as the sort order above it: state
    // the requirement, let the search cost the Exchange that would establish it,
    // and let it choose. On a single-node catalog - which is the default, and DB25
    // today - every one of these is already satisfied and no Exchange appears.
    //
    // A JOIN needs its two inputs CO-LOCATED on the join keys: a row can only be
    // matched against a row on the same node. Each side is required hashed on ITS
    // OWN key columns, which is why the two requirements below are built from
    // different halves of the same HashKey.
    if (op == PhysicalOp::HashJoin || op == PhysicalOp::MergeJoin ||
        op == PhysicalOp::NestedLoopJoin || op == PhysicalOp::HashSemiJoin ||
        op == PhysicalOp::HashAntiJoin || op == PhysicalOp::NestedLoopSemiJoin ||
        op == PhysicalOp::NestedLoopAntiJoin) {
        if (keys.empty()) {
            // No key to partition on: a cross product pairs every row with every
            // other, which is only correct if they are all in one place.
            for (PhysicalProperties& r : reqs) {
                r.distribution = Distribution{DistributionKind::Single, {}};
            }
        } else if (reqs.size() == 2) {
            Distribution left{DistributionKind::Hashed, {}};
            Distribution right{DistributionKind::Hashed, {}};
            for (const HashKey& k : keys) {
                left.keys.push_back(k.left_index);
                right.keys.push_back(k.right_index);
            }
            reqs[0].distribution = std::move(left);
            reqs[1].distribution = std::move(right);
        }
    }

    // An aggregate needs every row of a group on one node, or it computes a
    // partial per node and calls it the answer. A SCALAR aggregate is the extreme
    // case of that: one group, so one node.
    if (op == PhysicalOp::HashAggregate || op == PhysicalOp::StreamingAggregate ||
        op == PhysicalOp::HashGroupingSets) {
        Distribution d{DistributionKind::Single, {}};
        if (group_key_count != 0) {
            d = Distribution{DistributionKind::Hashed, {}};
            // The grouping keys are the first `group_key_count` entries of
            // `group_sort` when they are expressible as columns at all; when they
            // are not (GROUP BY a + b), there is no column to partition on and
            // Single is the only correct requirement.
            if (group_sort.size() >= group_key_count) {
                for (std::uint32_t i = 0; i < group_key_count; ++i) {
                    d.keys.push_back(group_sort[i].column);
                }
            } else {
                d = Distribution{DistributionKind::Single, {}};
            }
        }
        if (!reqs.empty()) reqs[0].distribution = std::move(d);
    }

    // DISTINCT is over every output column, so duplicates must meet.
    if (op == PhysicalOp::HashDistinct || op == PhysicalOp::StreamingDistinct) {
        Distribution d{DistributionKind::Hashed, {}};
        for (std::uint32_t i = 0; i < group_sort.size(); ++i) d.keys.push_back(group_sort[i].column);
        if (d.keys.empty()) d = Distribution{DistributionKind::Single, {}};
        if (!reqs.empty()) reqs[0].distribution = std::move(d);
    }

    // Set operations, recursion and the write path all require Single, and each
    // for its own reason rather than by a blanket rule: a hash set operation has
    // to see both sides' duplicates together, a fixpoint's working table is one
    // table, and a distributed write is a transaction problem rather than a
    // planning one. Each is its own increment; requiring Single is what makes
    // saying that honest instead of optimistic.
    if (op == PhysicalOp::UnionAll || op == PhysicalOp::HashSetOp ||
        op == PhysicalOp::RecursiveFixpoint || op == PhysicalOp::CreateTableAs ||
        op == PhysicalOp::Insert || op == PhysicalOp::Update || op == PhysicalOp::Delete) {
        for (PhysicalProperties& r : reqs) {
            r.distribution = Distribution{DistributionKind::Single, {}};
        }
    }

    // The reference engine's join operators consume ROW-format input (they
    // materialize rows into a hash table or merge row streams), so a columnar
    // subplan feeding a join must be converted - and that conversion is priced,
    // which is what makes the substrate choice a real trade rather than "columnar
    // is always cheaper". A vectorized columnar join is a later operator; when it
    // arrives it simply declares a different requirement here.
    // Both aggregates materialize rows too - into a hash table, or into a running
    // group accumulator - so they take the same row-format requirement.
    if (op == PhysicalOp::HashJoin || op == PhysicalOp::MergeJoin ||
        op == PhysicalOp::NestedLoopJoin || op == PhysicalOp::HashAggregate ||
        op == PhysicalOp::StreamingAggregate || op == PhysicalOp::Window ||
        op == PhysicalOp::HashDistinct || op == PhysicalOp::StreamingDistinct ||
        op == PhysicalOp::UnionAll || op == PhysicalOp::HashSetOp ||
        op == PhysicalOp::HashSemiJoin || op == PhysicalOp::HashAntiJoin ||
        op == PhysicalOp::NestedLoopSemiJoin || op == PhysicalOp::NestedLoopAntiJoin ||
        op == PhysicalOp::HashGroupingSets ||
        // Both write rows: the fixpoint into the working table, the CTAS into the
        // new table. A table is stored row-wise here, so a columnar subplan
        // feeding either must be converted, and that conversion is priced like
        // any other.
        op == PhysicalOp::RecursiveFixpoint || op == PhysicalOp::CreateTableAs ||
        // The write path materializes rows into a table, which is stored row-wise.
        op == PhysicalOp::Insert || op == PhysicalOp::Update ||
        op == PhysicalOp::Delete) {
        for (PhysicalProperties& r : reqs) r.format = StorageFormat::Row;
    }

    // An UPDATE or a DELETE reads the rows it is about to modify, and a lagging
    // copy is not merely a worse source - it is the wrong set of rows. So the
    // input requirement is FRESH, which is a correctness constraint rather than a
    // preference: no enforcer manufactures freshness, so a target available only
    // as a stale replica makes the statement unplannable, and it fails saying so
    // rather than modifying rows chosen from a lagging read.
    //
    // An INSERT is deliberately NOT here. Its input is the SOURCE query, not the
    // target: `INSERT INTO t SELECT ... FROM archive` may legitimately read a
    // replica, and requiring otherwise would refuse a query that is fine.
    if ((op == PhysicalOp::Update || op == PhysicalOp::Delete) && !reqs.empty()) {
        reqs[0].freshness = Freshness::Fresh;
    }

    // The requirement that makes a streaming aggregate what it is: its input must
    // already be grouped, i.e. sorted on the grouping keys. It is cheaper per row
    // than hashing and dearer once a Sort has to be enforced to supply this - the
    // same trade as MergeJoin against HashJoin, and settled the same way, by
    // costing both routes rather than by a rule.
    if (op == PhysicalOp::StreamingAggregate && !reqs.empty()) {
        reqs[0].sort.assign(group_sort.begin(), group_sort.end());
        reqs[0].format = StorageFormat::Row;
    }

    // A window operator needs its input sorted by PARTITION BY then ORDER BY -
    // partitions contiguous, and ordered within each. `group_sort` carries that
    // combined key list, on the same footing as it carries a streaming
    // aggregate's grouping order: in both cases it is the ordered key set the
    // operator consumes.
    if (op == PhysicalOp::Window && !reqs.empty()) {
        reqs[0].sort.assign(group_sort.begin(), group_sort.end());
        reqs[0].format = StorageFormat::Row;
    }

    // A streaming DISTINCT needs its input sorted on every output column, so that
    // duplicates are adjacent. `group_sort` carries that key list - the third use
    // of the same idea: the ordered key set the operator consumes.
    if (op == PhysicalOp::StreamingDistinct && !reqs.empty()) {
        reqs[0].sort.assign(group_sort.begin(), group_sort.end());
        reqs[0].format = StorageFormat::Row;
    }

    if (op == PhysicalOp::MergeJoin) {
        PhysicalProperties left, right;
        left.sort.reserve(keys.size());
        right.sort.reserve(keys.size());
        for (const HashKey& k : keys) {
            left.sort.push_back(SortKey{k.left_index, false});
            right.sort.push_back(SortKey{k.right_index, false});
        }
        left.format = StorageFormat::Row;
        right.format = StorageFormat::Row;
        if (reqs.size() == 2) { reqs[0] = std::move(left); reqs[1] = std::move(right); }
    }
    return reqs;  // every other operator: no requirement on its inputs
}

PhysicalProperties derive(const PhysicalNode& node) {
    std::vector<PhysicalProperties> input_props;
    input_props.reserve(node.children.size());
    for (const auto& c : node.children) input_props.push_back(derive(*c));

    PhysicalProperties p = derive_op(node.op, node.hash_keys, node.scan_format, input_props,
                                    node.scan_freshness, node.sort_keys,
                                    node.scan_distribution,
                                    static_cast<std::uint32_t>(node.group_keys.size()));
    // FormatConvert alone still carries its established property post-hoc; see the
    // note on derive_op. It is never a memo group, so nothing else has to agree.
    if (node.op == PhysicalOp::FormatConvert) p.format = node.target_format;
    // The same post-hoc treatment, for the same reason: an Exchange is only ever
    // built by enforce(), never a memo group, so nothing asks what it provides
    // before it exists.
    if (node.op == PhysicalOp::Exchange) p.distribution = node.target_distribution;
    return p;
}

PhysicalNodePtr enforce(PhysicalNodePtr input, const PhysicalProperties& required) {
    // derive() walks the whole subtree, so calling it once per decision made
    // enforcement quadratic in plan depth: extraction enforces per child, and
    // each enforce re-walked everything below it. Derive ONCE, then track what
    // each enforcer establishes - the enforcers are the only thing changing the
    // properties, and each one's effect on them is exactly its own definition.
    PhysicalProperties have = derive(*input);
    if (satisfies(have, required)) {
        return input;  // already satisfied - no enforcer inserted
    }
    PhysicalNodePtr node = std::move(input);

    // Distribution FIRST, before the format conversion and the sort. Moving rows
    // is the expensive step, and doing it first means the sort that follows runs
    // on each node's own share rather than being done and then scattered - and a
    // sort's order does not survive a repartition anyway, so the other order
    // would establish an order and then destroy it.
    if (!distribution_satisfies(have.distribution, required.distribution)) {
        node = make_exchange(std::move(node), required.distribution);
        have.distribution = required.distribution;
        // No order survives an exchange - see derive_op - so a sort that was
        // already satisfied has to be re-established above it. Which is exactly
        // why the exchange goes FIRST: doing it the other way round would
        // establish an order and then destroy it.
        have.sort.clear();
    }
    // Format next, so the sort (if also needed) runs on the required format.
    if (required.format != StorageFormat::Any && have.format != required.format) {
        node = make_format_convert(std::move(node), required.format);
        have.format = required.format;  // what a FormatConvert establishes
    }
    // Then the order.
    if (!sort_prefix(have.sort, required.sort)) {
        node = make_sort(std::move(node), required.sort);
        have.sort = required.sort;      // what a Sort establishes
    }
    // Verify rather than assume. The enforcers establish order and format; NO
    // enforcer establishes freshness, so a Fresh requirement over stale input
    // reaches here unsatisfied. Returning the input anyway - as this used to -
    // hands back a plan that reads lagging data for a query that must not, which
    // is precisely the correctness hole the Freshness property exists to close.
    // Checking the postcondition rather than special-casing freshness also means
    // any future unenforceable property is caught the day it is added.
    //
    // This one derive stays deliberately: it reads the TREE rather than the
    // tracked value, so it would catch an enforcer whose real effect diverged
    // from what this function believes it establishes. Checking `have` here
    // instead would only confirm the bookkeeping agrees with itself.
    if (!satisfies(derive(*node), required)) {
        return nullptr;
    }
    return node;
}

}  // namespace db25::physical
