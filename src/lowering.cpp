#include "db25/physical/lowering.hpp"

#include "db25/physical/cost.hpp"
#include "db25/physical/arity_vec.hpp"
#include "db25/physical/memo.hpp"
#include "db25/physical/properties.hpp"
#include "db25/physical/structural_key.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"

#include <cstdint>
#include <algorithm>
#include <limits>
#include <optional>
#include <span>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace db25::physical {
namespace {

// The logical operator name as the spec's implementation rules spell it. Returns
// nullptr for a logical operator Increment 0 does not lower.
const char* logical_op_name(plan::LogicalOp op) {
    switch (op) {
        case plan::LogicalOp::Scan:    return "Scan";
        case plan::LogicalOp::Filter:  return "Filter";
        case plan::LogicalOp::Project: return "Project";
        case plan::LogicalOp::Join:    return "Join";
        case plan::LogicalOp::Sort:    return "Sort";
        case plan::LogicalOp::Limit:   return "Limit";
        case plan::LogicalOp::Aggregate: return "Aggregate";
        case plan::LogicalOp::Window:  return "Window";
        case plan::LogicalOp::Distinct: return "Distinct";
        case plan::LogicalOp::SetOp:   return "SetOp";
        case plan::LogicalOp::Values:  return "Values";
        case plan::LogicalOp::SemiJoin: return "SemiJoin";
        case plan::LogicalOp::AntiJoin: return "AntiJoin";
        default:                       return nullptr;
    }
}

// The built-in mapping used when no spec is supplied. A join offers BOTH keyed
// and keyless candidates: with no spec, a cross product still has to have a legal
// implementation, and only the nested loop is one.
std::span<const PhysicalOp> builtin_physical(plan::LogicalOp op) {
    // Static tables: the built-in mapping is fixed, so answering with a span
    // costs nothing per node.
    static constexpr PhysicalOp kScan[]{PhysicalOp::SeqScan};
    static constexpr PhysicalOp kFilter[]{PhysicalOp::Filter};
    static constexpr PhysicalOp kProject[]{PhysicalOp::Project};
    static constexpr PhysicalOp kJoin[]{PhysicalOp::HashJoin, PhysicalOp::NestedLoopJoin};
    static constexpr PhysicalOp kSort[]{PhysicalOp::Sort};
    static constexpr PhysicalOp kLimit[]{PhysicalOp::Limit};
    // Two candidates, like a join: hashing works on any input, streaming needs a
    // sorted one and is cheaper when it gets it. The search costs both.
    static constexpr PhysicalOp kAggregate[]{PhysicalOp::HashAggregate,
                                             PhysicalOp::StreamingAggregate,
                                             PhysicalOp::HashGroupingSets};
    static constexpr PhysicalOp kWindow[]{PhysicalOp::Window};
    static constexpr PhysicalOp kDistinct[]{PhysicalOp::HashDistinct,
                                            PhysicalOp::StreamingDistinct};
    // UnionAll and HashSetOp are not alternatives for one another - they compute
    // different things - so BOTH are offered and applicability, not cost, picks:
    // UNION ALL admits only the concatenation, everything else only the hash.
    static constexpr PhysicalOp kSetOp[]{PhysicalOp::UnionAll, PhysicalOp::HashSetOp};
    static constexpr PhysicalOp kValues[]{PhysicalOp::ValuesScan};
    static constexpr PhysicalOp kSemi[]{PhysicalOp::HashSemiJoin,
                                        PhysicalOp::NestedLoopSemiJoin};
    static constexpr PhysicalOp kAnti[]{PhysicalOp::HashAntiJoin,
                                        PhysicalOp::NestedLoopAntiJoin};
    switch (op) {
        case plan::LogicalOp::Scan:    return kScan;
        case plan::LogicalOp::Filter:  return kFilter;
        case plan::LogicalOp::Project: return kProject;
        case plan::LogicalOp::Join:    return kJoin;
        case plan::LogicalOp::Sort:    return kSort;
        case plan::LogicalOp::Limit:   return kLimit;
        case plan::LogicalOp::Aggregate: return kAggregate;
        case plan::LogicalOp::Window:  return kWindow;
        case plan::LogicalOp::Distinct: return kDistinct;
        case plan::LogicalOp::SetOp:   return kSetOp;
        case plan::LogicalOp::Values:  return kValues;
        case plan::LogicalOp::SemiJoin: return kSemi;
        case plan::LogicalOp::AntiJoin: return kAnti;
        default:                       return {};
    }
}

// Every physical operator a logical operator may lower to: the spec's
// implementation rules when a spec is supplied, else the built-in mapping. More
// than one candidate makes the choice cost-based.
std::span<const PhysicalOp> resolve_candidates(plan::LogicalOp op, const LoweringContext& ctx) {
    const char* ln = logical_op_name(op);
    if (ln == nullptr) return {};
    if (ctx.spec != nullptr) return ctx.spec->ops_for_logical(ln);
    return builtin_physical(op);
}

// Split a join predicate into equi-join keys and the residual conjuncts that are
// not keys. PER CONJUNCT, deliberately: an all-or-nothing walk (one non-equi
// conjunct discarding every key already found) turns the ordinary shape
// `a.k = b.k AND a.d < b.d` into a KEYLESS join - a cross product with a filter,
// quadratic where it should be linear, and with MergeJoin made inapplicable
// because it has no keys left to merge on.
//
// `left_width` is the number of columns on the join's left input, so a right-side
// column index is offset back into the right input's own schema. Anything that is
// not a cross-side column equality - including a same-side equality, which
// constrains one input rather than relating the two - stays in `residual` and is
// re-checked per candidate row.
void split_join_predicate(const plan::Expr* e, std::uint32_t left_width,
                          std::vector<HashKey>& keys, std::vector<const plan::Expr*>& residual) {
    if (e == nullptr) return;
    if (e->kind == plan::ExprKind::BinaryOp && e->bin_op == ast::BinaryOp::And) {
        const plan::Expr* l = e->children.size() > 0 ? e->children[0].get() : nullptr;
        const plan::Expr* r = e->children.size() > 1 ? e->children[1].get() : nullptr;
        split_join_predicate(l, left_width, keys, residual);
        split_join_predicate(r, left_width, keys, residual);
        return;
    }
    if (e->kind == plan::ExprKind::BinaryOp && e->bin_op == ast::BinaryOp::Equal &&
        e->children.size() == 2 &&
        e->children[0]->kind == plan::ExprKind::ColumnRef &&
        e->children[1]->kind == plan::ExprKind::ColumnRef) {
        const std::uint32_t i = e->children[0]->input_index;
        const std::uint32_t j = e->children[1]->input_index;
        if (i < left_width && j >= left_width) { keys.push_back({i, j - left_width}); return; }
        if (j < left_width && i >= left_width) { keys.push_back({j, i - left_width}); return; }
    }
    residual.push_back(e);
}

// Do two OVER clauses partition and order identically? Compared structurally, on
// the expressions themselves: two window functions may share a node only if one
// input ordering serves both, and that is exactly what this asks.
bool same_window_spec(const plan::WindowSpecIR& a, const plan::WindowSpecIR& b) {
    if (a.partition_by.size() != b.partition_by.size()) return false;
    for (std::size_t i = 0; i < a.partition_by.size(); ++i) {
        if (!expr_structurally_equal(a.partition_by[i].get(), b.partition_by[i].get())) return false;
    }
    if (a.order_by.size() != b.order_by.size()) return false;
    for (std::size_t i = 0; i < a.order_by.size(); ++i) {
        if (a.order_by[i].descending != b.order_by[i].descending) return false;
        if (a.order_by[i].nulls_order_explicit != b.order_by[i].nulls_order_explicit) return false;
        if (a.order_by[i].nulls_order_explicit &&
            a.order_by[i].nulls_first != b.order_by[i].nulls_first) {
            return false;
        }
        if (!expr_structurally_equal(a.order_by[i].expr.get(), b.order_by[i].expr.get())) {
            return false;
        }
    }
    // The FRAME is deliberately NOT compared. Two functions differing only in
    // their frame want the same input ordering, so one Window operator serves
    // both; the frame is carried inside each borrowed expression and consumed by
    // the executor, not by the sort requirement.
    return true;
}

// A group's grouping information, derived from the payload it already carries.
// `orderable` is a COUNT COMPARISON rather than a separate flag: the sort keys are
// populated only when every grouping key could be turned into one, so the two
// sizes agreeing is exactly the condition, and there is no second field to fall
// out of step with the first.
// How many literal rows a ValuesScan group produces. A FROM-less SELECT carries
// no expressions and no columns, and is exactly one row - so an empty payload
// means one row, not zero, and dividing by the width would say the wrong thing.
double values_rows_of(const Group& g) {
    if (g.op_split == 0) return 1.0;
    return static_cast<double>(g.op_exprs.size() / g.op_split);
}

// A group's table name, or the empty string. `table_name` is a BORROWED pointer
// and null for every operator but a scan, so every reader needs this - returning a
// reference to a static empty string rather than constructing one per call.
const std::string& table_name_of(const Group& g) {
    static const std::string empty;
    return g.table_name != nullptr ? *g.table_name : empty;
}

GroupingSpec grouping_of(const Group& g) {
    return GroupingSpec{g.op_split, g.op_split == g.sort_keys.size(),
                        !g.grouping_sets.empty(),
                        static_cast<std::uint32_t>(g.grouping_sets.size())};
}

// Add every applicable candidate for one group: each implementation the rules
// offer, times each storage format the substrate has, times each build side that
// is a real choice. Shared by ordinary lowering and by the reordering DP below,
// which differ only in where the inputs and the predicate came from - so a new
// applicability rule or a new candidate axis lands in ONE loop rather than two
// that drift apart.
std::size_t add_candidates(Memo& memo, GroupId g, std::span<const PhysicalOp> cands,
                           std::span<const FormatAvailability> formats,
                           const std::vector<GroupId>& inputs,
                           const std::vector<HashKey>& keys,
                           const std::vector<const plan::Expr*>& residual) {
    const ast::JoinType join_kind = memo.group(g).join_kind;
    const GroupingSpec group_spec = grouping_of(memo.group(g));
    // The candidate count is known before the loop, so the vector never has to
    // grow and re-move what it already holds.
    // Reserve the EXACT candidate count, build sides included. Reserving only
    // cands * formats left the vector one grow short as soon as a join offered
    // both build sides, and that grow showed up as +2 allocations on the budget
    // query - a reserve that is nearly right is a reserve that reallocates.
    std::size_t n_cand = 0;
    for (const PhysicalOp cand : cands) {
        n_cand += formats.size() * (is_build_side_choosable(cand, join_kind) ? 2u : 1u);
    }
    // RESERVE, not assign: the DP adds the candidates of every split to the same
    // group, so this runs more than once per group and must never shrink what is
    // already there.
    memo.group(g).exprs.reserve(memo.group(g).exprs.size() + n_cand);
    std::size_t added = 0;
    for (const PhysicalOp cand : cands) {
        // A candidate whose precondition does not hold is not a cheaper option, it
        // is not an option at all.
        if (!is_applicable(cand, keys, join_kind, group_spec, memo.group(g).set_op)) continue;
        // Where the build side is a real choice, offer BOTH and let the search
        // cost them like any other pair. Where it is not, offer only the
        // semantically fixed one - a swapped candidate there would compute a
        // different relation, not a cheaper plan.
        const bool choose_build = is_build_side_choosable(cand, join_kind);
        for (const FormatAvailability& fa : formats) {
            for (int side = 0; side < (choose_build ? 2 : 1); ++side) {
                GroupExpr ge;
                ge.op = cand;
                ge.scan_format = fa.format;
                ge.scan_freshness = fa.freshness;
                ge.build_right = (side == 0);
                // Per candidate, because a candidate IS (operator, input groups) -
                // which is what lets the DP put two associations of the same join
                // in one group, reading different inputs on different keys.
                ge.inputs = inputs;
                ge.hash_keys = keys;
                ge.residual = residual;
                // Recomputed per candidate rather than hoisted out of these loops.
                // Hoisting looks like an obvious win - it depends on neither the
                // format nor the build side - but the result must then be COPIED
                // into each candidate, and a copy of an ArityVec<PhysicalProperties>
                // copies the sort vectors inside it. Measured: hoisting made the
                // budget query WORSE by one allocation, not better.
                ge.input_reqs = required_input_properties(cand, keys, memo.group(g).sort_keys);
                memo.add_expr(g, std::move(ge));
                ++added;
            }
        }
    }
    return added;
}

// ---- phase 1: exploration -------------------------------------------------
// Build the memo's groups and their candidate group-expressions. NO costing and
// NO winner here: what a candidate costs, and therefore which one wins, depends
// on the requirement it is being optimized for - which exploration does not know.
// Separating the two phases is what makes the search property-directed rather
// than a bottom-up sweep that happens to consult properties.
GroupId explore(const plan::LogicalNode& n, Memo& memo, const LoweringContext& ctx,
                const CardinalityModel& card, const StorageCatalog& storage,
                LoweringArena& arena, std::size_t& candidates, std::size_t& shared,
                std::size_t& joins, std::size_t& regions, std::string& error);

// ---- join reordering: the interval DP -------------------------------------
// A region of INNER / CROSS joins has many association trees returning the same
// rows at very different costs, and which one the query text happened to write is
// not a planning decision. This enumerates them.
//
// The state is a RANGE of leaves, because an in-order traversal gives every
// subtree a contiguous one (see join_order.hpp). So the DP is the classic
// interval DP: one memo group per range [first, last), and one group-expression
// per split point - each split being a different association, all producing the
// same relation, which is exactly what a memo group is for. That is O(n^2) groups
// and O(n^3) candidates, against the O(3^n) of the general subset DP, and it is
// the whole of what re-association (as opposed to permutation) can reach.
//
// TWO RULES BOUND IT, and neither is about cost:
//
//   Only CONNECTED splits are enumerated - a split whose join has a conjunct
//   touching both sides. Not primarily to avoid cross products, but because the
//   cardinality model charges one flat join_selectivity per join: a cross product
//   estimated at a fraction of |A|x|B| is not merely imprecise, it is off by that
//   fraction, and the DP would happily choose the plan built on that error. A
//   region with no connected tree at all is left exactly as written.
//
//   The leaf count is capped. n^2 groups and n^3 candidates is fine at ten leaves
//   and is not at sixty-four, and a planner that takes longer than the query is
//   not a faster planner.
constexpr std::size_t kMaxReorderLeaves = 10;

// Does the join over [first, split) and [split, last) carry a conjunct relating
// its two sides? A conjunct placed here that names only one side constrains an
// input; it does not join anything, so it does not make the pair connected.
bool split_connects(const JoinRegion& region, std::size_t first, std::size_t split,
                    std::size_t last) {
    const std::uint64_t l = range_mask(first, split);
    const std::uint64_t r = range_mask(split, last);
    for (const RegionConjunct& c : region.conjuncts) {
        if (!placed_here(c, first, split, last)) continue;
        if ((c.leaf_mask & l) != 0 && (c.leaf_mask & r) != 0) return true;
    }
    return false;
}

GroupId explore_join_region(const plan::LogicalNode& n, const JoinRegion& region,
                            Memo& memo, const LoweringContext& ctx,
                            const CardinalityModel& card, const StorageCatalog& storage,
                            LoweringArena& arena, std::size_t& candidates,
                            std::size_t& shared, std::size_t& joins,
                            std::size_t& regions, std::string& error) {
    const std::size_t nl = region.leaf_count();
    if (nl > kMaxReorderLeaves) return kInvalidGroup;
    const auto at = [nl](std::size_t first, std::size_t last) { return first * (nl + 1) + last; };

    // Pass one: which ranges have a connected tree at all. Run BEFORE anything is
    // built, so a region the DP cannot cover costs nothing and is lowered the
    // ordinary way - the fallback has to be free, or a query with a cross product
    // in it would pay for an enumeration that found nothing.
    std::vector<char> viable((nl + 1) * (nl + 1), 0);
    for (std::size_t i = 0; i < nl; ++i) viable[at(i, i + 1)] = 1;  // a leaf is its own plan
    for (std::size_t width = 2; width <= nl; ++width) {
        for (std::size_t first = 0; first + width <= nl; ++first) {
            const std::size_t last = first + width;
            for (std::size_t split = first + 1; split < last; ++split) {
                if (viable[at(first, split)] == 0 || viable[at(split, last)] == 0) continue;
                if (!split_connects(region, first, split, last)) continue;
                viable[at(first, last)] = 1;
                break;
            }
        }
    }
    if (viable[at(0, nl)] == 0) return kInvalidGroup;

    // Pass two: build. Leaves first, then every viable range in increasing width,
    // so a range's inputs already exist when its candidates are added.
    std::vector<GroupId> gid((nl + 1) * (nl + 1), kInvalidGroup);
    for (std::size_t i = 0; i < nl; ++i) {
        const GroupId lg = explore(*region.leaves[i], memo, ctx, card, storage, arena,
                                   candidates, shared, joins, regions, error);
        if (lg == kInvalidGroup) return kInvalidGroup;  // error is set
        gid[at(i, i + 1)] = lg;
    }
    joins += nl - 1;  // the region's joins, however it ends up associated
    ++regions;

    const std::span<const PhysicalOp> cands = resolve_candidates(plan::LogicalOp::Join, ctx);
    if (cands.empty()) {
        error = "no implementation rule for logical operator 'Join'";
        return kInvalidGroup;
    }
    static const FormatAvailability kRowOnly[]{FormatAvailability{StorageFormat::Row}};

    for (std::size_t width = 2; width <= nl; ++width) {
        for (std::size_t first = 0; first + width <= nl; ++first) {
            const std::size_t last = first + width;
            if (viable[at(first, last)] == 0) continue;
            // The range's schema is its leaves' columns concatenated, which for the
            // WHOLE region is the logical node's own output - so the parent, which
            // holds indices into that, sees no change whatever the DP chooses. A
            // proper sub-range produced no logical node, so its schema is
            // synthesized here and owned by the arena.
            const plan::Schema* out = region.output;
            if (width != nl) {
                arena.schemas.push_back(std::make_unique<plan::Schema>());
                plan::Schema& sch = *arena.schemas.back();
                std::size_t w = 0;
                for (std::size_t i = first; i < last; ++i) w += region.leaves[i]->output.size();
                sch.reserve(w);
                for (std::size_t i = first; i < last; ++i) {
                    for (const plan::ColumnSchema& c : region.leaves[i]->output) sch.push_back(c);
                }
                out = &sch;
            }
            const GroupId g = memo.add_group(*out);
            // An INNER join: every join the DP builds carries a connecting
            // conjunct, so none of them is the cross product that CROSS names.
            memo.group(g).join_kind = ast::JoinType::Inner;

            bool rows_set = false;
            for (std::size_t split = first + 1; split < last; ++split) {
                if (viable[at(first, split)] == 0 || viable[at(split, last)] == 0) continue;
                if (!split_connects(region, first, split, last)) continue;
                const std::vector<GroupId> inputs{gid[at(first, split)], gid[at(split, last)]};
                // This join's own left input is the region's columns
                // [leaf_offset[first], end_of(split)) - contiguous, so its width is
                // a subtraction and every conjunct index a single offset away.
                const std::uint32_t left_width =
                    region.end_of(split) - region.leaf_offset[first];
                std::vector<HashKey> keys;
                std::vector<const plan::Expr*> residual;
                for (const RegionConjunct& c : region.conjuncts) {
                    if (!placed_here(c, first, split, last)) continue;
                    split_join_predicate(translate_conjunct(c, region, first, arena),
                                         left_width, keys, residual);
                }
                candidates += add_candidates(memo, g, cands, kRowOnly, inputs, keys, residual);
                if (!rows_set) {
                    // Cardinality is a property of the GROUP, and every split of a
                    // range yields the same one: rows multiply and the model
                    // charges one selectivity per join, so |A||B|s . |C|s and
                    // |A|s . |B||C|s are the same product. Taken from the first
                    // split rather than averaged, so the estimate is a number the
                    // model actually produced.
                    const double in_rows[2] = {memo.group(inputs[0]).rows,
                                               memo.group(inputs[1]).rows};
                    memo.set_rows(g, operator_rows(PhysicalOp::HashJoin,
                                                   std::span<const double>{in_rows, 2}, "", card,
                                                   LimitSpec{}, GroupingSpec{}, ast::SetOp::Union,
                                                   0.0));
                    rows_set = true;
                }
            }
            if (memo.group(g).exprs.empty()) {
                // Viability said a connected tree exists, so a split was offered;
                // reaching here means no physical operator was applicable to any of
                // them, which is a planning failure and not a reason to reorder less.
                error = "no applicable physical operator for a reordered join";
                return kInvalidGroup;
            }
            gid[at(first, last)] = g;
        }
    }
    return gid[at(0, nl)];
}

GroupId explore(const plan::LogicalNode& n, Memo& memo, const LoweringContext& ctx,
                const CardinalityModel& card, const StorageCatalog& storage,
                LoweringArena& arena, std::size_t& candidates, std::size_t& shared,
                std::size_t& joins, std::size_t& regions, std::string& error) {
    // A reorderable join region is planned as a whole rather than node by node:
    // the tree the query wrote is one of the trees the DP enumerates, not the
    // frame the others have to fit inside. When the region cannot be covered
    // (a cross product in it, or too many leaves) this returns kInvalidGroup with
    // no error and lowering proceeds exactly as it did before.
    if (ctx.reorder_joins && n.op == plan::LogicalOp::Join) {
        JoinRegion region;
        if (collect_join_region(n, region)) {
            const GroupId g = explore_join_region(n, region, memo, ctx, card, storage, arena,
                                                  candidates, shared, joins, regions, error);
            if (g != kInvalidGroup) return g;
            if (!error.empty()) return kInvalidGroup;
        }
    }
    if (n.op == plan::LogicalOp::Join) ++joins;
    const std::span<const PhysicalOp> cands = resolve_candidates(n.op, ctx);
    if (cands.empty()) {
        const char* ln = logical_op_name(n.op);
        error = ln ? ("no implementation rule for logical operator '" + std::string(ln) + "'")
                   : "unsupported logical operator in lowering";
        return kInvalidGroup;
    }

    std::vector<GroupId> inputs;
    for (std::size_t i = 0; i < n.child_count(); ++i) {
        const GroupId g = explore(*n.child(i), memo, ctx, card, storage, arena, candidates,
                                  shared, joins, regions, error);
        if (g == kInvalidGroup) return kInvalidGroup;
        inputs.push_back(g);
    }

    // The payload every candidate in this group SHARES - what the logical operator
    // is, rather than how it is computed. Assembled here before the group exists so
    // that dedup can key on it; moved in below.
    //
    // The inputs, join keys and residual are NOT here: they belong to each
    // candidate, because a candidate is (operator, input groups) and two candidates
    // in one group may read different inputs. Today they never do - lowering is one
    // group per logical node - so these locals are simply copied onto every
    // candidate, and the code says where they will diverge.
    Group payload;
    std::vector<HashKey> hash_keys;
    std::vector<const plan::Expr*> residual;
    switch (n.op) {
        case plan::LogicalOp::Scan:
            payload.table_name = &n.table_name;   // borrowed; the logical node outlives us
            break;
        case plan::LogicalOp::Filter:
            payload.predicate = n.predicate.get();  // borrowed
            break;
        case plan::LogicalOp::Project:
            for (const auto& e : n.exprs) payload.op_exprs.push_back(e.get());
            break;
        case plan::LogicalOp::SemiJoin:
        case plan::LogicalOp::AntiJoin:
        case plan::LogicalOp::Join: {
            // WHICH join this is. Everything below chooses an ALGORITHM; this
            // records the relational operator that algorithm has to implement.
            payload.join_kind = n.join_type;
            const std::uint32_t left_width =
                n.child(0) != nullptr ? static_cast<std::uint32_t>(n.child(0)->output.size()) : 0;
            // Keys and residual are independent outputs: a join may have both
            // (an equi-key plus an extra condition), either, or neither (CROSS).
            split_join_predicate(n.predicate.get(), left_width, hash_keys, residual);
            break;
        }
        case plan::LogicalOp::Sort:
            // ORDER BY. A key is a positional column index, so a key expression
            // that is not a plain column reference has no index to be - and
            // inventing one would sort by the wrong column. Fail honestly instead;
            // `ORDER BY a + b` needs the sort to carry an expression, which is a
            // change to the property system, not a line here.
            payload.sort_keys.reserve(n.sort_keys.size());
            for (const plan::SortKeyIR& k : n.sort_keys) {
                if (k.expr == nullptr || k.expr->kind != plan::ExprKind::ColumnRef) {
                    error = "unsupported sort key expression in lowering "
                            "(only a plain column reference is a positional sort key)";
                    return kInvalidGroup;
                }
                SortKey sk;
                sk.column = k.expr->input_index;
                sk.descending = k.descending;
                // NULLS FIRST / NULLS LAST changes the result, so it is carried,
                // not dropped. `nulls_specified` records whether the QUERY had an
                // opinion, which is what separates an explicit request from the
                // engine default and from an enforcer's indifference.
                sk.nulls_specified = k.nulls_order_explicit;
                sk.nulls_first = k.nulls_first;
                payload.sort_keys.push_back(sk);
            }
            break;
        case plan::LogicalOp::Aggregate: {
            // ROLLUP / CUBE / GROUPING SETS ask for several grouping-key
            // combinations at once. Neither operator here computes that, and
            // treating it as a plain GROUP BY would return the wrong ROWS while
            // reporting success - so it fails, and says which construct it is.
            // GROUPING SETS / ROLLUP / CUBE. Each set is a BITMASK over the
            // grouping keys; a key absent from a set is NULL in that set's output
            // rows. More than 64 grouping keys cannot be expressed - and 64 keys
            // under CUBE is 2^64 sets, so the query is unplannable long before the
            // representation is.
            if (!n.grouping_sets.empty()) {
                if (n.group_keys.size() > 64) {
                    error = "more than 64 grouping keys with GROUPING SETS in lowering";
                    return kInvalidGroup;
                }
                payload.grouping_sets.reserve(n.grouping_sets.size());
                for (const auto& set : n.grouping_sets) {
                    std::uint64_t mask = 0;
                    for (const std::uint32_t k : set) {
                        if (k >= n.group_keys.size()) {
                            error = "grouping set names a key that does not exist in lowering";
                            return kInvalidGroup;
                        }
                        mask |= (std::uint64_t{1} << k);
                    }
                    payload.grouping_sets.push_back(mask);
                }
            }
            payload.op_exprs.reserve(n.group_keys.size() + n.aggregates.size());
            for (const auto& e : n.group_keys) payload.op_exprs.push_back(e.get());
            payload.op_split = static_cast<std::uint32_t>(n.group_keys.size());
            for (const auto& e : n.aggregates) payload.op_exprs.push_back(e.get());

            // The grouping order, when it can be expressed at all. A sort
            // requirement is positional, so a key that is not a plain column
            // reference has no index to be. Rather than fail - hashing needs no
            // order and can hash any expression - the keys are simply not
            // recorded, which makes the streaming candidate inapplicable. The
            // count mismatch IS the signal; see GroupingSpec::orderable.
            bool orderable = true;
            for (std::uint32_t i = 0; i < payload.op_split; ++i) {
                const plan::Expr* e = payload.op_exprs[i];
                if (e == nullptr || e->kind != plan::ExprKind::ColumnRef) { orderable = false; break; }
            }
            if (orderable) {
                payload.sort_keys.reserve(payload.op_split);
                for (std::uint32_t i = 0; i < payload.op_split; ++i) {
                    const plan::Expr* e = payload.op_exprs[i];
                    payload.sort_keys.push_back(SortKey{e->input_index, false, false, false});
                }
            }
            break;
        }
        case plan::LogicalOp::Window: {
            if (n.window_functions.empty()) {
                error = "window node with no window functions in lowering";
                return kInvalidGroup;
            }
            // Every window function on ONE node must share an OVER clause, because
            // the operator states ONE input sort requirement. Two functions with
            // different PARTITION BY / ORDER BY need two Window operators fed by
            // two sorts; declaring one requirement and computing both would emit a
            // plan whose stated requirement does not describe what it needs - the
            // defect this planner keeps finding. Splitting the node is a
            // transformation rule, not a line here, so for now it fails and says
            // so.
            const plan::WindowSpecIR* spec = nullptr;
            for (const auto& e : n.window_functions) {
                if (e == nullptr) {
                    error = "null window function in lowering";
                    return kInvalidGroup;
                }
                if (spec == nullptr) { spec = &e->window; continue; }
                if (!same_window_spec(*spec, e->window)) {
                    error = "window functions with different OVER clauses on one "
                            "node are not yet implemented in physical lowering";
                    return kInvalidGroup;
                }
            }
            // PARTITION BY then ORDER BY, as one positional key list. A key that
            // is not a plain column reference cannot BE a positional sort key, and
            // unlike an aggregate there is no order-free alternative to fall back
            // to - so this fails rather than inventing an index.
            for (const auto& e : spec->partition_by) {
                if (e == nullptr || e->kind != plan::ExprKind::ColumnRef) {
                    error = "unsupported PARTITION BY expression in lowering "
                            "(only a plain column reference is a positional sort key)";
                    return kInvalidGroup;
                }
                payload.sort_keys.push_back(SortKey{e->input_index, false, false, false});
            }
            for (const plan::SortKeyIR& k : spec->order_by) {
                if (k.expr == nullptr || k.expr->kind != plan::ExprKind::ColumnRef) {
                    error = "unsupported window ORDER BY expression in lowering "
                            "(only a plain column reference is a positional sort key)";
                    return kInvalidGroup;
                }
                payload.sort_keys.push_back(SortKey{k.expr->input_index, k.descending,
                                                   k.nulls_order_explicit, k.nulls_first});
            }
            payload.op_exprs.reserve(n.window_functions.size());
            for (const auto& e : n.window_functions) payload.op_exprs.push_back(e.get());
            break;
        }
        case plan::LogicalOp::Distinct:
            // DISTINCT is over EVERY output column, and those are positional by
            // construction - column i of the child is column i here - so the sort
            // key list is always expressible and a streaming implementation is
            // always available. No honest-failure path is needed, unlike the
            // aggregate's grouping keys, which are arbitrary expressions.
            payload.sort_keys.reserve(n.output.size());
            for (std::uint32_t i = 0; i < n.output.size(); ++i) {
                payload.sort_keys.push_back(SortKey{i, false, false, false});
            }
            break;
        case plan::LogicalOp::SetOp:
            payload.set_op = n.set_op;
            break;
        case plan::LogicalOp::Values: {
            // Flattened row-major, `op_split` columns wide. A FROM-less SELECT is
            // one row of zero columns, which is why the width is carried
            // separately rather than inferred - it cannot be divided out of an
            // empty list.
            if (!n.value_rows.empty()) {
                payload.op_split = static_cast<std::uint32_t>(n.value_rows.front().size());
                for (const auto& row : n.value_rows) {
                    if (row.size() != payload.op_split) {
                        error = "VALUES rows of differing width in lowering";
                        return kInvalidGroup;
                    }
                    for (const auto& e : row) payload.op_exprs.push_back(e.get());
                }
            }
            break;
        }
        case plan::LogicalOp::Limit:
            payload.limits.has_limit = n.has_limit;
            payload.limits.limit = n.limit;
            payload.limits.has_offset = n.has_offset;
            payload.limits.offset = n.offset;
            break;
        default:
            break;
    }

    // Structural identity: if an equivalent subtree has already been explored,
    // share its group rather than build a second copy. Children are explored (and
    // shared) first, so by the time this runs the input GroupIds are themselves
    // canonical - which is what makes a shallow key sufficient.
    //
    // The key is built ONLY when dedup is on. Building it unconditionally and
    // testing the flag afterwards still copied four vectors per group, which cost
    // most of the feature's price on queries that had opted out of it.
    GroupKey key;
    if (ctx.dedup) {
        key.logical_op = static_cast<int>(n.op);
        // Borrowed from the LOGICAL NODE, not from the local GroupExpr: the index
        // outlives this call, and pointing at a local was a stack-use-after-return
        // (ASan caught it on the first run). Only a Scan carries a table name.
        key.table_name = n.op == plan::LogicalOp::Scan ? &n.table_name : nullptr;
        key.output = &n.output;
        key.inputs = inputs;
        key.predicate = payload.predicate;
        key.residual = residual;
        key.op_exprs = payload.op_exprs;
        key.op_split = payload.op_split;
        key.grouping_sets = payload.grouping_sets;
        key.set_op = payload.set_op;
        key.hash_keys = hash_keys;
        key.join_kind = payload.join_kind;
        key.sort_keys = payload.sort_keys;
        key.limits = payload.limits;
        key.finish();
        if (const std::optional<GroupId> existing = memo.find_group(key)) {
            ++shared;
            return *existing;
        }
    }

    const GroupId g = memo.add_group(n.output);
    {
        Group& grp = memo.group(g);
        grp.table_name = payload.table_name;
        grp.predicate = payload.predicate;
        grp.op_exprs = std::move(payload.op_exprs);
        grp.op_split = payload.op_split;
        grp.grouping_sets = std::move(payload.grouping_sets);
        grp.set_op = payload.set_op;
        grp.join_kind = payload.join_kind;
        grp.sort_keys = payload.sort_keys;
        grp.limits = payload.limits;
    }

    // Cardinality belongs to the GROUP: the candidates are equivalent, so any of
    // them yields the same estimate, and it does not depend on any requirement.
    double in_rows_buf[2] = {};
    std::size_t n_in = 0;
    for (const GroupId in : inputs) { if (n_in == 2) break; in_rows_buf[n_in++] = memo.group(in).rows; }
    memo.set_rows(g, operator_rows(cands.front(), std::span<const double>{in_rows_buf, n_in},
                                   table_name_of(memo.group(g)), card, memo.group(g).limits,
                                   grouping_of(memo.group(g)), memo.group(g).set_op,
                                   values_rows_of(memo.group(g))));

    // A scan is available once per storage format the table actually has; every
    // other operator has a single (irrelevant) format slot.
    std::vector<FormatAvailability> scan_formats{FormatAvailability{StorageFormat::Row}};
    if (n.op == plan::LogicalOp::Scan) {
        scan_formats = storage.formats_for(table_name_of(memo.group(g)));
        // A correctness constraint, not a cost one: drop the copies that cannot
        // answer this query at all. No enforcer could rescue them, so they must
        // not reach the cost comparison in the first place.
        if (ctx.required_freshness == Freshness::Fresh) {
            std::vector<FormatAvailability> eligible;
            for (const FormatAvailability& fa : scan_formats) {
                if (fa.freshness == Freshness::Fresh) eligible.push_back(fa);
            }
            if (eligible.empty()) {
                error = "table '" + table_name_of(memo.group(g)) +
                        "' has no copy fresh enough for this query";
                return kInvalidGroup;
            }
            scan_formats = std::move(eligible);
        }
    }

    candidates += add_candidates(memo, g, cands, scan_formats, inputs, hash_keys, residual);
    if (memo.group(g).exprs.empty()) {
        const char* ln = logical_op_name(n.op);
        error = std::string("no applicable physical operator for logical '") +
                (ln ? ln : "?") + "'";
        return kInvalidGroup;
    }
    if (ctx.dedup) memo.index_group(key, g);
    return g;
}

// ---- phase 2: property-directed optimization ------------------------------
// The search proper: "the best plan for group `g` that satisfies `required`",
// memoized under that requirement so the same goal is never re-derived.
//
// For each candidate the search costs TWO routes and keeps the cheaper:
//
//   enforce   - optimize the children for what the OPERATOR needs, then pay for
//               the enforcers that make this operator's own output satisfy the
//               consumer (a Sort above a hash join, say);
//   push down - where the operator preserves the property, require it of the
//               INPUT instead, so nothing is enforced above.
//
// Neither is universally better, which is exactly why it is a cost comparison
// rather than a rule. Sorting below a Filter sorts rows the Filter discards;
// sorting above a join sorts the (larger) join output.
struct Optimizer {
    Memo& memo;
    const CalibrationProfile& cal;
    bool prune = true;
    // The D5 budget guard. When engaged the search still costs every APPLICABLE
    // candidate with the same cost model - it just stops exploring the second
    // route, so the enforce-vs-push-down choice is not made and enforcement is
    // simply charged on top. That halves the work per candidate and bounds the
    // goals a group is asked for, at the price of a plan that may not be the one
    // the full search would have chosen.
    bool greedy = false;
    std::size_t goals = 0;
    std::size_t pruned = 0;

    static constexpr double kNoBound = std::numeric_limits<double>::infinity();
    // Every operator in the IR is a leaf, unary, or binary. expected_arity() is
    // the authority; this is the compile-time bound the scratch buffers use, and
    // a static_assert in lowering.cpp keeps the two from drifting apart.
    static constexpr std::size_t kMaxArity = 2;

    // Optimize each input under `reqs`, accumulating cost and what they provide.
    // Returns false when an input cannot meet its requirement at any price, OR
    // when the running total has already exceeded `budget` - at which point the
    // candidate that asked for these inputs cannot win and the remaining inputs
    // need not be optimized at all.
    // `provided_out` is a caller-owned fixed array, not a vector: arity is at most
    // kMaxArity, so the search has no reason to reach the heap once per candidate
    // per goal. This is the allocation the search MULTIPLIES - every other one is
    // bounded by plan size rather than by how hard the search works.
    bool optimize_inputs(const ArityVec<GroupId>& inputs,
                         const ArityVec<PhysicalProperties>& reqs, double budget,
                         double& cost_out, PhysicalProperties* provided_out,
                         std::size_t& provided_count) {
        cost_out = 0.0;
        provided_count = 0;
        for (std::size_t i = 0; i < inputs.size() && i < kMaxArity; ++i) {
            const PhysicalProperties& r =
                i < reqs.size() ? reqs[i] : Group::unconstrained();
            const std::optional<std::uint32_t> w =
                optimize(inputs[i], r, prune ? budget - cost_out : kNoBound);
            if (!w) return false;
            const WinnerEntry& e = memo.group(inputs[i]).winners[*w];
            cost_out += e.cost;
            if (prune && cost_out >= budget) return false;  // cannot win; stop here
            provided_out[provided_count++] = e.provided;
        }
        return true;
    }

    // `budget` is a strict upper bound: any plan costing at least this much is of
    // no use to the caller, so the search may abandon it unfinished. The bound is
    // ADMISSIBLE - it only ever discards candidates whose cost is already at least
    // the budget, and every term in a plan's cost is non-negative, so a partial
    // cost is a lower bound on the total. That is what lets pruning be a pure
    // speedup: it can never discard the optimum.
    std::optional<std::uint32_t> optimize(GroupId g, const PhysicalProperties& required,
                                          double budget = kNoBound) {
        if (const auto memoized = memo.group(g).winner_index_for(required)) {
            return memoized;  // this goal is already settled
        }
        ++goals;

        const double out_rows = memo.group(g).rows;
        const std::size_t n_exprs = memo.group(g).exprs.size();

        std::uint32_t best = static_cast<std::uint32_t>(n_exprs);
        double best_cost = std::numeric_limits<double>::infinity();
        PhysicalProperties best_provided;
        ArityVec<PhysicalProperties> best_input_required;

        for (std::uint32_t i = 0; i < n_exprs; ++i) {
            // The tightest bound available for this candidate: the caller's, or
            // the best this group has already found, whichever is smaller.
            const double bound = prune ? std::min(budget, best_cost) : kNoBound;

            // A reference is safe here because of a phase invariant: EXPLORATION
            // IS COMPLETE before optimization begins, so no group's candidate list
            // grows while the search runs. (Groups are a deque and winners are
            // addressed by index, so neither of those moves either.) If a later
            // unit ever applies rules DURING search, this is the line that has to
            // change - copying it back cost five allocations per candidate.
            const Group& grp = memo.group(g);
            const GroupExpr& ge = grp.exprs[i];
            double in_rows_buf[kMaxArity] = {};
            std::size_t n_in = 0;
            for (const GroupId in : ge.inputs) {
                if (n_in == kMaxArity) break;
                in_rows_buf[n_in++] = memo.group(in).rows;
            }
            const std::span<const double> in_rows{in_rows_buf, n_in};
            const ArityVec<PhysicalProperties>& op_reqs = ge.input_reqs;
            const double own_cost =
                operator_cost(ge.op, in_rows, out_rows, cal, ge.scan_format, ge.build_right,
                              static_cast<std::uint32_t>(grp.grouping_sets.size()));

            // The operator's own work alone already costs at least the bound, so
            // no arrangement of its inputs can rescue it.
            if (own_cost >= bound) { ++pruned; continue; }

            // --- route 1: satisfy the operator's own needs, enforce on top ---
            double child_cost = 0.0;
            PhysicalProperties child_props[kMaxArity];
            std::size_t n_child = 0;
            if (optimize_inputs(ge.inputs, op_reqs, bound - own_cost, child_cost,
                                child_props, n_child)) {
                const PhysicalProperties provided =
                    derive_op(ge.op, ge.hash_keys, ge.scan_format,
                              std::span<const PhysicalProperties>{child_props, n_child},
                              ge.scan_freshness, grp.sort_keys);
                const double top = enforcement_cost(provided, required, out_rows, cal);
                const double total = child_cost + own_cost + top;
                if (total < best_cost) {
                    best_cost = total;
                    best = i;
                    best_provided = provided;
                    best_input_required = op_reqs;
                    if (top != 0.0) {  // an enforcer will sit above this operator
                        if (required.format != StorageFormat::Any) {
                            best_provided.format = required.format;
                        }
                        if (!required.sort.empty()) best_provided.sort = required.sort;
                    }
                }
            }

            // --- route 2: push the requirement into the input instead ---
            // Skipped under the budget guard: this is the half of the search that
            // explores an ALTERNATIVE placement, and it is what the guard trades
            // away first because route 1 alone still yields a valid, costed plan.
            if (greedy) continue;
            if (const std::optional<PhysicalProperties> down =
                    pushdown_requirement(ge.op, required)) {
                ArityVec<PhysicalProperties> pushed = op_reqs;
                if (!pushed.empty()) {
                    // The operator's own need AND the consumer's, on input 0.
                    PhysicalProperties combined = *down;
                    if (pushed[0].format != StorageFormat::Any) {
                        combined.format = pushed[0].format;
                    }
                    if (combined.sort.empty()) combined.sort = pushed[0].sort;
                    pushed[0] = combined;

                    double pd_cost = 0.0;
                    PhysicalProperties pd_props[kMaxArity];
                    std::size_t n_pd = 0;
                    if (optimize_inputs(ge.inputs, pushed, bound - own_cost, pd_cost,
                                        pd_props, n_pd)) {
                        const PhysicalProperties provided = derive_op(
                            ge.op, ge.hash_keys, ge.scan_format,
                            std::span<const PhysicalProperties>{pd_props, n_pd},
                            ge.scan_freshness, grp.sort_keys);
                        const double top =
                            enforcement_cost(provided, required, out_rows, cal);
                        const double total = pd_cost + own_cost + top;
                        if (total < best_cost) {
                            best_cost = total;
                            best = i;
                            best_provided = provided;
                            best_input_required = pushed;
                        }
                    }
                }
            }
        }

        if (best == n_exprs || best_cost == std::numeric_limits<double>::infinity()) {
            // Nothing here can serve this goal - either at any price, or within
            // the caller's budget.
            //
            // Note what is NOT needed here: a guard against memoizing a
            // budget-cutoff as though it were this group's true optimum. A
            // candidate abandoned on the bound never becomes `best`, so a cutoff
            // reaches this line with nothing to record. (Checked, not assumed:
            // making this line memoize `best` when one exists is a no-op, because
            // `best` being set implies a finite cost and this line is only reached
            // when there is none.)
            //
            // The failure itself is deliberately not memoized either. A goal that
            // could not be met under a tight budget may well be met under a looser
            // one, and recording "impossible" would make a bound from one caller
            // silently answer a different caller's question.
            return std::nullopt;
        }
        memo.set_winner(g, required, best, best_cost, best_provided,
                        std::move(best_input_required));
        return memo.group(g).winner_index_for(required);
    }
};

}  // namespace

LoweringResult lower(const plan::LogicalNode& root, const LoweringContext& ctx) {
    LoweringResult result;
    // The planner is a pure function of its declared inputs; where an input was
    // not supplied, the documented default stands in for it.
    const CalibrationProfile cal =
        ctx.calibration != nullptr ? *ctx.calibration : default_calibration();
    const CardinalityModel card =
        ctx.cardinality != nullptr ? *ctx.cardinality : CardinalityModel{};
    const StorageCatalog storage = ctx.storage != nullptr ? *ctx.storage : StorageCatalog{};

    Memo memo;
    const GroupId root_group =
        explore(root, memo, ctx, card, storage, result.arena, result.candidates_considered,
                result.groups_shared, result.join_count, result.join_regions_enumerated,
                result.error);
    if (root_group == kInvalidGroup) {
        return result;  // ok stays false, error set
    }
    memo.set_root(root_group);
    result.memo_groups = memo.size();

    const std::uint32_t budget = ctx.max_join_count_override != 0
                                     ? ctx.max_join_count_override
                                     : (ctx.spec != nullptr ? ctx.spec->budget.max_join_count
                                                            : SearchBudget{}.max_join_count);
    result.budget_guard_engaged = result.join_count > budget;

    Optimizer opt{memo, cal, ctx.prune, result.budget_guard_engaged, 0, 0};
    if (!opt.optimize(root_group, ctx.required_output)) {
        result.error = "no plan satisfies the required output properties";
        return result;
    }
    result.optimization_goals = opt.goals;
    result.candidates_pruned = opt.pruned;

    result.plan = memo.extract_winner_for(root_group, ctx.required_output);
    if (!result.plan) {
        result.error = "winner extraction failed (a group had no winner, or a required "
                       "property could not be enforced)";
        return result;
    }

    // The ROOT's enforcer, which nothing else would insert. Every other enforcer
    // is placed by a parent onto its child; the root has no parent, so a plan
    // whose top operator does not itself provide the required output would
    // otherwise be returned unenforced - reported ok while not satisfying the
    // requirement it was optimized for. (The cost of this enforcer was already
    // charged: it is the `top` term the root's winner paid.)
    result.plan = enforce(std::move(result.plan), ctx.required_output);
    if (!result.plan) {
        result.error = "the required output properties cannot be enforced";
        return result;
    }

    // Postcondition, checked rather than assumed - the planner does not get to
    // report success on a plan that does not satisfy what it was asked for.
    if (!satisfies(derive(*result.plan), ctx.required_output)) {
        result.plan.reset();
        result.error = "internal: extracted plan does not satisfy the required output";
        return result;
    }
    result.ok = true;
    return result;
}

}  // namespace db25::physical
