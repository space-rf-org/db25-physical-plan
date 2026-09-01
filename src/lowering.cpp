#include "db25/physical/lowering.hpp"

#include "db25/physical/cost.hpp"
#include "db25/physical/memo.hpp"
#include "db25/physical/properties.hpp"
#include "db25/physical/structural_key.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"

#include <cstdint>
#include <algorithm>
#include <limits>
#include <optional>
#include <optional>
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
        default:                       return nullptr;
    }
}

// The built-in mapping used when no spec is supplied. A join offers BOTH keyed
// and keyless candidates: with no spec, a cross product still has to have a legal
// implementation, and only the nested loop is one.
std::vector<PhysicalOp> builtin_physical(plan::LogicalOp op) {
    switch (op) {
        case plan::LogicalOp::Scan:    return {PhysicalOp::SeqScan};
        case plan::LogicalOp::Filter:  return {PhysicalOp::Filter};
        case plan::LogicalOp::Project: return {PhysicalOp::Project};
        case plan::LogicalOp::Join:    return {PhysicalOp::HashJoin,
                                               PhysicalOp::NestedLoopJoin};
        default:                       return {};
    }
}

// Every physical operator a logical operator may lower to: the spec's
// implementation rules when a spec is supplied, else the built-in mapping. More
// than one candidate makes the choice cost-based.
std::vector<PhysicalOp> resolve_candidates(plan::LogicalOp op, const LoweringContext& ctx) {
    std::vector<PhysicalOp> out;
    const char* ln = logical_op_name(op);
    if (ln == nullptr) return out;
    if (ctx.spec != nullptr) {
        for (const std::string& name : ctx.spec->physicals_for_logical(ln)) {
            if (const std::optional<PhysicalOp> p = physical_op_from_name(name)) out.push_back(*p);
        }
        return out;
    }
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

// ---- phase 1: exploration -------------------------------------------------
// Build the memo's groups and their candidate group-expressions. NO costing and
// NO winner here: what a candidate costs, and therefore which one wins, depends
// on the requirement it is being optimized for - which exploration does not know.
// Separating the two phases is what makes the search property-directed rather
// than a bottom-up sweep that happens to consult properties.
GroupId explore(const plan::LogicalNode& n, Memo& memo, const LoweringContext& ctx,
                const CardinalityModel& card, const StorageCatalog& storage,
                std::size_t& candidates, std::size_t& shared, std::string& error) {
    const std::vector<PhysicalOp> cands = resolve_candidates(n.op, ctx);
    if (cands.empty()) {
        const char* ln = logical_op_name(n.op);
        error = ln ? ("no implementation rule for logical operator '" + std::string(ln) + "'")
                   : "unsupported logical operator in lowering";
        return kInvalidGroup;
    }

    std::vector<GroupId> inputs;
    for (std::size_t i = 0; i < n.child_count(); ++i) {
        const GroupId g =
            explore(*n.child(i), memo, ctx, card, storage, candidates, shared, error);
        if (g == kInvalidGroup) return kInvalidGroup;
        inputs.push_back(g);
    }

    // The payload is shared by every candidate: they are alternative ALGORITHMS
    // for the same logical operator, not different operators.
    GroupExpr base;
    base.inputs = inputs;
    switch (n.op) {
        case plan::LogicalOp::Scan:
            base.table_name = n.table_name;
            break;
        case plan::LogicalOp::Filter:
            base.predicate = n.predicate.get();  // borrowed
            break;
        case plan::LogicalOp::Project:
            for (const auto& e : n.exprs) base.projections.push_back(e.get());
            break;
        case plan::LogicalOp::Join: {
            const std::uint32_t left_width =
                n.child(0) != nullptr ? static_cast<std::uint32_t>(n.child(0)->output.size()) : 0;
            // Keys and residual are independent outputs: a join may have both
            // (an equi-key plus an extra condition), either, or neither (CROSS).
            split_join_predicate(n.predicate.get(), left_width, base.hash_keys, base.residual);
            break;
        }
        default:
            break;
    }

    // Structural identity: if an equivalent subtree has already been explored,
    // share its group rather than build a second copy. Children are explored (and
    // shared) first, so by the time this runs the input GroupIds are themselves
    // canonical - which is what makes a shallow key sufficient.
    GroupKey key;
    key.logical_op = static_cast<int>(n.op);
    // Borrowed from the LOGICAL NODE, not from the local GroupExpr: the index
    // outlives this call, and pointing at a local was a stack-use-after-return
    // (ASan caught it on the first run). Only a Scan carries a table name, which
    // is exactly when base.table_name was set.
    key.table_name = n.op == plan::LogicalOp::Scan ? &n.table_name : nullptr;
    key.output = &n.output;
    key.inputs = inputs;
    key.predicate = base.predicate;
    key.residual = base.residual;
    key.projections = base.projections;
    key.hash_keys = base.hash_keys;
    key.finish();
    if (const std::optional<GroupId> existing = memo.find_group(key)) {
        ++shared;
        return *existing;
    }

    const GroupId g = memo.add_group(n.output);

    // Cardinality belongs to the GROUP: the candidates are equivalent, so any of
    // them yields the same estimate, and it does not depend on any requirement.
    std::vector<double> input_rows;
    input_rows.reserve(inputs.size());
    for (const GroupId in : inputs) input_rows.push_back(memo.group(in).rows);
    memo.set_rows(g, operator_rows(cands.front(), input_rows, base.table_name, card));

    // A scan is available once per storage format the table actually has; every
    // other operator has a single (irrelevant) format slot.
    std::vector<FormatAvailability> scan_formats{FormatAvailability{StorageFormat::Row}};
    if (n.op == plan::LogicalOp::Scan) {
        scan_formats = storage.formats_for(base.table_name);
        // A correctness constraint, not a cost one: drop the copies that cannot
        // answer this query at all. No enforcer could rescue them, so they must
        // not reach the cost comparison in the first place.
        if (ctx.required_freshness == Freshness::Fresh) {
            std::vector<FormatAvailability> eligible;
            for (const FormatAvailability& fa : scan_formats) {
                if (fa.freshness == Freshness::Fresh) eligible.push_back(fa);
            }
            if (eligible.empty()) {
                error = "table '" + base.table_name +
                        "' has no copy fresh enough for this query";
                return kInvalidGroup;
            }
            scan_formats = std::move(eligible);
        }
    }

    for (const PhysicalOp cand : cands) {
        // A candidate whose precondition does not hold is not a cheaper option, it
        // is not an option at all.
        if (!is_applicable(cand, base.hash_keys)) continue;
        for (const FormatAvailability& fa : scan_formats) {
            GroupExpr ge = base;
            ge.op = cand;
            ge.scan_format = fa.format;
            ge.scan_freshness = fa.freshness;
            ge.input_reqs = required_input_properties(cand, ge.hash_keys);
            memo.add_expr(g, std::move(ge));
            ++candidates;
        }
    }
    if (memo.group(g).exprs.empty()) {
        const char* ln = logical_op_name(n.op);
        error = std::string("no applicable physical operator for logical '") +
                (ln ? ln : "?") + "'";
        return kInvalidGroup;
    }
    memo.index_group(key, g);
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
    std::size_t goals = 0;
    std::size_t pruned = 0;

    static constexpr double kNoBound = std::numeric_limits<double>::infinity();

    // Optimize each input under `reqs`, accumulating cost and what they provide.
    // Returns false when an input cannot meet its requirement at any price, OR
    // when the running total has already exceeded `budget` - at which point the
    // candidate that asked for these inputs cannot win and the remaining inputs
    // need not be optimized at all.
    bool optimize_inputs(const std::vector<GroupId>& inputs,
                         const std::vector<PhysicalProperties>& reqs, double budget,
                         double& cost_out, std::vector<PhysicalProperties>& provided_out) {
        cost_out = 0.0;
        provided_out.clear();
        for (std::size_t i = 0; i < inputs.size(); ++i) {
            const PhysicalProperties& r =
                i < reqs.size() ? reqs[i] : Group::unconstrained();
            const std::optional<std::uint32_t> w =
                optimize(inputs[i], r, prune ? budget - cost_out : kNoBound);
            if (!w) return false;
            const WinnerEntry& e = memo.group(inputs[i]).winners[*w];
            cost_out += e.cost;
            if (prune && cost_out >= budget) return false;  // cannot win; stop here
            provided_out.push_back(e.provided);
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
        std::vector<PhysicalProperties> best_input_required;

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
            const GroupExpr& ge = memo.group(g).exprs[i];
            std::vector<double> in_rows;
            in_rows.reserve(ge.inputs.size());
            for (const GroupId in : ge.inputs) in_rows.push_back(memo.group(in).rows);
            const std::vector<PhysicalProperties>& op_reqs = ge.input_reqs;
            const double own_cost =
                operator_cost(ge.op, in_rows, out_rows, cal, ge.scan_format);

            // The operator's own work alone already costs at least the bound, so
            // no arrangement of its inputs can rescue it.
            if (own_cost >= bound) { ++pruned; continue; }

            // --- route 1: satisfy the operator's own needs, enforce on top ---
            double child_cost = 0.0;
            std::vector<PhysicalProperties> child_props;
            if (optimize_inputs(ge.inputs, op_reqs, bound - own_cost, child_cost,
                                child_props)) {
                const PhysicalProperties provided = derive_op(
                    ge.op, ge.hash_keys, ge.scan_format, child_props, ge.scan_freshness);
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
            if (const std::optional<PhysicalProperties> down =
                    pushdown_requirement(ge.op, required)) {
                std::vector<PhysicalProperties> pushed = op_reqs;
                if (!pushed.empty()) {
                    // The operator's own need AND the consumer's, on input 0.
                    PhysicalProperties combined = *down;
                    if (pushed[0].format != StorageFormat::Any) {
                        combined.format = pushed[0].format;
                    }
                    if (combined.sort.empty()) combined.sort = pushed[0].sort;
                    pushed[0] = combined;

                    double pd_cost = 0.0;
                    std::vector<PhysicalProperties> pd_props;
                    if (optimize_inputs(ge.inputs, pushed, bound - own_cost, pd_cost,
                                        pd_props)) {
                        const PhysicalProperties provided =
                            derive_op(ge.op, ge.hash_keys, ge.scan_format, pd_props,
                                      ge.scan_freshness);
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
    const GroupId root_group = explore(root, memo, ctx, card, storage,
                                       result.candidates_considered, result.groups_shared,
                                       result.error);
    if (root_group == kInvalidGroup) {
        return result;  // ok stays false, error set
    }
    memo.set_root(root_group);
    result.memo_groups = memo.size();

    Optimizer opt{memo, cal, ctx.prune, 0, 0};
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
