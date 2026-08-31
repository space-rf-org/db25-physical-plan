#include "db25/physical/lowering.hpp"

#include "db25/physical/cost.hpp"
#include "db25/physical/memo.hpp"
#include "db25/physical/properties.hpp"

#include "db25/ast/node_types.hpp"
#include "db25/plan/expr_ir.hpp"

#include <cstdint>
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

// The built-in single-candidate mapping used when no spec is supplied.
std::optional<PhysicalOp> builtin_physical(plan::LogicalOp op) {
    switch (op) {
        case plan::LogicalOp::Scan:    return PhysicalOp::SeqScan;
        case plan::LogicalOp::Filter:  return PhysicalOp::Filter;
        case plan::LogicalOp::Project: return PhysicalOp::Project;
        case plan::LogicalOp::Join:    return PhysicalOp::HashJoin;
        default:                       return std::nullopt;
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
    if (const std::optional<PhysicalOp> p = builtin_physical(op)) out.push_back(*p);
    return out;
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

GroupId lower_into_memo(const plan::LogicalNode& n, Memo& memo, const LoweringContext& ctx,
                        const CalibrationProfile& cal, const CardinalityModel& card,
                        const StorageCatalog& storage, std::size_t& candidates,
                        std::string& error) {
    const std::vector<PhysicalOp> cands = resolve_candidates(n.op, ctx);
    if (cands.empty()) {
        const char* ln = logical_op_name(n.op);
        error = ln ? ("no implementation rule for logical operator '" + std::string(ln) + "'")
                   : "unsupported logical operator in Increment-0 lowering";
        return kInvalidGroup;
    }

    std::vector<GroupId> inputs;
    for (std::size_t i = 0; i < n.child_count(); ++i) {
        const GroupId g =
            lower_into_memo(*n.child(i), memo, ctx, cal, card, storage, candidates, error);
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

    const GroupId g = memo.add_group(n.output);

    // Cardinality belongs to the GROUP: the candidates are equivalent, so any of
    // them yields the same estimate. Costing a parent then reads its inputs' rows
    // straight off their groups instead of re-deriving them down the tree.
    std::vector<double> input_rows;
    input_rows.reserve(inputs.size());
    for (const GroupId in : inputs) input_rows.push_back(memo.group(in).rows);
    const double out_rows = operator_rows(cands.front(), input_rows, base.table_name, card);
    memo.set_rows(g, out_rows);

    // An input group's contribution is the cost of its own winning subplan, which
    // is already settled: children are lowered (and chosen) before their parent.
    double inputs_cost = 0.0;
    for (const GroupId in : inputs) inputs_cost += memo.group(in).best_cost();

    // A scan is available once per storage format the table actually has; every
    // other operator has a single (irrelevant) format slot. This is what turns the
    // HTAP substrate into an ordinary costed choice rather than a special case.
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
        // A candidate pays for the enforcers it will cause. Without this a
        // MergeJoin over unsorted inputs would look cheaper than it is, get
        // chosen, and then have Sorts inserted at extraction that nobody costed.
        double enforce_cost = 0.0;
        const std::vector<PhysicalProperties> reqs =
            required_input_properties(cand, ge.hash_keys);
        for (std::size_t i = 0; i < inputs.size() && i < reqs.size(); ++i) {
            enforce_cost += enforcement_cost(memo.group(inputs[i]).provided, reqs[i],
                                             input_rows[i], cal);
        }
        ge.cost = inputs_cost + enforce_cost +
                  operator_cost(cand, input_rows, out_rows, cal, fa.format);
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
    memo.select_cheapest(g);  // the cost-based choice

    // Settle what this group now provides, so its parent can ask.
    std::vector<PhysicalProperties> input_props;
    input_props.reserve(inputs.size());
    for (const GroupId in : inputs) input_props.push_back(memo.group(in).provided);
    const GroupExpr& won = memo.group(g).exprs[*memo.group(g).winner];
    memo.set_provided(g, derive_op(won.op, won.hash_keys, won.scan_format, input_props,
                                   won.scan_freshness));
    return g;
}

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
    const GroupId root_group = lower_into_memo(root, memo, ctx, cal, card, storage,
                                               result.candidates_considered, result.error);
    if (root_group == kInvalidGroup) {
        return result;  // ok stays false, error set
    }
    memo.set_root(root_group);
    result.memo_groups = memo.size();

    result.plan = memo.extract_winner();
    if (!result.plan) {
        result.error = "winner extraction failed (a group had no winner)";
        return result;
    }
    result.ok = true;
    return result;
}

}  // namespace db25::physical
