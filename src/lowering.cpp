#include "db25/physical/lowering.hpp"

#include "db25/physical/memo.hpp"

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

// Resolve the physical operator for a logical operator: via the spec's
// implementation rules when a spec is supplied, else the built-in mapping.
std::optional<PhysicalOp> resolve_physical(plan::LogicalOp op, const LoweringContext& ctx) {
    const char* ln = logical_op_name(op);
    if (ln == nullptr) return std::nullopt;
    if (ctx.spec != nullptr) {
        const std::string pn = ctx.spec->physical_for_logical(ln);
        if (pn.empty()) return std::nullopt;
        return physical_op_from_name(pn);
    }
    return builtin_physical(op);
}

// Decompose an equi-join predicate into hash keys. Returns true only if the whole
// predicate is a conjunction of cross-side column equalities (so no residual is
// needed); on anything else it returns false and leaves `keys` for the caller to
// discard. `left_width` is the number of columns on the join's left input, so a
// right-side column index is offset back into the right input's own schema.
bool extract_keys(const plan::Expr* e, std::uint32_t left_width, std::vector<HashKey>& keys) {
    if (e == nullptr) return false;
    if (e->kind == plan::ExprKind::BinaryOp && e->bin_op == ast::BinaryOp::And) {
        const plan::Expr* l = e->children.size() > 0 ? e->children[0].get() : nullptr;
        const plan::Expr* r = e->children.size() > 1 ? e->children[1].get() : nullptr;
        return extract_keys(l, left_width, keys) && extract_keys(r, left_width, keys);
    }
    if (e->kind == plan::ExprKind::BinaryOp && e->bin_op == ast::BinaryOp::Equal &&
        e->children.size() == 2 &&
        e->children[0]->kind == plan::ExprKind::ColumnRef &&
        e->children[1]->kind == plan::ExprKind::ColumnRef) {
        const std::uint32_t i = e->children[0]->input_index;
        const std::uint32_t j = e->children[1]->input_index;
        if (i < left_width && j >= left_width) { keys.push_back({i, j - left_width}); return true; }
        if (j < left_width && i >= left_width) { keys.push_back({j, i - left_width}); return true; }
    }
    return false;
}

GroupId lower_into_memo(const plan::LogicalNode& n, Memo& memo, const LoweringContext& ctx,
                        std::string& error) {
    const std::optional<PhysicalOp> pop = resolve_physical(n.op, ctx);
    if (!pop) {
        const char* ln = logical_op_name(n.op);
        error = ln ? ("no implementation rule for logical operator '" + std::string(ln) + "'")
                   : "unsupported logical operator in Increment-0 lowering";
        return kInvalidGroup;
    }

    std::vector<GroupId> inputs;
    for (std::size_t i = 0; i < n.child_count(); ++i) {
        const GroupId g = lower_into_memo(*n.child(i), memo, ctx, error);
        if (g == kInvalidGroup) return kInvalidGroup;
        inputs.push_back(g);
    }

    GroupExpr ge;
    ge.op = *pop;
    ge.inputs = inputs;
    switch (n.op) {
        case plan::LogicalOp::Scan:
            ge.table_name = n.table_name;
            break;
        case plan::LogicalOp::Filter:
            ge.predicate = n.predicate.get();  // borrowed
            break;
        case plan::LogicalOp::Project:
            for (const auto& e : n.exprs) ge.projections.push_back(e.get());
            break;
        case plan::LogicalOp::Join: {
            const std::uint32_t left_width =
                n.child(0) != nullptr ? static_cast<std::uint32_t>(n.child(0)->output.size()) : 0;
            std::vector<HashKey> keys;
            const plan::Expr* pred = n.predicate.get();
            const bool clean = (pred == nullptr) ? true : extract_keys(pred, left_width, keys);
            if (clean) {
                ge.hash_keys = std::move(keys);  // pure equi-join (or CROSS: no keys)
            } else {
                ge.predicate = pred;  // keep the whole predicate as a residual
            }
            break;
        }
        default:
            break;
    }

    const GroupId g = memo.add_group(n.output);
    const std::uint32_t idx = memo.add_expr(g, std::move(ge));
    memo.set_winner(g, idx);
    return g;
}

}  // namespace

LoweringResult lower(const plan::LogicalNode& root, const LoweringContext& ctx) {
    LoweringResult result;
    Memo memo;
    const GroupId root_group = lower_into_memo(root, memo, ctx, result.error);
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
