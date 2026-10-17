#include "db25/physical/sexpr.hpp"

#include "db25/ast/node_types.hpp"  // binary_op_to_string, unary_op_to_string, data_type_to_string

#include <string>
#include <variant>

namespace db25::physical {
namespace {

// ---- expression rendering ------------------------------------------------
// A compact, deterministic renderer for the expression kinds Increment 0 lowers
// (column refs, literals, comparisons). Anything else falls through to a generic
// `(<kind> <child>...)` form so the render is always total and structure-
// sensitive; richer per-kind rendering is added as later increments lower more.

std::string render_literal(const plan::LiteralValue& v) {
    if (const auto* i = std::get_if<std::int64_t>(&v.value)) return std::to_string(*i);
    if (const auto* d = std::get_if<double>(&v.value)) return std::to_string(*d);
    if (const auto* s = std::get_if<std::string>(&v.value)) return "'" + *s + "'";
    if (const auto* b = std::get_if<bool>(&v.value)) return *b ? "true" : "false";
    return "NULL";  // std::monostate
}

std::string render_expr(const Expr* e) {
    if (e == nullptr) return "_";
    switch (e->kind) {
        case plan::ExprKind::ColumnRef:
            return "(col #" + std::to_string(e->input_index) + ")";
        case plan::ExprKind::OuterRef:
            return "(outer d" + std::to_string(e->outer_depth) + " #" +
                   std::to_string(e->input_index) + ")";
        case plan::ExprKind::Literal:
            return "(lit " + render_literal(e->value) + ")";
        case plan::ExprKind::BinaryOp: {
            const Expr* l = e->children.size() > 0 ? e->children[0].get() : nullptr;
            const Expr* r = e->children.size() > 1 ? e->children[1].get() : nullptr;
            return std::string("(") + ast::binary_op_to_string(e->bin_op) + " " +
                   render_expr(l) + " " + render_expr(r) + ")";
        }
        case plan::ExprKind::UnaryOp: {
            const Expr* o = e->children.empty() ? nullptr : e->children[0].get();
            return std::string("(") + ast::unary_op_to_string(e->un_op) + " " +
                   render_expr(o) + ")";
        }
        case plan::ExprKind::Aggregate: {
            // DISTINCT, FILTER and ORDER BY are rendered, not skipped. They are
            // part of WHAT the aggregate computes, not decoration: `COUNT(*)` and
            // `COUNT(*) FILTER (WHERE salary > 100)` return different numbers.
            //
            // The plan itself was never wrong - a physical node borrows the whole
            // logical Expr, so an executor sees all three. But this writer is what
            // the goldens are made of, and a golden that renders both as `(COUNT)`
            // cannot fail when one silently becomes the other. Exactly the hole
            // that let an INNER and a LEFT join share a physical golden.
            std::string s = "(" + e->func_name;
            if (e->distinct) s += " distinct";
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            if (e->filter != nullptr) s += " :filter " + render_expr(e->filter.get());
            if (!e->agg_order_by.empty()) {
                s += " :order [";
                for (std::size_t i = 0; i < e->agg_order_by.size(); ++i) {
                    if (i != 0) s += " ";
                    s += render_expr(e->agg_order_by[i].expr.get());
                    s += e->agg_order_by[i].descending ? " desc" : " asc";
                }
                s += "]";
            }
            return s + ")";
        }
        case plan::ExprKind::WindowFunction: {
            // The OVER clause is rendered for the same reason an aggregate's
            // FILTER is: RANK() OVER (PARTITION BY a) and RANK() OVER (PARTITION
            // BY b) compute different values, and a golden that renders both as
            // `(RANK)` cannot fail when one becomes the other.
            std::string s = "(" + e->func_name;
            if (e->distinct) s += " distinct";
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            {
                s += " :over [";
                if (!e->window.partition_by.empty()) {
                    s += ":partition [";
                    for (std::size_t i = 0; i < e->window.partition_by.size(); ++i) {
                        if (i != 0) s += " ";
                        s += render_expr(e->window.partition_by[i].get());
                    }
                    s += "]";
                }
                if (!e->window.order_by.empty()) {
                    if (!e->window.partition_by.empty()) s += " ";
                    s += ":order [";
                    for (std::size_t i = 0; i < e->window.order_by.size(); ++i) {
                        if (i != 0) s += " ";
                        s += render_expr(e->window.order_by[i].expr.get());
                        s += e->window.order_by[i].descending ? " desc" : " asc";
                    }
                    s += "]";
                }
                // The frame is carried as text at this layer (the logical IR does
                // not interpret it yet), so it is rendered as text rather than
                // dropped - a plan that silently forgot ROWS BETWEEN would compute
                // a different answer.
                if (e->window.frame.present) s += " :frame \"" + e->window.frame.spec + "\"";
                s += "]";
            }
            return s + ")";
        }
        case plan::ExprKind::ScalarFunction: {
            std::string s = "(" + e->func_name;
            if (e->distinct) s += " distinct";
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            return s + ")";
        }
        // The kinds below carry their meaning in a FLAG or an ENUM rather than in
        // their children, so the generic `(<kind> <child>...)` form rendered a
        // predicate and its negation identically:
        //
        //   age IS NULL      and  age IS NOT NULL   ->  (IsNull (col #1))
        //   city LIKE 'a%'   and  city NOT LIKE 'a%' ->  (Like (col #1) (lit 'a%'))
        //   age IN (1,2)     and  age NOT IN (1,2)  ->  (InList (col #1) ...)
        //   x IS TRUE / IS FALSE / IS NOT UNKNOWN   ->  (BooleanTest (col #1))
        //   CAST(x AS TEXT)  and  CAST(x AS BIGINT) ->  (Cast (col #0))
        //
        // The plans were never wrong - a physical node borrows the whole logical
        // Expr, flags included, so an executor sees the negation. But this writer
        // is what the staged goldens are made of, and a golden that renders both
        // senses the same cannot fail when one silently becomes the other. That
        // is the hole this file's Aggregate and WindowFunction cases were written
        // to close; these kinds were left in the generic branch and kept it open.
        case plan::ExprKind::IsNull: {
            const Expr* o = e->children.empty() ? nullptr : e->children[0].get();
            return std::string(e->negated() ? "(IsNotNull " : "(IsNull ") +
                   render_expr(o) + ")";
        }
        case plan::ExprKind::Like: {
            // Four distinct operators share this kind: LIKE / NOT LIKE / ILIKE /
            // NOT ILIKE. The optional ESCAPE operand is a child and renders as one.
            std::string s = "(";
            if (e->negated()) s += "Not";
            s += e->case_insensitive() ? "ILike" : "Like";
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            return s + ")";
        }
        case plan::ExprKind::InList: {
            std::string s = e->negated() ? "(NotInList" : "(InList";
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            return s + ")";
        }
        case plan::ExprKind::BooleanTest: {
            // IS [NOT] TRUE / FALSE / UNKNOWN: six operators, one kind. The
            // target is an enum and the NOT is a flag, so neither is in the
            // children.
            const char* target = e->bool_test == plan::BoolTest::True      ? "true"
                                 : e->bool_test == plan::BoolTest::False   ? "false"
                                                                           : "unknown";
            std::string s = "(BooleanTest :is";
            if (e->negated()) s += "-not";
            s += " ";
            s += target;
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            return s + ")";
        }
        case plan::ExprKind::Cast: {
            // The result type appears in the enclosing node's out= schema only
            // when the cast is the whole output expression. Nested in a
            // predicate - `(> (Cast (col #1)) (lit 1))` - nothing else records
            // it, and a cast to TEXT compares differently from a cast to BIGINT.
            std::string s = std::string("(Cast :to ") +
                            ast::data_type_to_string(e->target_type);
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            return s + ")";
        }
        case plan::ExprKind::Parameter:
            return "(param $" + std::to_string(e->param_index) + ")";
        case plan::ExprKind::Subquery: {
            // Today's optimizer decorrelates EXISTS / IN / scalar subqueries into
            // semi, anti and apply joins, so a Subquery expression does not reach
            // a lowered plan. Rendered anyway, and rendered with its
            // discriminators, because "unreachable" is a property of the current
            // optimizer and not of this writer - the day one survives lowering,
            // the render should not quietly lose which kind it was. The inner
            // plan is NOT rendered: it is a whole LogicalNode tree and this
            // function renders expressions. :sub marks that something is there.
            const char* kind = e->subquery_kind == plan::SubqueryKind::Scalar ? "scalar"
                               : e->subquery_kind == plan::SubqueryKind::In   ? "in"
                               : e->subquery_kind == plan::SubqueryKind::Exists ? "exists"
                                                                               : "quantified";
            std::string s = std::string("(Subquery :kind ") + kind;
            if (e->negated()) s += " :not";
            if (e->correlated) s += " :correlated";
            if (e->subquery_kind == plan::SubqueryKind::Quantified) {
                s += std::string(" :sense ") + (e->quant_all() ? "all" : "any");
                s += std::string(" :op ") + ast::binary_op_to_string(e->bin_op);
            }
            if (e->sub_plan != nullptr) s += " :sub";
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            return s + ")";
        }
        default: {
            // Still reached by Case, Between, Row and ScalarFunction-like kinds
            // whose meaning IS their children. The negation flag is rendered here
            // too: if a future kind starts carrying one, it shows up as `:not`
            // rather than disappearing, which is the failure this block of cases
            // exists to prevent.
            std::string s = std::string("(") + plan::expr_kind_to_string(e->kind);
            if (e->negated()) s += " :not";
            for (const auto& c : e->children) s += " " + render_expr(c.get());
            return s + ")";
        }
    }
}

// ---- schema rendering ----------------------------------------------------
std::string render_schema(const Schema& schema) {
    std::string s = "[";
    for (std::size_t i = 0; i < schema.size(); ++i) {
        const ColumnSchema& c = schema[i];
        if (i != 0) s += " ";
        s += c.name;
        s += ":";
        s += ast::data_type_to_string(c.type);
        if (c.nullable) s += "?";
    }
    return s + "]";
}

// A SET list: which column each assignment targets and what it is set to. The
// column is a catalog COLUMN ID, not a positional index into anything in this
// plan - an assignment names a column of the target relation, which the operator
// writes and does not read - so it is rendered with a `col` prefix that says so.
std::string render_assignments(const std::vector<plan::Assignment>* a) {
    std::string out = "set=[";
    if (a != nullptr) {
        for (std::size_t i = 0; i < a->size(); ++i) {
            if (i != 0) out += " ";
            out += "(col" + std::to_string((*a)[i].target_column_id) + " " +
                   render_expr((*a)[i].value.get()) + ")";
        }
    }
    out += "]";
    return out;
}

// ---- node rendering ------------------------------------------------------
void render_node(const PhysicalNode& n, const std::string& indent, std::string& out) {
    out += indent;
    out += "(";
    out += physical_op_to_string(n.op);

    switch (n.op) {
        case PhysicalOp::SeqScan:
            out += " table=" + n.table_name;
            out += std::string(" fmt=") + storage_format_to_string(n.scan_format);
            // Only the interesting case is rendered: a fresh scan is the norm.
            if (n.scan_freshness == Freshness::Stale) out += " stale";
            // Likewise single-node is the norm, and printing it on every scan of
            // every plan would be noise. A PARTITIONED table is the fact worth
            // seeing, and it says which columns it is partitioned on.
            if (n.scan_distribution.kind != DistributionKind::Single) {
                out += std::string(" dist=") +
                       distribution_kind_to_string(n.scan_distribution.kind);
                if (!n.scan_distribution.keys.empty()) {
                    out += " on=[";
                    for (std::size_t i = 0; i < n.scan_distribution.keys.size(); ++i) {
                        if (i != 0) out += " ";
                        out += "#" + std::to_string(n.scan_distribution.keys[i]);
                    }
                    out += "]";
                }
            }
            break;
        case PhysicalOp::Filter:
            out += " pred=" + render_expr(n.predicate);
            break;
        case PhysicalOp::Project: {
            out += " exprs=[";
            for (std::size_t i = 0; i < n.projections.size(); ++i) {
                if (i != 0) out += " ";
                out += render_expr(n.projections[i]);
            }
            out += "]";
            break;
        }
        case PhysicalOp::HashJoin:
        case PhysicalOp::MergeJoin:
        case PhysicalOp::NestedLoopJoin: {
            // The join kind comes FIRST, and is always printed - including for an
            // inner join. A physical join that does not say which relational join
            // it implements is ambiguous to any reader, and a field that is only
            // printed when it differs from a default trains readers to assume the
            // default is what silence means. Silence is what let INNER and LEFT
            // render byte-identical goldens.
            out += " kind=";
            out += join_kind_to_string(n.join_kind);
            // Which side is materialized. Printed only for the operator that
            // actually hashes - a merge join and a nested loop build nothing, and
            // a field meaningless for two of the three operators sharing this case
            // would be noise rather than information.
            if (n.op == PhysicalOp::HashJoin) {
                out += n.build_right ? " build=right" : " build=left";
            }
            out += " keys=[";
            for (std::size_t i = 0; i < n.hash_keys.size(); ++i) {
                if (i != 0) out += " ";
                out += "(L#" + std::to_string(n.hash_keys[i].left_index) + " R#" +
                       std::to_string(n.hash_keys[i].right_index) + ")";
            }
            out += "]";
            if (!n.residual.empty()) {
                out += " residual=[";
                for (std::size_t i = 0; i < n.residual.size(); ++i) {
                    if (i != 0) out += " ";
                    out += render_expr(n.residual[i]);
                }
                out += "]";
            }
            break;
        }
        case PhysicalOp::HashSemiJoin:
        case PhysicalOp::HashAntiJoin:
        case PhysicalOp::NestedLoopSemiJoin:
        case PhysicalOp::NestedLoopAntiJoin:
            // No kind= here: semi and anti are named by the OPERATOR, and the
            // outer-join kinds do not apply to them.
            out += " keys=[";
            for (std::size_t i = 0; i < n.hash_keys.size(); ++i) {
                if (i != 0) out += " ";
                out += "(L#" + std::to_string(n.hash_keys[i].left_index) + " R#" +
                       std::to_string(n.hash_keys[i].right_index) + ")";
            }
            out += "]";
            if (!n.residual.empty()) {
                out += " residual=[";
                for (std::size_t i = 0; i < n.residual.size(); ++i) {
                    if (i != 0) out += " ";
                    out += render_expr(n.residual[i]);
                }
                out += "]";
            }
            break;
        case PhysicalOp::Sort: {
            out += " by=[";
            for (std::size_t i = 0; i < n.sort_keys.size(); ++i) {
                if (i != 0) out += " ";
                const SortKey& k = n.sort_keys[i];
                out += "#" + std::to_string(k.column) + (k.descending ? " desc" : " asc");
                // Printed only when the query asked, so an enforcer's key (which
                // has no opinion) is visibly different from an explicit request.
                if (k.nulls_specified) out += k.nulls_first ? " nulls-first" : " nulls-last";
            }
            out += "]";
            break;
        }
        case PhysicalOp::HashGroupingSets:
        case PhysicalOp::HashAggregate:
        case PhysicalOp::StreamingAggregate: {
            out += " keys=[";
            for (std::size_t i = 0; i < n.group_keys.size(); ++i) {
                if (i != 0) out += " ";
                out += render_expr(n.group_keys[i]);
            }
            out += "] aggs=[";
            for (std::size_t i = 0; i < n.aggregates.size(); ++i) {
                if (i != 0) out += " ";
                out += render_expr(n.aggregates[i]);
            }
            out += "]";
            // The grouping sets, as the key indices active in each. Rendered
            // because they are WHAT IS COMPUTED: GROUP BY ROLLUP(a, b) and GROUP
            // BY CUBE(a, b) share every key and every aggregate and differ only
            // here, so a plan omitting them renders the two identically while they
            // return different rows.
            if (!n.grouping_sets.empty()) {
                out += " sets=[";
                for (std::size_t i = 0; i < n.grouping_sets.size(); ++i) {
                    if (i != 0) out += " ";
                    out += "(";
                    bool first = true;
                    for (std::uint32_t k = 0; k < 64; ++k) {
                        if ((n.grouping_sets[i] & (std::uint64_t{1} << k)) == 0) continue;
                        if (!first) out += " ";
                        out += std::to_string(k);
                        first = false;
                    }
                    out += ")";
                }
                out += "]";
            }
            break;
        }
        case PhysicalOp::UnionAll:
        case PhysicalOp::HashSetOp:
            // Printed for both, including UnionAll where it is implied by the
            // operator name: a set operation that does not say which one it is
            // reads the same as any other, and silence is what let INNER and LEFT
            // joins share a golden.
            out += " kind=";
            out += set_op_to_string(n.set_op);
            break;
        case PhysicalOp::ValuesScan: {
            out += " rows=[";
            const std::size_t w = n.values_columns;
            for (std::size_t i = 0; i < n.values.size(); ++i) {
                if (i != 0) out += (w != 0 && i % w == 0) ? ") (" : " ";
                if (i == 0) out += "(";
                out += render_expr(n.values[i]);
            }
            if (!n.values.empty()) out += ")";
            out += "]";
            break;
        }
        case PhysicalOp::HashDistinct:
        case PhysicalOp::StreamingDistinct:
            break;  // no payload: DISTINCT is over every output column
        case PhysicalOp::Window: {
            out += " fns=[";
            for (std::size_t i = 0; i < n.window_functions.size(); ++i) {
                if (i != 0) out += " ";
                out += render_expr(n.window_functions[i]);
            }
            out += "]";
            break;
        }
        case PhysicalOp::Limit: {
            if (n.limits.has_limit) out += " limit=" + std::to_string(n.limits.limit);
            if (n.limits.has_offset) out += " offset=" + std::to_string(n.limits.offset);
            break;
        }
        case PhysicalOp::FormatConvert:
            out += std::string(" to=") + storage_format_to_string(n.target_format);
            break;
        case PhysicalOp::RecursiveFixpoint:
            // The CTE's name, and WHICH union. UNION de-duplicates across
            // iterations and UNION ALL does not, which on cyclic data is the
            // difference between terminating and not - so it is printed, for the
            // same reason a set operation and a join kind are.
            out += " cte=" + n.table_name;
            out += " kind=";
            out += set_op_to_string(n.set_op);
            break;
        case PhysicalOp::WorkingTableScan:
            out += " cte=" + n.table_name;
            break;
        case PhysicalOp::CreateTableAs:
            out += " table=" + n.table_name;
            break;
        case PhysicalOp::Insert: {
            out += " table=" + n.table_name;
            // The explicit column list, when the statement wrote one. Empty means
            // every column in declaration order, which is a different statement -
            // so the two are rendered differently rather than identically.
            if (n.target_columns != nullptr && !n.target_columns->empty()) {
                out += " cols=[";
                for (std::size_t i = 0; i < n.target_columns->size(); ++i) {
                    if (i != 0) out += " ";
                    out += (*n.target_columns)[i];
                }
                out += "]";
            }
            if (n.conflict_action != plan::ConflictAction::None) {
                out += " conflict=";
                out += n.conflict_action == plan::ConflictAction::DoNothing ? "nothing"
                                                                           : "update";
                if (n.conflict_columns != nullptr && !n.conflict_columns->empty()) {
                    out += " on=[";
                    for (std::size_t i = 0; i < n.conflict_columns->size(); ++i) {
                        if (i != 0) out += " ";
                        out += (*n.conflict_columns)[i];
                    }
                    out += "]";
                }
                if (n.conflict_action == plan::ConflictAction::DoUpdate) {
                    out += " " + render_assignments(n.assignments);
                }
            }
            break;
        }
        case PhysicalOp::Update:
            out += " table=" + n.table_name;
            out += " " + render_assignments(n.assignments);
            break;
        case PhysicalOp::Delete:
            out += " table=" + n.table_name;
            break;
        case PhysicalOp::Exchange: {
            out += std::string(" to=") +
                   distribution_kind_to_string(n.target_distribution.kind);
            // The partitioning columns, when there are any. A repartition that did
            // not say WHICH columns it hashes on would be indistinguishable from
            // one that hashes on others, and the two send rows to different nodes.
            if (!n.target_distribution.keys.empty()) {
                out += " on=[";
                for (std::size_t i = 0; i < n.target_distribution.keys.size(); ++i) {
                    if (i != 0) out += " ";
                    out += "#" + std::to_string(n.target_distribution.keys[i]);
                }
                out += "]";
            }
            break;
        }
    }

    out += " out=" + render_schema(n.output);

    for (const auto& child : n.children) {
        out += "\n";
        render_node(*child, indent + "  ", out);
    }
    out += ")";
}

}  // namespace

std::string physical_to_sexpr(const PhysicalNode& node) {
    std::string out;
    render_node(node, "", out);
    return out;
}

}  // namespace db25::physical
