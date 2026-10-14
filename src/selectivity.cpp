#include "db25/physical/selectivity.hpp"

#include <algorithm>
#include <variant>

namespace db25::physical {
namespace {

using plan::Expr;
using plan::ExprKind;
using ast::BinaryOp;

constexpr double kMinSel = 1e-6;   // never claim a predicate keeps nothing at all
constexpr double kMaxSel = 1.0;

double clamp_sel(double s) { return std::clamp(s, kMinSel, kMaxSel); }

// A ColumnRef that names a BASE table column. A computed column carries
// (0, 0) - the binder marks it so - and must not be looked up, or every
// expression in the query would collide on one statistics entry.
const ColumnStats* stats_for(const Expr& e, const StatisticsCatalog& stats) {
    if (e.kind != ExprKind::ColumnRef) return nullptr;
    if (e.ref_table_id == 0 || e.ref_column_id == 0) return nullptr;
    return stats.column(e.ref_table_id, e.ref_column_id);
}

// The numeric value of a literal, for range arithmetic. Only the arms that have
// an order: an integer or a double. A string literal has an order too, but
// comparing it against a numeric min/max would be nonsense, so it declines.
bool literal_number(const Expr& e, double& out) {
    if (e.kind != ExprKind::Literal) return false;
    if (const auto* i = std::get_if<std::int64_t>(&e.value.value)) {
        out = static_cast<double>(*i);
        return true;
    }
    if (const auto* d = std::get_if<double>(&e.value.value)) {
        out = *d;
        return true;
    }
    return false;
}

bool is_comparison(BinaryOp op) {
    switch (op) {
        case BinaryOp::Equal:
        case BinaryOp::NotEqual:
        case BinaryOp::LessThan:
        case BinaryOp::LessEqual:
        case BinaryOp::GreaterThan:
        case BinaryOp::GreaterEqual:
            return true;
        default:
            return false;
    }
}

// Mirror a comparison when the column is on the RIGHT. `1000 < salary` and
// `salary > 1000` are the same predicate, and an estimator that handled only one
// spelling would silently fall back on the other.
BinaryOp mirror(BinaryOp op) {
    switch (op) {
        case BinaryOp::LessThan:    return BinaryOp::GreaterThan;
        case BinaryOp::LessEqual:   return BinaryOp::GreaterEqual;
        case BinaryOp::GreaterThan: return BinaryOp::LessThan;
        case BinaryOp::GreaterEqual:return BinaryOp::LessEqual;
        default:                    return op;  // Equal and NotEqual are symmetric
    }
}

// `col <op> literal` where the column has a known numeric range: the fraction of
// the domain the predicate admits, by linear interpolation. Uniformity is the
// assumption, and it is the assumption a histogram exists to remove - stated
// here rather than hidden, because on skewed data this is where the error is.
double range_selectivity(BinaryOp op, const ColumnStats& cs, double v) {
    const double lo = cs.min_value;
    const double hi = cs.max_value;
    if (!(hi > lo)) {
        // A degenerate domain: every row holds the same value. Then the
        // predicate is not a fraction at all, it is true or false for all rows,
        // and answering a fraction would be worse than answering the extreme.
        switch (op) {
            case BinaryOp::LessThan:     return v > lo ? 1.0 : 0.0;
            case BinaryOp::LessEqual:    return v >= lo ? 1.0 : 0.0;
            case BinaryOp::GreaterThan:  return v < lo ? 1.0 : 0.0;
            case BinaryOp::GreaterEqual: return v <= lo ? 1.0 : 0.0;
            default:                     return 1.0;
        }
    }
    const double span = hi - lo;
    double s = 0.0;
    switch (op) {
        case BinaryOp::LessThan:
        case BinaryOp::LessEqual:
            s = (v - lo) / span;
            break;
        case BinaryOp::GreaterThan:
        case BinaryOp::GreaterEqual:
            s = (hi - v) / span;
            break;
        default:
            return 1.0;
    }
    return std::clamp(s, 0.0, 1.0);
}

}  // namespace

double selectivity_of(const Expr* pred, const StatisticsCatalog& stats,
                      const SelectivityDefaults& defaults) {
    // No predicate filters nothing. Charging a selectivity here would invent a
    // restriction the query did not write.
    if (pred == nullptr) return 1.0;

    switch (pred->kind) {
        case ExprKind::BinaryOp: {
            if (pred->bin_op == BinaryOp::And) {
                // Independence. Wrong whenever the conjuncts are correlated,
                // which is often - and correcting it needs multi-column
                // statistics, not a fudge factor on the product.
                double s = 1.0;
                for (const auto& c : pred->children) {
                    s *= selectivity_of(c.get(), stats, defaults);
                }
                return clamp_sel(s);
            }
            if (pred->bin_op == BinaryOp::Or) {
                // P(A or B) = 1 - (1-P(A))(1-P(B)), again under independence.
                double inv = 1.0;
                for (const auto& c : pred->children) {
                    inv *= 1.0 - selectivity_of(c.get(), stats, defaults);
                }
                return clamp_sel(1.0 - inv);
            }
            if (!is_comparison(pred->bin_op) || pred->children.size() != 2) {
                return defaults.unknown;
            }

            // Which side is the column, which the constant. Either spelling.
            const Expr* lhs = pred->children[0].get();
            const Expr* rhs = pred->children[1].get();
            BinaryOp op = pred->bin_op;
            const ColumnStats* cs = stats_for(*lhs, stats);
            const Expr* other = rhs;
            if (cs == nullptr) {
                cs = stats_for(*rhs, stats);
                other = lhs;
                op = mirror(op);
            }
            const bool equality =
                pred->bin_op == BinaryOp::Equal || pred->bin_op == BinaryOp::NotEqual;
            if (cs == nullptr) {
                // No statistics for either side. Fall back BY CLASS: an equality
                // and a range are not equally selective, and the old model's one
                // constant for both is the thing this replaces.
                return equality ? defaults.equality : defaults.inequality;
            }
            // A comparison between two COLUMNS is not a constant-selectivity
            // question and the range arithmetic below does not apply to it.
            double v = 0.0;
            if (!literal_number(*other, v)) {
                return equality ? defaults.equality : defaults.inequality;
            }
            // NULL satisfies no comparison, so the non-null fraction bounds
            // every answer below.
            const double non_null = 1.0 - std::clamp(cs->null_fraction, 0.0, 1.0);

            if (equality) {
                if (cs->ndv <= 0.0) return clamp_sel(defaults.equality * non_null);
                const double eq = non_null / cs->ndv;
                return clamp_sel(pred->bin_op == BinaryOp::Equal ? eq : non_null - eq);
            }
            if (!cs->has_range) return clamp_sel(defaults.inequality * non_null);
            return clamp_sel(range_selectivity(op, *cs, v) * non_null);
        }

        case ExprKind::IsNull: {
            if (pred->children.size() != 1) return defaults.unknown;
            const ColumnStats* cs = stats_for(*pred->children[0], stats);
            if (cs == nullptr) return defaults.unknown;
            const double nf = std::clamp(cs->null_fraction, 0.0, 1.0);
            // IS NOT NULL is the complement, and the parser records the NOT as a
            // flag rather than a separate node - so an estimator that ignored the
            // flag would answer IS NULL's number for both.
            const bool negated = (pred->expr_flags & plan::ExprFlagNegated) != 0;
            return clamp_sel(negated ? 1.0 - nf : nf);
        }

        case ExprKind::Between: {
            // value BETWEEN low AND high: the intersection of two ranges, which
            // under the same uniformity assumption is one subtraction rather than
            // the product of two independent halves. Treating it as a product
            // would square the selectivity of a range that is really one band.
            if (pred->children.size() != 3) return defaults.unknown;
            const ColumnStats* cs = stats_for(*pred->children[0], stats);
            double lo_v = 0.0;
            double hi_v = 0.0;
            if (cs == nullptr || !literal_number(*pred->children[1], lo_v) ||
                !literal_number(*pred->children[2], hi_v)) {
                return defaults.inequality;
            }
            const double non_null = 1.0 - std::clamp(cs->null_fraction, 0.0, 1.0);
            if (!cs->has_range) return clamp_sel(defaults.inequality * non_null);
            const double ge = range_selectivity(BinaryOp::GreaterEqual, *cs, lo_v);
            const double le = range_selectivity(BinaryOp::LessEqual, *cs, hi_v);
            // Both fractions are measured from opposite ends of the same domain,
            // so their overlap is what the band keeps.
            const double band = std::clamp(ge + le - 1.0, 0.0, 1.0);
            const bool negated = (pred->expr_flags & plan::ExprFlagNegated) != 0;
            return clamp_sel((negated ? 1.0 - band : band) * non_null);
        }

        case ExprKind::InList: {
            // value IN (a, b, c): n equality predicates OR-ed, so n/ndv, capped.
            if (pred->children.empty()) return defaults.unknown;
            const ColumnStats* cs = stats_for(*pred->children[0], stats);
            const double n = static_cast<double>(pred->children.size() - 1);
            const bool negated = (pred->expr_flags & plan::ExprFlagNegated) != 0;
            if (cs == nullptr || cs->ndv <= 0.0) {
                const double s = std::clamp(defaults.equality * n, 0.0, 1.0);
                return clamp_sel(negated ? 1.0 - s : s);
            }
            const double non_null = 1.0 - std::clamp(cs->null_fraction, 0.0, 1.0);
            const double s = std::clamp(n / cs->ndv, 0.0, 1.0) * non_null;
            return clamp_sel(negated ? non_null - s : s);
        }

        case ExprKind::UnaryOp: {
            // NOT <p>. The complement, which is only right under the two-valued
            // reading - with NULLs, NOT p and p do not partition the rows. The
            // error is bounded by the null fraction and is not worth a second
            // model until there is a measurement saying otherwise.
            if (pred->un_op == ast::UnaryOp::Not && pred->children.size() == 1) {
                return clamp_sel(1.0 - selectivity_of(pred->children[0].get(), stats, defaults));
            }
            return defaults.unknown;
        }

        default:
            // A LIKE, a function call, a CASE, a subquery. Each of these has a
            // real estimator in the literature and none of them has one here; the
            // honest answer is the model's "no idea" constant.
            return defaults.unknown;
    }
}

double distinct_groups(std::span<const plan::Expr* const> keys, double input_rows,
                       const StatisticsCatalog& stats) {
    if (keys.empty()) return 0.0;   // a scalar aggregate: the caller knows it is one row
    double product = 1.0;
    for (const plan::Expr* k : keys) {
        if (k == nullptr) return 0.0;
        const ColumnStats* cs = stats_for(*k, stats);
        // ONE unknown key makes the whole product unknown. Substituting a default
        // for it and multiplying would produce a number with a measurement on one
        // side and a guess on the other, reported as though both were measured.
        if (cs == nullptr || cs->ndv <= 0.0) return 0.0;
        product *= cs->ndv;
    }
    // Grouping cannot make more groups than it has rows.
    return input_rows > 0.0 ? std::min(product, input_rows) : product;
}


void join_key_ndv(const HashKeyVec& keys, const Schema& left, const Schema& right,
                  const StatisticsCatalog& stats, double& left_ndv, double& right_ndv) {
    left_ndv = 0.0;
    right_ndv = 0.0;
    if (keys.empty()) return;
    const HashKey& k = keys[0];
    const auto look = [&](const Schema& schema, std::uint32_t index) -> double {
        if (index >= schema.size()) return 0.0;
        const plan::ColumnSchema& c = schema[index];
        // (0, 0) is the binder's mark for a column that is not a base-table
        // column, on the same rule the predicate estimator follows.
        if (c.table_id == 0 || c.column_id == 0) return 0.0;
        const ColumnStats* cs = stats.column(c.table_id, c.column_id);
        if (cs == nullptr) return 0.0;
        // A UNIQUE column has exactly as many distinct values as the table has
        // rows, which is a guarantee rather than a sample - but the row count is
        // the statistics catalog's, not the cardinality model's, so prefer the
        // measured ndv and fall back to the table's rows only when it is absent.
        if (cs->ndv > 0.0) return cs->ndv;
        if (cs->unique) {
            const TableStats* t = stats.table(c.table_id);
            return t != nullptr ? t->rows : 0.0;
        }
        return 0.0;
    };
    left_ndv = look(left, k.left_index);
    right_ndv = look(right, k.right_index);
}

}  // namespace db25::physical
