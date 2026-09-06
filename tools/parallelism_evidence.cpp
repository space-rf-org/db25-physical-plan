// The measurements behind "parallelism is declared, not costed" (see spec.hpp).
//
// Not a test - tests/test_parallelism.cpp holds the guards. This regenerates the
// NUMBERS, so the decision can be re-examined rather than taken on trust, and so
// a reader who doubts it can run the thing that produced it.
//
// The question was whether a work-span cost model - carrying total work AND
// critical path, rather than one scalar - would ever RANK two plans differently
// from the scalar model. Under Amdahl a plan's elapsed time is W*(s + (1-s)/P),
// so the costlier plan W2 overtakes W1 exactly when W2*s2 < W1*s1, i.e. when
//
//        s2 / s1  <  W1 / W2
//
// The right-hand side is a cost ratio this model can measure. It IS the
// break-even: how much more parallel the loser would have to be. Nothing below
// assumes a serial fraction for any operator.
#include "db25/physical/cost.hpp"
#include "db25/physical/physical_plan.hpp"

#include <cstdio>
#include <memory>
#include <string>

using namespace db25::physical;
using db25::ast::DataType;

static Schema cols(const std::string& t, int n) {
    Schema s;
    for (int i = 0; i < n; ++i)
        s.push_back({t + std::to_string(i), DataType::Integer, false, 0,
                     static_cast<std::uint32_t>(i), t, false});
    return s;
}
static PhysicalNodePtr nd(PhysicalOp o) { return std::make_unique<PhysicalNode>(o); }
static PhysicalNodePtr scan(const std::string& t, StorageFormat f = StorageFormat::Row) {
    auto s = nd(PhysicalOp::SeqScan);
    s->table_name = t; s->output = cols(t, 4); s->scan_format = f;
    return s;
}
static PhysicalNodePtr wrap(PhysicalOp o, PhysicalNodePtr in) {
    auto x = nd(o); x->output = in->output; x->children.push_back(std::move(in));
    return x;
}
static PhysicalNodePtr sorted(const std::string& t) {
    auto s = wrap(PhysicalOp::Sort, scan(t)); s->sort_keys = {SortKey{0}}; return s;
}

static CalibrationProfile cal = default_calibration();
static CardinalityModel card;

static void row(const char* what, const char* an, PhysicalNodePtr a,
                const char* bn, PhysicalNodePtr b) {
    const double wa = cost_of(*a, cal, card), wb = cost_of(*b, cal, card);
    const bool first = wa <= wb;
    const double r = first ? wb / wa : wa / wb;
    std::printf("  %-32s %-24s %-24s %8.2fx  %s\n", what, first ? an : bn,
                first ? bn : an, r, r < 1.35 ? "close" : "");
}

int main() {
    card.base_rows["big"] = 20000000.0;
    card.base_rows["mid"] = 2000000.0;
    card.base_rows["p"] = 2000000.0;
    card.base_rows["q"] = 2200000.0;

    std::printf("BREAK-EVEN: how much more parallel must the loser be to overtake?\n\n");
    std::printf("  %-32s %-24s %-24s %9s\n", "decision", "winner", "loser", "break-even");
    std::printf("  %s\n", std::string(96, '-').c_str());
    {
        auto hj = nd(PhysicalOp::HashJoin); hj->hash_keys.push_back(HashKey{0, 0});
        hj->output = cols("big", 4);
        hj->children.push_back(scan("big")); hj->children.push_back(scan("mid"));
        auto mj = nd(PhysicalOp::MergeJoin); mj->hash_keys.push_back(HashKey{0, 0});
        mj->output = cols("big", 4);
        mj->children.push_back(sorted("big")); mj->children.push_back(sorted("mid"));
        row("join algorithm", "HashJoin", std::move(hj), "Sort+MergeJoin", std::move(mj));
    }
    row("aggregate algorithm", "HashAggregate",
        wrap(PhysicalOp::HashAggregate, scan("big")), "Sort+StreamingAggregate",
        wrap(PhysicalOp::StreamingAggregate, sorted("big")));
    row("distinct algorithm", "HashDistinct", wrap(PhysicalOp::HashDistinct, scan("big")),
        "Sort+StreamingDistinct", wrap(PhysicalOp::StreamingDistinct, sorted("big")));
    row("aggregate, input already sorted", "StreamingAggregate",
        wrap(PhysicalOp::StreamingAggregate, scan("big")), "HashAggregate",
        wrap(PhysicalOp::HashAggregate, scan("big")));
    row("scan substrate", "Column scan", scan("big", StorageFormat::Column),
        "Row scan", scan("big"));
    {
        auto l = nd(PhysicalOp::HashJoin); l->hash_keys.push_back(HashKey{0, 0});
        l->output = cols("p", 4); l->build_right = false;
        l->children.push_back(scan("p")); l->children.push_back(scan("q"));
        auto r = nd(PhysicalOp::HashJoin); r->hash_keys.push_back(HashKey{0, 0});
        r->output = cols("p", 4); r->build_right = true;
        r->children.push_back(scan("p")); r->children.push_back(scan("q"));
        row("hash join build side", "build left", std::move(l), "build right", std::move(r));
    }

    std::printf("\nTHE SORT GAP IS THE COMPLEXITY CLASS, NOT A COEFFICIENT.\n");
    std::printf("A sort is n*log(n) and an aggregate is n, so the gap never closes and\n");
    std::printf("WIDENS with data - no recalibration of the coefficients can reach it.\n\n");
    std::printf("  %14s %12s\n", "rows", "break-even");
    for (const double n : {10.0, 100.0, 1000.0, 100000.0, 20000000.0, 1000000000.0}) {
        CardinalityModel c; c.base_rows["t"] = n;
        auto ha = wrap(PhysicalOp::HashAggregate, scan("t"));
        auto so = wrap(PhysicalOp::Sort, scan("t")); so->sort_keys = {SortKey{0}};
        auto sa = wrap(PhysicalOp::StreamingAggregate, std::move(so));
        std::printf("  %14.0f %11.2fx\n", n, cost_of(*sa, cal, c) / cost_of(*ha, cal, c));
    }
    return 0;
}
