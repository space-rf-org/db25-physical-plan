#include "db25/physical/pipeline.hpp"

#include <cstddef>

namespace db25::physical {

const char* edge_kind_to_string(EdgeKind k) noexcept {
    switch (k) {
        case EdgeKind::Streaming:    return "streaming";
        case EdgeKind::Materialized: return "materialized";
        case EdgeKind::Rescanned:    return "rescanned";
        case EdgeKind::Separate:     return "separate";
    }
    return "?";
}

EdgeKind edge_kind(PhysicalOp op, std::size_t index, bool build_right) noexcept {
    switch (op) {
        // Leaves have no inputs; the question does not arise. Answering
        // Streaming rather than asserting keeps callers from having to check
        // arity first, and no edge to a leaf exists to be misread.
        case PhysicalOp::SeqScan:
        case PhysicalOp::ValuesScan:
        case PhysicalOp::WorkingTableScan:
            return EdgeKind::Streaming;

        // One row in, one row out, no state kept across rows.
        case PhysicalOp::Filter:
        case PhysicalOp::Project:
        case PhysicalOp::FormatConvert:
        case PhysicalOp::Limit:
            return EdgeKind::Streaming;

        // Streaming, and this is exactly what makes them worth the sort they
        // require: a streaming aggregate emits a group the moment it closes and a
        // streaming distinct emits a row the moment it knows it is not a
        // duplicate. Neither holds the input. If they did they would be their
        // hashing counterparts with extra steps.
        case PhysicalOp::StreamingAggregate:
        case PhysicalOp::StreamingDistinct:
            return EdgeKind::Streaming;

        // A window function reads its input in order and appends columns to the
        // rows it passes on. It buffers WITHIN a partition when a frame looks
        // forward - `ROWS BETWEEN CURRENT ROW AND UNBOUNDED FOLLOWING` cannot
        // answer until the partition ends - but it never holds the whole input,
        // so the pipeline runs through it. The per-partition buffer is real and
        // is an executor's business, not a plan boundary.
        case PhysicalOp::Window:
            return EdgeKind::Streaming;

        // The write path streams: rows arrive one at a time and are written one
        // at a time. Each is a pipeline SINK - nothing consumes its output in a
        // loop - but that is a fact about its output, not about this edge.
        case PhysicalOp::CreateTableAs:
        case PhysicalOp::Insert:
        case PhysicalOp::Update:
        case PhysicalOp::Delete:
            return EdgeKind::Streaming;

        // A sort cannot emit its first row until it has seen its last.
        case PhysicalOp::Sort:
            return EdgeKind::Materialized;

        // A hash table on every input row, and no output until it is built.
        case PhysicalOp::HashAggregate:
        case PhysicalOp::HashGroupingSets:
        case PhysicalOp::HashDistinct:
            return EdgeKind::Materialized;

        // THE ASYMMETRY. One side is built into a hash table and one is probed
        // against it, and WHICH is a per-plan decision the search makes - so this
        // answer depends on the candidate, not only on the operator. It is the
        // reason this function is per-edge rather than per-operator, and the
        // reason the spec's one-label-per-operator `kind` cannot replace it.
        case PhysicalOp::HashJoin:
            return (index == 1) == build_right ? EdgeKind::Materialized
                                               : EdgeKind::Streaming;

        // Semi and anti joins build from the RIGHT and probe with the left -
        // fixed, not chosen, because swapping them computes a different relation
        // (see is_build_side_choosable).
        case PhysicalOp::HashSemiJoin:
        case PhysicalOp::HashAntiJoin:
            return index == 1 ? EdgeKind::Materialized : EdgeKind::Streaming;

        // HashSetOp builds from the right and probes with the left, on the same
        // fixed footing. UnionAll builds nothing at all: it is two loops feeding
        // one consumer, which is Separate rather than Materialized - telling a
        // reader a concatenation allocates would be a lie about where the memory
        // goes.
        case PhysicalOp::HashSetOp:
            return index == 1 ? EdgeKind::Materialized : EdgeKind::Streaming;
        case PhysicalOp::UnionAll:
            return EdgeKind::Separate;

        // A merge join consumes both inputs in key order, buffering neither. But
        // they are two loops advancing against each other, not one loop with a
        // side input, so the LEFT drives the pipeline and the right is its own.
        // Separate, not Materialized: a merge join's whole point is that it
        // allocates nothing.
        case PhysicalOp::MergeJoin:
            return index == 0 ? EdgeKind::Streaming : EdgeKind::Separate;

        // The nested loops re-execute their right input once per left row. Not
        // buffered - recomputed - which is both why they are quadratic and why
        // they are the only implementation that can serve a correlated LATERAL
        // right side.
        case PhysicalOp::NestedLoopJoin:
        case PhysicalOp::NestedLoopSemiJoin:
        case PhysicalOp::NestedLoopAntiJoin:
            return index == 0 ? EdgeKind::Streaming : EdgeKind::Rescanned;

        // The anchor runs once and streams; the recursive term runs once per
        // iteration, which is the same relationship a nested loop has with its
        // right input and is recorded the same way.
        case PhysicalOp::RecursiveFixpoint:
            return index == 0 ? EdgeKind::Streaming : EdgeKind::Rescanned;

        // An exchange is where a pipeline ENDS and another begins: the sending
        // side is one loop and the receiving side another, on a different node.
        // Nothing is buffered - rows go out as they are produced - so this is
        // `separate` rather than `materialized`, the same distinction a merge
        // join's right input gets.
        case PhysicalOp::Exchange:
            return EdgeKind::Separate;
    }
    return EdgeKind::Streaming;
}

namespace {

// Walk a pipeline downward from `n`, collecting the operators a row passes
// through and finding where its rows come from. Children reached over a
// non-streaming edge belong to other pipelines and are not entered here.
void collect_members(const PhysicalNode& n, Pipeline& p) {
    for (std::size_t i = 0; i < n.children.size(); ++i) {
        if (edge_kind(n.op, i, n.build_right) != EdgeKind::Streaming) continue;
        collect_members(*n.children[i], p);
    }
    // After the children, so `members` reads from the source upward - the order
    // a row actually travels, and the order an executor would push it.
    p.members.push_back(&n);
    if (p.source == nullptr) p.source = &n;
}

// Every node that ENDS a pipeline: the plan root, and every child reached over a
// non-streaming edge.
//
// POST-ORDER, and that is not only for determinism. A pipeline's dependencies -
// the pipelines whose output it consumes - are rooted deeper in the tree than it
// is, so visiting children first emits every pipeline AFTER the ones it needs.
// The resulting order is therefore a legal execution schedule, not merely a
// stable numbering: a hash join's build pipeline comes out before the probe
// pipeline that reads its table.
void collect_sinks(const PhysicalNode& n, std::vector<const PhysicalNode*>& out) {
    for (std::size_t i = 0; i < n.children.size(); ++i) {
        const PhysicalNode& c = *n.children[i];
        collect_sinks(c, out);
        if (edge_kind(n.op, i, n.build_right) != EdgeKind::Streaming) out.push_back(&c);
    }
}

}  // namespace

std::vector<Pipeline> pipelines(const PhysicalNode& root) {
    std::vector<const PhysicalNode*> sinks;
    collect_sinks(root, sinks);
    sinks.push_back(&root);  // the root's own pipeline is the last one to run

    std::vector<Pipeline> out;
    out.reserve(sinks.size());
    for (const PhysicalNode* s : sinks) {
        Pipeline p;
        p.sink = s;
        collect_members(*s, p);
        out.push_back(std::move(p));
    }
    return out;
}

std::string pipelines_to_string(const PhysicalNode& root) {
    std::string out;
    const std::vector<Pipeline> ps = pipelines(root);
    for (std::size_t i = 0; i < ps.size(); ++i) {
        out += "#" + std::to_string(i) + " ";
        for (std::size_t j = 0; j < ps[i].members.size(); ++j) {
            if (j != 0) out += " -> ";
            out += physical_op_to_string(ps[i].members[j]->op);
        }
        out += "\n";
    }
    return out;
}

}  // namespace db25::physical
