#include "db25/physical/spec.hpp"

#include "db25/physical/pipeline.hpp"

#include "db25/physical/sexpr_read.hpp"

#include <charconv>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace db25::physical {

bool CapabilityProfile::can_execute(const std::string& op) const {
    for (const std::string& e : executes) {
        if (e == op) return true;
    }
    return false;
}

const OperatorSpec* PhysicalSpec::find_operator(const std::string& name) const {
    for (const OperatorSpec& o : operators) {
        if (o.name == name) return &o;
    }
    return nullptr;
}

std::vector<std::string> PhysicalSpec::physicals_for_logical(const std::string& logical) const {
    std::vector<std::string> out;
    for (const ImplRule& r : impl_rules) {
        if (r.logical == logical) out.push_back(r.physical);
    }
    return out;
}

std::span<const PhysicalOp> PhysicalSpec::ops_for_logical(const std::string& logical) const {
    const auto it = resolved_rules.find(logical);
    if (it == resolved_rules.end()) return {};
    return it->second;
}

std::optional<PhysicalSpec> parse_spec(const std::string& text, std::string& error) {
    SNode root;
    if (!read_sexpr(text, root, error)) return std::nullopt;
    if (!root.is_list || root.head() != "physical-spec") {
        error = "top-level form must be (physical-spec ...)";
        return std::nullopt;
    }

    PhysicalSpec spec;

    if (const SNode* v = root.child("version")) {
        const std::string s = value_of(*v);
        int parsed = -1;
        const auto res = std::from_chars(s.data(), s.data() + s.size(), parsed);
        if (res.ec != std::errc{}) { error = "version is not an integer: '" + s + "'"; return std::nullopt; }
        spec.version = parsed;
    }

    if (const SNode* ops = root.child("operators")) {
        for (const SNode& op : ops->list) {
            if (!op.is_list || op.head() != "operator") continue;  // skip the 'operators' head atom
            OperatorSpec os;
            if (const SNode* n = op.child("name")) os.name = value_of(*n);
            if (const SNode* a = op.child("arity")) {
                const std::string s = value_of(*a);
                unsigned long parsed = 0;
                const auto res = std::from_chars(s.data(), s.data() + s.size(), parsed);
                if (res.ec != std::errc{}) { error = "arity is not an integer: '" + s + "'"; return std::nullopt; }
                os.arity = static_cast<std::size_t>(parsed);
            }
            if (const SNode* k = op.child("kind")) os.kind = value_of(*k);
            if (const SNode* e = op.child("edges")) {
                for (std::size_t i = 1; i < e->list.size(); ++i) {
                    if (!e->list[i].is_list) os.edges.push_back(e->list[i].atom);
                }
            }
            if (os.name.empty()) { error = "an operator has no name"; return std::nullopt; }
            spec.operators.push_back(std::move(os));
        }
    }

    if (const SNode* rules = root.child("implementation-rules")) {
        for (const SNode& r : rules->list) {
            if (!r.is_list || r.head() != "rule") continue;  // skip the head atom
            ImplRule ir;
            if (const SNode* l = r.child("logical")) ir.logical = value_of(*l);
            if (const SNode* p = r.child("physical")) ir.physical = value_of(*p);
            if (ir.logical.empty() || ir.physical.empty()) {
                error = "an implementation rule is missing its logical or physical operator";
                return std::nullopt;
            }
            spec.impl_rules.push_back(std::move(ir));
        }
    }

    if (const SNode* prof = root.child("capability-profile")) {
        if (const SNode* n = prof->child("name")) spec.profile.name = value_of(*n);
        if (const SNode* ex = prof->child("executes")) {
            // (executes A B C ...) - every atom after the head is an operator name.
            for (std::size_t i = 1; i < ex->list.size(); ++i) {
                if (!ex->list[i].is_list) spec.profile.executes.push_back(ex->list[i].atom);
            }
        }
    }

    if (const SNode* b = root.child("search-budget")) {
        if (const SNode* n = b->child("max-join-count")) {
            const std::string v = value_of(*n);
            unsigned long parsed = spec.budget.max_join_count;
            if (std::from_chars(v.data(), v.data() + v.size(), parsed).ec == std::errc{}) {
                spec.budget.max_join_count = static_cast<std::uint32_t>(parsed);
            }
        }
    }

    // Resolve the implementation rules to operators once, here, rather than on
    // every logical node the planner lowers.
    for (const ImplRule& r : spec.impl_rules) {
        if (const std::optional<PhysicalOp> op = physical_op_from_name(r.physical)) {
            spec.resolved_rules[r.logical].push_back(*op);
        }
    }

    return spec;
}

std::optional<PhysicalSpec> load_spec(const std::string& path, std::string& error) {
    std::ifstream in(path, std::ios::binary);
    if (!in) { error = "cannot open spec file: " + path; return std::nullopt; }
    std::ostringstream ss;
    ss << in.rdbuf();
    return parse_spec(ss.str(), error);
}

std::vector<std::string> check_conformance(const PhysicalSpec& spec) {
    std::vector<std::string> problems;

    if (spec.version < 0) {
        problems.push_back("spec has no version");
    }

    // Every operator the planner can emit must be declared, with the IR's arity,
    // and be executable by the reference profile.
    for (const PhysicalOp op : kAllPhysicalOps) {
        const std::string name = physical_op_to_string(op);
        const OperatorSpec* os = spec.find_operator(name);
        if (os == nullptr) {
            problems.push_back("operator '" + name +
                               "' is emittable by the planner but not declared in the spec");
        } else if (os->arity != expected_arity(op)) {
            problems.push_back("operator '" + name + "' arity mismatch: spec " +
                               std::to_string(os->arity) + " vs IR " +
                               std::to_string(expected_arity(op)));
        } else {
            // Where rows stop, declared per INPUT and checked against the code.
            // The build side of a hash-family operator is a per-PLAN decision, so
            // the spec declares the build-right shape and edge_kind() is asked for
            // that shape here; the flip is a code-level fact a test pins.
            if (os->edges.size() != expected_arity(op)) {
                problems.push_back("operator '" + name + "' declares " +
                                   std::to_string(os->edges.size()) +
                                   " edge kind(s) for " + std::to_string(expected_arity(op)) +
                                   " input(s)");
            } else {
                bool materializes = false;
                for (std::size_t i = 0; i < os->edges.size(); ++i) {
                    const std::string got = edge_kind_to_string(edge_kind(op, i, true));
                    if (got != os->edges[i]) {
                        problems.push_back("operator '" + name + "' input " +
                                           std::to_string(i) + ": spec says '" +
                                           os->edges[i] + "', the IR says '" + got + "'");
                    }
                    materializes = materializes || os->edges[i] == "materialized";
                }
                // `kind` is the summary `edges` implies, not an independent claim.
                // An operator that buffers an input is a breaker; one that does
                // not is not, however else its inputs are consumed.
                const std::string implied =
                    expected_arity(op) == 0 ? "access-path"
                                            : (materializes ? "pipeline-breaker" : "pipeline");
                if (os->kind != implied) {
                    problems.push_back("operator '" + name + "' kind '" + os->kind +
                                       "' contradicts its edges, which imply '" + implied + "'");
                }
            }
        }
        if (!spec.profile.can_execute(name)) {
            problems.push_back("operator '" + name +
                               "' is not marked executable by capability profile '" +
                               spec.profile.name + "'");
        }
    }

    // The spec must not declare an operator the IR does not know.
    for (const OperatorSpec& os : spec.operators) {
        bool known = false;
        for (const PhysicalOp op : kAllPhysicalOps) {
            if (os.name == physical_op_to_string(op)) { known = true; break; }
        }
        if (!known) {
            problems.push_back("spec declares operator '" + os.name +
                               "' that the IR does not know");
        }
    }

    // The profile must only claim to execute declared operators.
    for (const std::string& e : spec.profile.executes) {
        if (spec.find_operator(e) == nullptr) {
            problems.push_back("capability profile executes undeclared operator '" + e + "'");
        }
    }

    // Every implementation rule must target a declared, executable operator (its
    // logical side is validated against the logical IR at lowering time).
    for (const ImplRule& r : spec.impl_rules) {
        if (spec.find_operator(r.physical) == nullptr) {
            problems.push_back("implementation rule for '" + r.logical +
                               "' targets undeclared physical operator '" + r.physical + "'");
        } else if (!spec.profile.can_execute(r.physical)) {
            problems.push_back("implementation rule for '" + r.logical +
                               "' targets non-executable operator '" + r.physical + "'");
        }
    }

    return problems;
}

}  // namespace db25::physical
