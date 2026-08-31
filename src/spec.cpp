#include "db25/physical/spec.hpp"

#include <charconv>
#include <cstddef>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace db25::physical {
namespace {

// ---- a minimal, zero-dependency s-expression reader ----------------------
// Nodes are either an atom (a bare token) or a list (parenthesized sequence).
// `;` begins a line comment. This is all the IDL needs; it is deliberately not a
// general Lisp reader (no quoting, no strings with spaces).
struct SNode {
    bool is_list = false;
    std::string atom;            // when !is_list
    std::vector<SNode> list;     // when is_list

    [[nodiscard]] const std::string& head() const {
        static const std::string empty;
        return (is_list && !list.empty() && !list[0].is_list) ? list[0].atom : empty;
    }
    // First child list whose head atom == name, or nullptr.
    [[nodiscard]] const SNode* child(const std::string& name) const {
        if (!is_list) return nullptr;
        for (const SNode& c : list) {
            if (c.is_list && c.head() == name) return &c;
        }
        return nullptr;
    }
};

class Reader {
public:
    explicit Reader(const std::string& text) : text_(text) {}

    // Parse a single top-level s-expression. Returns false + error on failure.
    bool parse(SNode& out, std::string& error) {
        skip_ws();
        if (pos_ >= text_.size()) { error = "empty spec"; return false; }
        if (!read_node(out, error)) return false;
        skip_ws();
        if (pos_ < text_.size()) { error = "trailing content after top-level form"; return false; }
        return true;
    }

private:
    const std::string& text_;
    std::size_t pos_ = 0;

    void skip_ws() {
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ';') {                       // line comment to end of line
                while (pos_ < text_.size() && text_[pos_] != '\n') ++pos_;
            } else if (c == ' ' || c == '\t' || c == '\r' || c == '\n') {
                ++pos_;
            } else {
                break;
            }
        }
    }

    bool read_node(SNode& out, std::string& error) {
        skip_ws();
        if (pos_ >= text_.size()) { error = "unexpected end of input"; return false; }
        if (text_[pos_] == '(') {
            ++pos_;  // consume '('
            out.is_list = true;
            while (true) {
                skip_ws();
                if (pos_ >= text_.size()) { error = "unterminated list"; return false; }
                if (text_[pos_] == ')') { ++pos_; return true; }
                SNode child;
                if (!read_node(child, error)) return false;
                out.list.push_back(std::move(child));
            }
        }
        if (text_[pos_] == ')') { error = "unexpected ')'"; return false; }
        // atom: read until whitespace, paren, or comment
        const std::size_t start = pos_;
        while (pos_ < text_.size()) {
            const char c = text_[pos_];
            if (c == ' ' || c == '\t' || c == '\r' || c == '\n' || c == '(' ||
                c == ')' || c == ';') {
                break;
            }
            ++pos_;
        }
        out.is_list = false;
        out.atom = text_.substr(start, pos_ - start);
        return true;
    }
};

// Second child atom of `(key value)`, or empty.
std::string value_of(const SNode& kv) {
    return (kv.is_list && kv.list.size() >= 2 && !kv.list[1].is_list) ? kv.list[1].atom
                                                                      : std::string{};
}

}  // namespace

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

std::optional<PhysicalSpec> parse_spec(const std::string& text, std::string& error) {
    SNode root;
    if (!Reader(text).parse(root, error)) return std::nullopt;
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
            if (os.name.empty()) { error = "an operator has no name"; return std::nullopt; }
            spec.operators.push_back(std::move(os));
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

    return problems;
}

}  // namespace db25::physical
