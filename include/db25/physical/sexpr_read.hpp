#pragma once
// A minimal, zero-dependency s-expression reader shared by the spec loader and
// the calibration loader. Nodes are either an atom (a bare token) or a list
// (parenthesized sequence); `;` begins a line comment. This is deliberately not
// a general Lisp reader - no quoting, no strings with embedded spaces - only what
// the DB25 physical-plan IDL and calibration files need.
#include <string>
#include <vector>

namespace db25::physical {

struct SNode {
    bool is_list = false;
    std::string atom;         // when !is_list
    std::vector<SNode> list;  // when is_list

    // The head atom of a list ( (head ...) ), or empty.
    [[nodiscard]] const std::string& head() const;
    // First child list whose head atom == name, or nullptr.
    [[nodiscard]] const SNode* child(const std::string& name) const;
};

// Parse a single top-level s-expression from `text`. Returns false and writes a
// message to `error` on malformed input (unterminated list, stray ')', trailing
// content, empty input).
[[nodiscard]] bool read_sexpr(const std::string& text, SNode& out, std::string& error);

// The second child atom of a (key value) pair, or empty (a convenience the
// loaders share).
[[nodiscard]] std::string value_of(const SNode& kv);

}  // namespace db25::physical
