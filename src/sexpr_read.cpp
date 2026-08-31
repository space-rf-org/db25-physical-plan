#include "db25/physical/sexpr_read.hpp"

#include <cstddef>

namespace db25::physical {

const std::string& SNode::head() const {
    static const std::string empty;
    return (is_list && !list.empty() && !list[0].is_list) ? list[0].atom : empty;
}

const SNode* SNode::child(const std::string& name) const {
    if (!is_list) return nullptr;
    for (const SNode& c : list) {
        if (c.is_list && c.head() == name) return &c;
    }
    return nullptr;
}

std::string value_of(const SNode& kv) {
    return (kv.is_list && kv.list.size() >= 2 && !kv.list[1].is_list) ? kv.list[1].atom
                                                                      : std::string{};
}

namespace {

class Reader {
public:
    explicit Reader(const std::string& text) : text_(text) {}

    bool parse(SNode& out, std::string& error) {
        skip_ws();
        if (pos_ >= text_.size()) { error = "empty input"; return false; }
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
            if (c == ';') {  // line comment to end of line
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

}  // namespace

bool read_sexpr(const std::string& text, SNode& out, std::string& error) {
    return Reader(text).parse(out, error);
}

}  // namespace db25::physical
