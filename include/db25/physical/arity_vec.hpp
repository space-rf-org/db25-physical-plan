#pragma once
// A fixed-capacity sequence for the planner's per-operand data.
//
// Every operator in the physical IR is a leaf, unary, or binary - expected_arity()
// is the authority and returns at most 2. So the sequences indexed BY OPERAND
// (an operator's inputs, what it requires of each, what each provided) have a
// compile-time bound, and reaching the heap for one or two elements is pure waste:
// it was measured as a per-candidate and per-goal allocation, multiplied by
// however hard the search worked.
//
// Capacity is deliberately NOT silently exceeded. Overflow is a contract
// violation - a new operator with arity 3 - not a resize, so it asserts in debug
// and is clamped (never overrun) in release. kArityCap is checked against
// expected_arity() by a test, so the two cannot drift apart unnoticed.
#include <cassert>
#include <cstddef>
#include <vector>

namespace db25::physical {

inline constexpr std::size_t kArityCap = 2;

template <typename T>
class ArityVec {
public:
    ArityVec() = default;
    ArityVec(const std::vector<T>& v) { assign(v); }  // NOLINT: implicit is intended

    void assign(const std::vector<T>& v) {
        size_ = 0;
        for (const T& t : v) push_back(t);
    }
    void push_back(const T& v) {
        assert(size_ < kArityCap && "operator arity exceeds kArityCap");
        if (size_ >= kArityCap) return;  // release: never overrun
        data_[size_++] = v;
    }
    void clear() noexcept { size_ = 0; }

    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] T& operator[](std::size_t i) noexcept { return data_[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return data_[i]; }
    [[nodiscard]] T* begin() noexcept { return data_; }
    [[nodiscard]] T* end() noexcept { return data_ + size_; }
    [[nodiscard]] const T* begin() const noexcept { return data_; }
    [[nodiscard]] const T* end() const noexcept { return data_ + size_; }
    [[nodiscard]] const T* data() const noexcept { return data_; }

    [[nodiscard]] bool operator==(const ArityVec& o) const {
        if (size_ != o.size_) return false;
        for (std::size_t i = 0; i < size_; ++i) {
            if (!(data_[i] == o.data_[i])) return false;
        }
        return true;
    }

private:
    T data_[kArityCap]{};
    std::size_t size_ = 0;
};

}  // namespace db25::physical
