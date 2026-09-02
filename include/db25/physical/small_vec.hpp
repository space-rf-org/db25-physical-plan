#pragma once
// A small vector with INLINE capacity, for the planner's per-candidate key lists.
//
// A join's equi-key list is one or two columns in almost every query, and a
// std::vector reaches the heap for the first one. That is not a rounding error
// here: a group-expression OWNS its keys (since Increment 3.8a, because two
// candidates in one group may join on different keys), so the cost is one
// allocation per candidate per join - measured as four on the five-group budget
// query, and multiplied by every candidate a larger query enumerates.
//
// TRIVIALLY COPYABLE ELEMENTS ONLY, and deliberately so. That removes every
// element lifetime question - no placement new, no destructor loops, no
// exception safety around a partially-constructed buffer - and leaves a container
// short enough to read in one sitting. The planner's key lists are pairs of
// column indices; nothing here wants more.
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <initializer_list>
#include <new>
#include <type_traits>

namespace db25::physical {

template <typename T, std::uint32_t N>
class SmallVec {
    static_assert(std::is_trivially_copyable_v<T>,
                  "SmallVec stores elements by memcpy and never destroys them");
    static_assert(N > 0, "an inline capacity of zero is just a vector");

public:
    SmallVec() = default;
    SmallVec(std::initializer_list<T> xs) { for (const T& x : xs) push_back(x); }
    SmallVec(const SmallVec& o) { assign(o.data(), o.size_); }
    SmallVec& operator=(const SmallVec& o) {
        if (this != &o) assign(o.data(), o.size_);
        return *this;
    }
    // Moving STEALS a heap buffer and copies an inline one. There is nothing to
    // leave behind in the inline case - the elements are trivially copyable - so
    // the moved-from object is left empty either way, which is the state every
    // caller here expects.
    SmallVec(SmallVec&& o) noexcept { take(o); }
    SmallVec& operator=(SmallVec&& o) noexcept {
        if (this != &o) { release(); take(o); }
        return *this;
    }
    ~SmallVec() { release(); }

    void push_back(const T& v) {
        if (size_ == cap_) grow(cap_ * 2);
        data()[size_++] = v;
    }
    void clear() noexcept { size_ = 0; }
    void reserve(std::uint32_t n) { if (n > cap_) grow(n); }

    [[nodiscard]] std::uint32_t size() const noexcept { return size_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0; }
    [[nodiscard]] T* data() noexcept { return heap_ != nullptr ? heap_ : inline_; }
    [[nodiscard]] const T* data() const noexcept { return heap_ != nullptr ? heap_ : inline_; }
    [[nodiscard]] T& operator[](std::size_t i) noexcept { return data()[i]; }
    [[nodiscard]] const T& operator[](std::size_t i) const noexcept { return data()[i]; }
    [[nodiscard]] T* begin() noexcept { return data(); }
    [[nodiscard]] T* end() noexcept { return data() + size_; }
    [[nodiscard]] const T* begin() const noexcept { return data(); }
    [[nodiscard]] const T* end() const noexcept { return data() + size_; }

    // Equality is BY VALUE, not by whether the elements live inline or on the
    // heap - two key lists holding the same columns are the same key list, and a
    // memo group key compares them.
    [[nodiscard]] bool operator==(const SmallVec& o) const noexcept {
        if (size_ != o.size_) return false;
        for (std::uint32_t i = 0; i < size_; ++i) {
            if (!(data()[i] == o.data()[i])) return false;
        }
        return true;
    }

private:
    void assign(const T* src, std::uint32_t n) {
        reserve(n);
        if (n != 0) std::memcpy(data(), src, static_cast<std::size_t>(n) * sizeof(T));
        size_ = n;
    }
    void take(SmallVec& o) noexcept {
        if (o.heap_ != nullptr) {
            heap_ = o.heap_;
            cap_ = o.cap_;
            o.heap_ = nullptr;
            o.cap_ = N;
        } else if (o.size_ != 0) {
            std::memcpy(inline_, o.inline_, static_cast<std::size_t>(o.size_) * sizeof(T));
        }
        size_ = o.size_;
        o.size_ = 0;
    }
    void release() noexcept {
        if (heap_ != nullptr) { ::operator delete(heap_); heap_ = nullptr; cap_ = N; }
    }
    // ::operator new, NOT malloc. The allocation budget test counts heap
    // allocations by replacing global operator new, and a container that reached
    // the heap through malloc would spend allocations this project measures
    // without them appearing in the measurement - which is worse than the
    // allocations themselves. (Caught by a mutation: bypassing the inline buffer
    // entirely changed the budget by ZERO while malloc was used.)
    void grow(std::uint32_t want) {
        const std::uint32_t next = want > N ? want : N + 1;
        T* buf = static_cast<T*>(::operator new(static_cast<std::size_t>(next) * sizeof(T)));
        if (size_ != 0) std::memcpy(buf, data(), static_cast<std::size_t>(size_) * sizeof(T));
        if (heap_ != nullptr) ::operator delete(heap_);
        heap_ = buf;
        cap_ = next;
    }

    T* heap_ = nullptr;          // null while the elements live inline
    std::uint32_t size_ = 0;
    std::uint32_t cap_ = N;
    T inline_[N]{};
};

}  // namespace db25::physical
