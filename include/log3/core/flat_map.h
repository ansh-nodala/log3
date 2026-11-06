// log3/core/flat_map.h
//
// FlatMap<K, V, Hasher>: high-performance, open-addressed flat hash table with
// linear probing, metadata control bytes, and backward-shift deletion.
//
// Why not std::unordered_map:
//   * std::unordered_map allocates a separate heap node for EVERY inserted
//     element, destroying CPU cache locality and fragmenting the heap.
//   * FlatMap stores all control bytes and key-value slots in a single
//     contiguous allocation from log3's Allocator (or Arena!).
//   * Uses a compact 1-byte control tag per slot:
//       - 0x00        : kEmpty (slot has never been occupied)
//       - 0x80..0xFF  : Occupied (low 7 bits store the top 7 bits of the 64-bit
//                       hash as a fast fingerprint, avoiding 99% of full key
//                       comparisons during probing)
//   * Backward-shift deletion eliminates tombstones completely, so lookup
//     performance never degrades over long-running inference/training sessions
//     with frequent insertions and erasures.
//   * Move-only, no exceptions, returns Status (kOutOfMemory / kOutOfRange) on
//     every fallible operation, leaving the map unchanged on failure.
//
// Usage:
//   FlatMap<StrView, Index> tensor_index;
//   L3_RETURN_IF_ERROR(tensor_index.put("model.layers.0.attn.wq.weight", 0));
//   if (const Index* idx = tensor_index.find("model.layers.0.attn.wq.weight")) {
//     ...
//   }

#pragma once

#include <new>
#include <type_traits>
#include <utility>

#include "log3/core/assert.h"
#include "log3/core/memory.h"
#include "log3/core/platform.h"
#include "log3/core/status.h"
#include "log3/core/str_view.h"
#include "log3/core/types.h"

namespace l3 {

template <typename K, typename V, typename Hasher = Hash<K>>
class FlatMap {
  static_assert(!std::is_const_v<K> && !std::is_reference_v<K>,
                "FlatMap: K must be a non-const object type.");
  static_assert(!std::is_const_v<V> && !std::is_reference_v<V>,
                "FlatMap: V must be a non-const object type.");
  static_assert(std::is_nothrow_move_constructible_v<K> &&
                    std::is_nothrow_move_constructible_v<V>,
                "FlatMap: K and V must be nothrow move-constructible.");
  static_assert(std::is_nothrow_destructible_v<K> &&
                    std::is_nothrow_destructible_v<V>,
                "FlatMap: K and V must be nothrow destructible.");

 public:
  struct Entry {
    K key;
    V value;
  };

  // Forward iterator over occupied entries.
  template <bool IsConst>
  class IteratorImpl {
   public:
    using EntryType = std::conditional_t<IsConst, const Entry, Entry>;
    using CtrlPtr = const u8*;
    using SlotPtr = EntryType*;

    constexpr IteratorImpl() noexcept
        : ctrl_(nullptr), slots_(nullptr), index_(0), capacity_(0) {}

    constexpr IteratorImpl(CtrlPtr ctrl, SlotPtr slots, Index index,
                           Index capacity) noexcept
        : ctrl_(ctrl), slots_(slots), index_(index), capacity_(capacity) {
      advance_to_occupied();
    }

    EntryType& operator*() const noexcept {
      L3_DCHECK_LT(index_, capacity_);
      return slots_[index_];
    }

    EntryType* operator->() const noexcept {
      L3_DCHECK_LT(index_, capacity_);
      return &slots_[index_];
    }

    IteratorImpl& operator++() noexcept {
      ++index_;
      advance_to_occupied();
      return *this;
    }

    IteratorImpl operator++(int) noexcept {
      IteratorImpl tmp = *this;
      ++(*this);
      return tmp;
    }

    friend bool operator==(const IteratorImpl& a,
                           const IteratorImpl& b) noexcept {
      return a.slots_ == b.slots_ && a.index_ == b.index_;
    }

    friend bool operator!=(const IteratorImpl& a,
                           const IteratorImpl& b) noexcept {
      return !(a == b);
    }

   private:
    constexpr void advance_to_occupied() noexcept {
      while (index_ < capacity_ && (ctrl_[index_] & kOccupiedBit) == 0) {
        ++index_;
      }
    }

    CtrlPtr ctrl_;
    SlotPtr slots_;
    Index index_;
    Index capacity_;
  };

  using iterator = IteratorImpl<false>;
  using const_iterator = IteratorImpl<true>;

  // --- construction / destruction --------------------------------------------
  FlatMap() noexcept
      : ctrl_(nullptr),
        slots_(nullptr),
        size_(0),
        capacity_(0),
        allocator_(nullptr) {}

  explicit FlatMap(const Allocator& allocator) noexcept
      : ctrl_(nullptr),
        slots_(nullptr),
        size_(0),
        capacity_(0),
        allocator_(&allocator) {}

  FlatMap(FlatMap&& other) noexcept
      : ctrl_(other.ctrl_),
        slots_(other.slots_),
        size_(other.size_),
        capacity_(other.capacity_),
        allocator_(other.allocator_) {
    other.ctrl_ = nullptr;
    other.slots_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
  }

  FlatMap& operator=(FlatMap&& other) noexcept {
    if (this != &other) {
      clear();
      free_table();
      ctrl_ = other.ctrl_;
      slots_ = other.slots_;
      size_ = other.size_;
      capacity_ = other.capacity_;
      allocator_ = other.allocator_;

      other.ctrl_ = nullptr;
      other.slots_ = nullptr;
      other.size_ = 0;
      other.capacity_ = 0;
    }
    return *this;
  }

  FlatMap(const FlatMap&) = delete;
  FlatMap& operator=(const FlatMap&) = delete;

  ~FlatMap() {
    clear();
    free_table();
  }

  // --- observers -------------------------------------------------------------
  Index size() const noexcept { return size_; }
  Index capacity() const noexcept { return capacity_; }
  bool empty() const noexcept { return size_ == 0; }
  const Allocator& get_allocator() const noexcept { return allocator(); }

  iterator begin() noexcept { return iterator(ctrl_, slots_, 0, capacity_); }
  iterator end() noexcept {
    return iterator(ctrl_, slots_, capacity_, capacity_);
  }
  const_iterator begin() const noexcept {
    return const_iterator(ctrl_, slots_, 0, capacity_);
  }
  const_iterator end() const noexcept {
    return const_iterator(ctrl_, slots_, capacity_, capacity_);
  }

  // --- capacity management ---------------------------------------------------
  // Pre-allocate space for at least `min_entries` without rehashing.
  // Keeps load factor <= 75% (capacity >= ceil(min_entries * 4 / 3)).
  Status reserve(Index min_entries) noexcept {
    L3_DCHECK_GE(min_entries, 0);
    if (min_entries <= max_load(capacity_)) return Status::ok();

    Index target_cap = kMinCapacity;
    while (max_load(target_cap) < min_entries) {
      if (target_cap > kMaxIndex / 2) {
        return Status::error(StatusCode::kOutOfRange,
                             "FlatMap: capacity overflow");
      }
      target_cap *= 2;
    }
    return rehash(target_cap);
  }

  void clear() noexcept {
    if (size_ == 0 || ctrl_ == nullptr) return;
    for (Index i = 0; i < capacity_; ++i) {
      if ((ctrl_[i] & kOccupiedBit) != 0) {
        slots_[i].~Entry();
        ctrl_[i] = kEmptyTag;
      }
    }
    size_ = 0;
  }

  // --- lookup ----------------------------------------------------------------
  V* find(const K& key) noexcept {
    const Index idx = find_slot_index(key);
    return idx >= 0 ? &slots_[idx].value : nullptr;
  }

  const V* find(const K& key) const noexcept {
    const Index idx = find_slot_index(key);
    return idx >= 0 ? &slots_[idx].value : nullptr;
  }

  bool contains(const K& key) const noexcept {
    return find_slot_index(key) >= 0;
  }

  // --- insertion / update ----------------------------------------------------
  // Insert or overwrite `key` with `value`.
  // If `out_inserted` is non-null, sets *out_inserted = true if a new key was
  // inserted, or false if an existing key was overwritten.
  template <typename KeyArg, typename ValArg>
  Status put(KeyArg&& key, ValArg&& value,
             bool* out_inserted = nullptr) noexcept {
    L3_RETURN_IF_ERROR(ensure_room_for_one());

    const u64 h = Hasher{}(key);
    const u8 tag = make_tag(h);
    const Index mask = capacity_ - 1;
    Index idx = static_cast<Index>(h) & mask;

    for (;;) {
      const u8 c = ctrl_[idx];
      if (c == kEmptyTag) {
        ::new (static_cast<void*>(&slots_[idx]))
            Entry{K(std::forward<KeyArg>(key)), V(std::forward<ValArg>(value))};
        ctrl_[idx] = tag;
        ++size_;
        if (out_inserted != nullptr) *out_inserted = true;
        return Status::ok();
      }
      if (c == tag && slots_[idx].key == key) {
        slots_[idx].value = std::forward<ValArg>(value);
        if (out_inserted != nullptr) *out_inserted = false;
        return Status::ok();
      }
      idx = (idx + 1) & mask;
    }
  }

  // Insert `(key, value)` ONLY if `key` is not already in the map.
  // Returns a pointer to the value in the map via `out_value` (if non-null)
  // and sets `*out_inserted` (if non-null) to true when newly inserted.
  template <typename KeyArg, typename ValArg>
  Status try_insert(KeyArg&& key, ValArg&& value, V** out_value = nullptr,
                    bool* out_inserted = nullptr) noexcept {
    L3_RETURN_IF_ERROR(ensure_room_for_one());

    const u64 h = Hasher{}(key);
    const u8 tag = make_tag(h);
    const Index mask = capacity_ - 1;
    Index idx = static_cast<Index>(h) & mask;

    for (;;) {
      const u8 c = ctrl_[idx];
      if (c == kEmptyTag) {
        ::new (static_cast<void*>(&slots_[idx]))
            Entry{K(std::forward<KeyArg>(key)), V(std::forward<ValArg>(value))};
        ctrl_[idx] = tag;
        ++size_;
        if (out_value != nullptr) *out_value = &slots_[idx].value;
        if (out_inserted != nullptr) *out_inserted = true;
        return Status::ok();
      }
      if (c == tag && slots_[idx].key == key) {
        if (out_value != nullptr) *out_value = &slots_[idx].value;
        if (out_inserted != nullptr) *out_inserted = false;
        return Status::ok();
      }
      idx = (idx + 1) & mask;
    }
  }

  // --- erasure (backward-shift deletion, zero tombstones) --------------------
  // Removes `key` if present and returns true; returns false if not found.
  bool erase(const K& key) noexcept {
    const Index found = find_slot_index(key);
    if (found < 0) return false;

    const Index mask = capacity_ - 1;
    Index hole = found;
    Index scan = (hole + 1) & mask;

    // Shift subsequent entries back if their ideal home bucket is at or before
    // `hole` (cyclically modulo capacity_).
    while (ctrl_[scan] != kEmptyTag) {
      const u64 ideal_hash = Hasher{}(slots_[scan].key);
      const Index ideal = static_cast<Index>(ideal_hash) & mask;

      // Distance from `ideal` to `scan` vs distance from `ideal` to `hole`.
      const Index dist_scan = (scan - ideal) & mask;
      const Index dist_hole = (hole - ideal) & mask;

      if (dist_hole <= dist_scan) {
        slots_[hole] = std::move(slots_[scan]);
        ctrl_[hole] = ctrl_[scan];
        hole = scan;
      }
      scan = (scan + 1) & mask;
    }

    slots_[hole].~Entry();
    ctrl_[hole] = kEmptyTag;
    --size_;
    return true;
  }

 private:
  static constexpr u8 kEmptyTag = 0x00u;
  static constexpr u8 kOccupiedBit = 0x80u;
  static constexpr Index kMinCapacity = 16;
  static constexpr Size kBlockAlign =
      alignof(Entry) > kMinObjectAlignment ? alignof(Entry) : kMinObjectAlignment;

  static constexpr u8 make_tag(u64 hash) noexcept {
    return static_cast<u8>((hash >> 57) | kOccupiedBit);
  }

  // 75% max load factor: capacity * 3 / 4.
  static constexpr Index max_load(Index cap) noexcept {
    return cap - (cap >> 2);
  }

  const Allocator& allocator() const noexcept {
    return allocator_ != nullptr ? *allocator_ : default_allocator();
  }

  // Compute total byte size and slot offset for a combined (ctrl + slots) block.
  static bool compute_layout(Index cap, Size* out_slots_offset,
                             Size* out_total_bytes) noexcept {
    const Size ctrl_bytes = static_cast<Size>(cap);
    const Size slots_offset = align_up(ctrl_bytes, alignof(Entry));
    Size slots_bytes = 0;
    if (!checked_mul(static_cast<Size>(cap), sizeof(Entry), &slots_bytes)) {
      return false;
    }
    if (!checked_add(slots_offset, slots_bytes, out_total_bytes)) {
      return false;
    }
    *out_slots_offset = slots_offset;
    return true;
  }

  void free_table() noexcept {
    if (ctrl_ != nullptr) {
      Size slots_offset = 0;
      Size total_bytes = 0;
      compute_layout(capacity_, &slots_offset, &total_bytes);
      allocator().deallocate(ctrl_, total_bytes, kBlockAlign);
      ctrl_ = nullptr;
      slots_ = nullptr;
      capacity_ = 0;
    }
  }

  Status ensure_room_for_one() noexcept {
    if (L3_LIKELY(size_ + 1 <= max_load(capacity_))) {
      return Status::ok();
    }
    const Index new_cap = capacity_ == 0 ? kMinCapacity : capacity_ * 2;
    return rehash(new_cap);
  }

  Status rehash(Index new_cap) noexcept {
    L3_DCHECK(is_power_of_two(static_cast<Size>(new_cap)));

    Size slots_offset = 0;
    Size total_bytes = 0;
    if (!compute_layout(new_cap, &slots_offset, &total_bytes)) {
      return Status::error(StatusCode::kOutOfRange, "FlatMap: layout overflow");
    }

    void* raw = allocator().allocate(total_bytes, kBlockAlign);
    if (L3_UNLIKELY(raw == nullptr)) {
      return Status::errorf(StatusCode::kOutOfMemory,
                            "FlatMap: failed to allocate %zu bytes",
                            total_bytes);
    }

    auto* new_ctrl = static_cast<u8*>(raw);
    auto* new_slots =
        reinterpret_cast<Entry*>(static_cast<byte*>(raw) + slots_offset);
    mem_zero(new_ctrl, static_cast<Size>(new_cap));

    const Index mask = new_cap - 1;
    for (Index i = 0; i < capacity_; ++i) {
      if ((ctrl_[i] & kOccupiedBit) != 0) {
        const u64 h = Hasher{}(slots_[i].key);
        const u8 tag = make_tag(h);
        Index dst = static_cast<Index>(h) & mask;
        while (new_ctrl[dst] != kEmptyTag) {
          dst = (dst + 1) & mask;
        }
        ::new (static_cast<void*>(&new_slots[dst])) Entry(std::move(slots_[i]));
        new_ctrl[dst] = tag;
        slots_[i].~Entry();
      }
    }

    free_table();
    ctrl_ = new_ctrl;
    slots_ = new_slots;
    capacity_ = new_cap;
    return Status::ok();
  }

  Index find_slot_index(const K& key) const noexcept {
    if (size_ == 0) return -1;
    const u64 h = Hasher{}(key);
    const u8 tag = make_tag(h);
    const Index mask = capacity_ - 1;
    Index idx = static_cast<Index>(h) & mask;

    for (;;) {
      const u8 c = ctrl_[idx];
      if (c == kEmptyTag) return -1;
      if (c == tag && slots_[idx].key == key) return idx;
      idx = (idx + 1) & mask;
    }
  }

  u8* ctrl_;
  Entry* slots_;
  Index size_;
  Index capacity_;
  const Allocator* allocator_;
};

// Convenience alias for string-keyed maps (tensor registries, token tables).
template <typename V>
using StrMap = FlatMap<StrView, V>;

}  // namespace l3
