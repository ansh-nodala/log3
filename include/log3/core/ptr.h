// log3/core/ptr.h
//
// Intrusive reference counting: RefCounted (base class) + Ptr<T> (smart
// pointer). Tensor Storage and autograd graph Nodes are owned through Ptr<T>.
//
// Why not std::shared_ptr:
//   * One allocation, no separate control block. The count lives inside the
//     object, next to its data (better cache behavior).
//   * Ptr<T> is exactly one pointer wide. shared_ptr is two.
//   * You can make a Ptr from a raw pointer (e.g. `Ptr<Node>(this)`) at any
//     time, because the count is in the object. No enable_shared_from_this.
//   * Allocation goes through log3's Allocator (arena / pool / tracked), and
//     frees with the exact size via sized deallocation.
//   * No weak-count bookkeeping, no RTTI, no exceptions, no virtual dtor.
//
// How destruction works without a virtual destructor: make_ptr<T> knows the
// exact type T, so it records a tiny function "destroy this T" and the
// allocator in the object header. When the last Ptr goes away, that function
// runs ~T() and frees sizeof(T) bytes. So Ptr<Base> pointing at a Derived
// object destroys the Derived correctly and frees the right number of bytes,
// even if Base has no virtual destructor.
//
// Object header cost: 3 pointers (24 bytes on 64-bit): count + destroy
// function + allocator.
//
// Thread safety: the count is atomic. Copying/destroying Ptrs to the same
// object from several threads is safe. The pointed-to object itself is not
// synchronized.
//
// Rules for classes managed by Ptr:
//   * derive publicly from RefCounted
//   * create them only with make_ptr<T>() / make_ptr_in<T>()
//     (never on the stack, never with new/delete)
//   * the Allocator passed to make_ptr_in must outlive every object created
//     from it (the object stores a pointer to it). default_allocator() always
//     does.
//   * reference cycles leak (A holds Ptr<B>, B holds Ptr<A>). Break cycles in
//     the design; a weak pointer can be added later if needed.

#pragma once

#include <atomic>
#include <type_traits>
#include <utility>

#include "log3/core/assert.h"
#include "log3/core/memory.h"
#include "log3/core/platform.h"
#include "log3/core/status.h"
#include "log3/core/types.h"

namespace l3 {

class RefCounted;

namespace detail {
struct RefCountedAccess;
}  // namespace detail

// -----------------------------------------------------------------------------
// RefCounted
// -----------------------------------------------------------------------------
class RefCounted {
 public:
  using DestroyFn = void (*)(RefCounted* self) noexcept;

  RefCounted(const RefCounted&) = delete;
  RefCounted& operator=(const RefCounted&) = delete;

  // Current number of Ptrs. Informational under concurrency.
  u32 use_count() const noexcept {
    return count_.load(std::memory_order_relaxed);
  }

  // True if exactly one Ptr owns this object (e.g. safe to mutate in place /
  // copy-on-write). The acquire load pairs with other threads' releases.
  bool is_unique() const noexcept {
    return count_.load(std::memory_order_acquire) == 1;
  }

  void add_ref() const noexcept {
    count_.fetch_add(1, std::memory_order_relaxed);
  }

  // Drops one reference; destroys the object when it was the last one.
  void release() const noexcept {
#if L3_TSAN
    const u32 old = count_.fetch_sub(1, std::memory_order_acq_rel);
    L3_DCHECK(old > 0);
    if (old == 1) destroy_fn_(const_cast<RefCounted*>(this));
#else
    // Release on the decrement, acquire fence only on the final one: cheapest
    // correct pattern (the same one libstdc++/libc++ shared_ptr use).
    const u32 old = count_.fetch_sub(1, std::memory_order_release);
    L3_DCHECK(old > 0);
    if (old == 1) {
      std::atomic_thread_fence(std::memory_order_acquire);
      destroy_fn_(const_cast<RefCounted*>(this));
    }
#endif
  }

 protected:
  RefCounted() noexcept : count_(0), destroy_fn_(nullptr), allocator_(nullptr) {}
  ~RefCounted() = default;  // non-virtual and protected: no delete via base

 private:
  friend struct detail::RefCountedAccess;

  mutable std::atomic<u32> count_;
  DestroyFn destroy_fn_;
  const Allocator* allocator_;
};

static_assert(std::atomic<u32>::is_always_lock_free,
              "log3: atomic<u32> must be lock-free.");
static_assert(sizeof(RefCounted) == 3 * sizeof(void*),
              "log3: RefCounted header must stay 3 pointers wide.");

// -----------------------------------------------------------------------------
// Ptr<T>
// -----------------------------------------------------------------------------
template <typename T>
class Ptr {
 public:
  using element_type = T;

  constexpr Ptr() noexcept : ptr_(nullptr) {}
  constexpr Ptr(std::nullptr_t) noexcept : ptr_(nullptr) {}

  // Share an existing object: increments its count. This is how you get a Ptr
  // back from a raw pointer or from `this`.
  explicit Ptr(T* raw) noexcept : ptr_(raw) {
    static_assert(std::is_convertible_v<T*, const RefCounted*>,
                  "Ptr<T>: T must derive publicly from RefCounted.");
    if (ptr_ != nullptr) ptr_->add_ref();
  }

  Ptr(const Ptr& other) noexcept : ptr_(other.ptr_) {
    if (ptr_ != nullptr) ptr_->add_ref();
  }

  Ptr(Ptr&& other) noexcept : ptr_(other.ptr_) { other.ptr_ = nullptr; }

  // Ptr<Derived> -> Ptr<Base>
  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
  Ptr(const Ptr<U>& other) noexcept : ptr_(other.get()) {
    if (ptr_ != nullptr) ptr_->add_ref();
  }

  template <typename U,
            typename = std::enable_if_t<std::is_convertible_v<U*, T*>>>
  Ptr(Ptr<U>&& other) noexcept : ptr_(other.detach()) {}

  ~Ptr() {
    static_assert(std::is_convertible_v<T*, const RefCounted*>,
                  "Ptr<T>: T must derive publicly from RefCounted.");
    if (ptr_ != nullptr) ptr_->release();
  }

  Ptr& operator=(const Ptr& other) noexcept {
    Ptr(other).swap(*this);
    return *this;
  }

  Ptr& operator=(Ptr&& other) noexcept {
    Ptr(std::move(other)).swap(*this);
    return *this;
  }

  Ptr& operator=(std::nullptr_t) noexcept {
    reset();
    return *this;
  }

  // Take ownership of a pointer whose reference was already counted (does NOT
  // increment). The counterpart of detach().
  static Ptr adopt(T* raw) noexcept {
    Ptr p;
    p.ptr_ = raw;
    return p;
  }

  // Give up ownership WITHOUT decrementing; the caller now owns one reference.
  T* detach() noexcept {
    T* p = ptr_;
    ptr_ = nullptr;
    return p;
  }

  void reset() noexcept { Ptr().swap(*this); }

  void swap(Ptr& other) noexcept {
    T* tmp = ptr_;
    ptr_ = other.ptr_;
    other.ptr_ = tmp;
  }

  T* get() const noexcept { return ptr_; }

  T& operator*() const noexcept {
    L3_DCHECK(ptr_ != nullptr);
    return *ptr_;
  }

  T* operator->() const noexcept {
    L3_DCHECK(ptr_ != nullptr);
    return ptr_;
  }

  explicit operator bool() const noexcept { return ptr_ != nullptr; }

  u32 use_count() const noexcept { return ptr_ != nullptr ? ptr_->use_count() : 0; }

 private:
  T* ptr_;
};

static_assert(sizeof(Ptr<RefCounted>) == sizeof(void*),
              "log3: Ptr<T> must be exactly one pointer wide.");

// --- comparisons ---------------------------------------------------------------
template <typename A, typename B>
bool operator==(const Ptr<A>& a, const Ptr<B>& b) noexcept {
  return a.get() == b.get();
}
template <typename A, typename B>
bool operator!=(const Ptr<A>& a, const Ptr<B>& b) noexcept {
  return a.get() != b.get();
}
template <typename A>
bool operator==(const Ptr<A>& a, std::nullptr_t) noexcept {
  return a.get() == nullptr;
}
template <typename A>
bool operator!=(const Ptr<A>& a, std::nullptr_t) noexcept {
  return a.get() != nullptr;
}
template <typename A>
bool operator==(std::nullptr_t, const Ptr<A>& a) noexcept {
  return a.get() == nullptr;
}
template <typename A>
bool operator!=(std::nullptr_t, const Ptr<A>& a) noexcept {
  return a.get() != nullptr;
}

// Downcast (Ptr<Base> -> Ptr<Derived>). No RTTI: the caller guarantees the
// dynamic type. Increments the count.
template <typename T, typename U>
Ptr<T> static_ptr_cast(const Ptr<U>& p) noexcept {
  return Ptr<T>(static_cast<T*>(p.get()));
}

// -----------------------------------------------------------------------------
// Creation
// -----------------------------------------------------------------------------
namespace detail {

struct RefCountedAccess {
  static void bind(RefCounted* self, RefCounted::DestroyFn fn,
                   const Allocator* allocator) noexcept {
    self->destroy_fn_ = fn;
    self->allocator_ = allocator;
  }

  template <typename T>
  static void destroy_impl(RefCounted* self) noexcept {
    constexpr Size kAlign =
        alignof(T) > kMinObjectAlignment ? alignof(T) : kMinObjectAlignment;
    T* obj = static_cast<T*>(self);
    const Allocator* allocator = self->allocator_;  // read before ~T()
    obj->~T();
    allocator->deallocate(obj, sizeof(T), kAlign);
  }
};

}  // namespace detail

// Allocate and construct a T from `allocator`, wrapped in a Ptr (count = 1).
// On allocation failure returns a kOutOfMemory Status.
// T's constructor must not throw (the core is built without exceptions).
template <typename T, typename... Args>
Result<Ptr<T>> make_ptr_in(const Allocator& allocator, Args&&... args) noexcept {
  static_assert(std::is_convertible_v<T*, RefCounted*>,
                "make_ptr: T must derive publicly from RefCounted.");
  static_assert(!std::is_abstract_v<T>, "make_ptr: T must not be abstract.");
  constexpr Size kAlign =
      alignof(T) > kMinObjectAlignment ? alignof(T) : kMinObjectAlignment;

  void* mem = allocator.allocate(sizeof(T), kAlign);
  if (L3_UNLIKELY(mem == nullptr)) {
    return Status::errorf(StatusCode::kOutOfMemory,
                          "make_ptr: failed to allocate %zu bytes (allocator: %s)",
                          sizeof(T),
                          allocator.name != nullptr ? allocator.name : "?");
  }
  T* obj = ::new (mem) T(std::forward<Args>(args)...);
  detail::RefCountedAccess::bind(obj, &detail::RefCountedAccess::destroy_impl<T>,
                                 &allocator);
  return Ptr<T>(obj);
}

// Same, using default_allocator().
template <typename T, typename... Args>
Result<Ptr<T>> make_ptr(Args&&... args) noexcept {
  return make_ptr_in<T>(default_allocator(), std::forward<Args>(args)...);
}

}  // namespace l3
