#pragma once
#include <cstdint>
#include <atomic>
#include <vector>
#include <cstring>

// Single-producer / single-consumer lock-free ring buffer of byte blobs.
// Producer publishes contiguous messages: [u32 len][payload bytes...]
namespace data_logger {

struct RingBuffer {
    std::vector<uint8_t> buf;
    const size_t cap;
    std::atomic<size_t> head; // write
    std::atomic<size_t> tail; // read

    explicit RingBuffer(size_t capacity_bytes)
        : buf(capacity_bytes), cap(capacity_bytes), head(0), tail(0) {}

    // Internal utility
    size_t free_bytes() const {
        size_t h = head.load(std::memory_order_acquire);
        size_t t = tail.load(std::memory_order_acquire);
        return (t + cap - h - 1) % cap;
    }

    // Try to push a blob (len <= 2^32-1). Returns false if not enough space.
    bool push(const uint8_t* data, uint32_t len) {
        size_t need = sizeof(uint32_t) + len;
        size_t h = head.load(std::memory_order_relaxed);
        size_t t = tail.load(std::memory_order_acquire);
        size_t free = (t + cap - h - 1) % cap;
        if (need > free) return false;

        auto write_bytes = [&](const void* src, size_t n){
            size_t h2 = h % cap;
            size_t first = std::min(n, cap - h2);
            std::memcpy(&buf[h2], src, first);
            if (n > first) std::memcpy(&buf[0], (const uint8_t*)src + first, n - first);
            h += n;
        };

        write_bytes(&len, sizeof(uint32_t));
        write_bytes(data, len);
        head.store(h, std::memory_order_release);
        return true;
    }

    // Pop next blob into out. Returns false if empty.
    bool pop(std::vector<uint8_t>& out) {
        size_t t = tail.load(std::memory_order_relaxed);
        size_t h = head.load(std::memory_order_acquire);
        if (t == h) return false;

        uint32_t len;
        auto read_bytes = [&](void* dst, size_t n){
            size_t t2 = t % cap;
            size_t first = std::min(n, cap - t2);
            std::memcpy(dst, &buf[t2], first);
            if (n > first) std::memcpy((uint8_t*)dst + first, &buf[0], n - first);
            t += n;
        };

        read_bytes(&len, sizeof(uint32_t));
        out.resize(len);
        read_bytes(out.data(), len);
        tail.store(t, std::memory_order_release);
        return true;
    }
};

} // namespace data_logger
