#pragma once

// ----------------------------------------------------------------------------
// segment_writer.hpp
// Append-only zstd-framed segment writer with a lock-free ring buffer backend.
// Portable fsync handling for Linux, macOS, and Windows.
// ----------------------------------------------------------------------------

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include <zstd.h>  // linked via CMake (vendored or system)

#include "ringbuffer.hpp"  // data_logger::RingBuffer

// Platform headers for fsync/flush
#if defined(_WIN32)
  #include <io.h>
  #include <fcntl.h>
#elif defined(__APPLE__)
  #include <unistd.h>
  #include <fcntl.h>
#else
  #include <unistd.h>
  #include <fcntl.h>
#endif

namespace data_logger {

// ---------------------------------------------------------------------------
// Portable fsync helper
//   - Windows:   _commit(_fileno(fp))
//   - macOS:     fcntl(fd, F_FULLFSYNC) fallback to fsync
//   - POSIX:     fdatasync if available (define HAVE_FDATASYNC in CMake), else fsync
// ---------------------------------------------------------------------------
inline void portable_fsync(FILE* fp, bool do_fsync) {
    if (!fp || !do_fsync) return;

#if defined(_WIN32)
    _commit(_fileno(fp));  // flush file buffers to disk on Windows

#elif defined(__APPLE__)
    const int fd = ::fileno(fp);
    if (fd >= 0) {
        // F_FULLFSYNC is strongest; fall back to fsync if not supported
        if (::fcntl(fd, F_FULLFSYNC) == -1) {
            ::fsync(fd);
        }
    }

#else  // POSIX (Linux, *BSD, etc.)
    const int fd = ::fileno(fp);
    if (fd >= 0) {
      #if defined(HAVE_FDATASYNC)
        ::fdatasync(fd);
      #else
        ::fsync(fd);
      #endif
    }
#endif
}

// ---------------------------------------------------------------------------
// Configuration for the segment writer
// ---------------------------------------------------------------------------
struct SegmentWriterConfig {
    std::string dir;              // output directory for segment files
    size_t      rotate_bytes = 256u << 20; // 256 MB default rotation
    int         zstd_level   = 1;          // zstd compression level
    bool        fsync_on_close = true;     // fsync when closing/rotating segment
};

// ---------------------------------------------------------------------------
// SegmentWriter
// Consumes raw frames from a RingBuffer, compresses each to a zstd frame,
// and appends into rotating segment files: 00000.seg.zst, 00001.seg.zst, ...
// ---------------------------------------------------------------------------
class SegmentWriter {
public:
    SegmentWriter(SegmentWriterConfig c, RingBuffer& r)
        : cfg_(std::move(c)), rb_(r) {
        th_ = std::thread([this]{ this->run_(); });
    }

    ~SegmentWriter() {
        stop_and_join();
    }

    void stop_and_join() {
        const bool was_running = !stop_.exchange(true, std::memory_order_acq_rel);
        if (was_running && th_.joinable()) th_.join();
    }

private:
    SegmentWriterConfig cfg_;
    RingBuffer&         rb_;
    std::thread         th_;
    std::atomic<bool>   stop_{false};

    FILE*     fp_ = nullptr;
    size_t    written_in_seg_ = 0;
    uint64_t  seg_seq_ = 0;

    std::vector<uint8_t> out_buf_;  // compressed frame buffer
    std::vector<uint8_t> blob_;     // dequeued raw frame

    // Open a new rotating segment file
    void open_new_segment_() {
        if (fp_) close_segment_();

        char path[1024];
        // 00000.seg.zst, 00001.seg.zst, ...
        std::snprintf(path, sizeof(path), "%s/%05llu.seg.zst",
                      cfg_.dir.c_str(),
                      static_cast<unsigned long long>(seg_seq_++));

        fp_ = std::fopen(path, "wb");
        if (!fp_) {
            throw std::runtime_error(std::string("Failed to open segment file: ") + path);
        }
        written_in_seg_ = 0;
    }

    void close_segment_() {
        if (!fp_) return;
        std::fflush(fp_);
        portable_fsync(fp_, cfg_.fsync_on_close);
        std::fclose(fp_);
        fp_ = nullptr;
    }

    void write_compressed_frame_(const uint8_t* src, size_t src_size) {
        // Compress one logical frame to a single zstd frame
        const size_t max_dst = ZSTD_compressBound(src_size);
        out_buf_.resize(max_dst);
        const size_t n = ZSTD_compress(out_buf_.data(), out_buf_.size(),
                                       src, src_size, cfg_.zstd_level);
        if (ZSTD_isError(n)) {
            throw std::runtime_error(std::string("ZSTD_compress error: ") + ZSTD_getErrorName(n));
        }

        if (!fp_) open_new_segment_();

        const size_t wrote = std::fwrite(out_buf_.data(), 1, n, fp_);
        if (wrote != n) {
            throw std::runtime_error("fwrite failed while writing compressed frame");
        }
        written_in_seg_ += wrote;

        if (written_in_seg_ >= cfg_.rotate_bytes) {
            open_new_segment_();  // rotation (close + new)
        }
    }

    void run_() {
        // Start with first segment
        open_new_segment_();

        // Main loop: pop frames; if empty, nap briefly
        while (!stop_.load(std::memory_order_acquire)) {
            if (!rb_.pop(blob_)) {
                std::this_thread::sleep_for(std::chrono::microseconds(50));
                continue;
            }
            write_compressed_frame_(blob_.data(), blob_.size());
        }

        // Drain any remaining frames after stop
        while (rb_.pop(blob_)) {
            write_compressed_frame_(blob_.data(), blob_.size());
        }

        close_segment_();
    }
};

} // namespace data_logger
