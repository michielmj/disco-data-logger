#pragma once
#include <cstdint>
#include <unordered_map>
#include <vector>
#include <stdexcept>
#include <cmath>
#include "varint.hpp"
#include "sparse.hpp"

namespace data_logger {

// metadata kept per stream inside the C++ logger
struct StreamMeta {
    double epoch_scale;     // e.g., seconds_per_tick (1e-6, 1e-9, …)
    double value_scale;     // fixed-point scale for values (e.g., 1e-6)
    int64_t last_epoch_tick = 0;
    bool    has_last        = false;
};

struct FrameEncoder {
    std::vector<uint8_t> buf;
    void reset(){ buf.clear(); buf.reserve(1<<15); }

    // record: [stream_id][epoch_delta_ticks][n][first_idx][gaps...][quant_vals...]
    void add_record(uint32_t stream_id,
                    int64_t epoch_delta_ticks,
                    const uint32_t* idx, const double* vals, size_t n,
                    double value_scale)
    {
        data_logger::write_varu64(buf, stream_id);
        data_logger::write_varu64(buf, data_logger::zigzag_encode(epoch_delta_ticks));

        // indices + values
        data_logger::encode_u32_gaps(buf, idx, n);
        data_logger::encode_f64_quant_vals(buf, vals, n, value_scale);
    }
};

// Decoder for one frame (returns false on truncation)
struct FrameDecoder {
    const uint8_t* p;
    const uint8_t* end;
    FrameDecoder(const uint8_t* data, size_t len): p(data), end(data+len) {}

    bool next(uint32_t& stream_id, int64_t& epoch_delta_ticks,
              std::vector<uint32_t>& idx, std::vector<double>& vals, double value_scale)
    {
        if (p>=end) return false;
        uint64_t u;
        const uint8_t* np = data_logger::read_varu64(p,end,u);
        if(!np){ p=end; return false; }
        p = np; stream_id = (uint32_t)u;

        np = data_logger::read_varu64(p,end,u);
        if(!np){ p=end; return false; }
        p = np; epoch_delta_ticks = data_logger::zigzag_decode(u);

        if (!data_logger::decode_u32_gaps(p,end,idx)) { p=end; return false; }
        if (!data_logger::decode_f64_quant_vals(p,end,idx.size(), value_scale, vals)) { p=end; return false; }
        return true;
    }
};

} // namespace data_logger
