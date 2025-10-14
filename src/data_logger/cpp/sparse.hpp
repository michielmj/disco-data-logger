#pragma once
#include <vector>
#include <cstdint>
#include <cmath>
#include "varint.hpp"

namespace data_logger {

// Encode indices as: n, first_idx (absolute), then gaps (idx[i]-idx[i-1]-1)
inline void encode_u32_gaps(std::vector<uint8_t>& buf, const uint32_t* idx, size_t n){
    data_logger::write_varu64(buf, n);
    if (n==0) return;
    data_logger::write_varu64(buf, idx[0]);
    for (size_t i=1;i<n;++i){
        uint64_t gap = uint64_t(idx[i]) - uint64_t(idx[i-1]) - 1;
        data_logger::write_varu64(buf, gap);
    }
}

// Decode back to absolute indices
inline bool decode_u32_gaps(const uint8_t*& p, const uint8_t* end, std::vector<uint32_t>& out){
    uint64_t n; p = data_logger::read_varu64(p,end,n); if(!p) return false;
    out.resize(n);
    if (n==0) return true;
    uint64_t v; p = data_logger::read_varu64(p,end,v); if(!p) return false;
    out[0] = (uint32_t)v;
    for (size_t i=1;i<n;++i){
        p = data_logger::read_varu64(p,end,v); if(!p) return false;
        out[i] = (uint32_t)( (uint64_t)out[i-1] + 1 + v );
    }
    return true;
}

// Quantize doubles to int64 with given scale, then ZigZag+Varint
inline void encode_f64_quant_vals(std::vector<uint8_t>& buf, const double* vals, size_t n, double value_scale){
    for (size_t i=0;i<n;++i){
        int64_t q = (int64_t) llround(vals[i] / value_scale);
        data_logger::write_varu64(buf, data_logger::zigzag_encode(q));
    }
}
inline bool decode_f64_quant_vals(const uint8_t*& p, const uint8_t* end, size_t n, double value_scale, std::vector<double>& out){
    out.resize(n);
    for (size_t i=0;i<n;++i){
        uint64_t u; p = data_logger::read_varu64(p,end,u); if(!p) return false;
        int64_t q = data_logger::zigzag_decode(u);
        out[i] = (double)q * value_scale;
    }
    return true;
}

} // namespace data_logger
