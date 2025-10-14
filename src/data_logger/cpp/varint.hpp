#pragma once
#include <cstdint>
#include <vector>

namespace data_logger {
inline uint64_t zigzag_encode(int64_t v){ return (uint64_t(v) << 1) ^ uint64_t(v >> 63); }
inline int64_t  zigzag_decode(uint64_t v){ return int64_t((v >> 1) ^ (~(v & 1) + 1)); }

inline void write_varu64(std::vector<uint8_t>& b, uint64_t v){
    while (v >= 0x80){ b.push_back(uint8_t(v | 0x80)); v >>= 7; }
    b.push_back(uint8_t(v));
}
inline const uint8_t* read_varu64(const uint8_t* p, const uint8_t* end, uint64_t& out){
    uint64_t r=0; int sh=0;
    while (p<end){ uint8_t c=*p++; r|=uint64_t(c&0x7F)<<sh; if(!(c&0x80)){ out=r; return p; } sh+=7; }
    return nullptr;
}
} // namespace data_logger
