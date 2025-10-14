#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <unordered_map>
#include <fstream>
#include <zstd.h>

#include "cpp/varint.hpp"
#include "cpp/sparse.hpp"
#include "cpp/stream_frame.hpp"
#include "cpp/ringbuffer.hpp"
#include "cpp/segment_writer.hpp"

namespace py = pybind11;
using namespace data_logger;

struct LoggerCore {
    RingBuffer rb;
    std::unique_ptr<SegmentWriter> writer;
    FrameEncoder enc;

    // stream_id -> meta
    std::unordered_map<uint32_t, StreamMeta> streams;
    uint32_t next_stream_id = 1; // 0 is reserved (unused)

    LoggerCore(const std::string& dir, size_t ring_bytes, size_t rotate_bytes, int zstd_level)
        : rb(ring_bytes)
    {
        SegmentWriterConfig cfg;
        cfg.dir = dir; cfg.rotate_bytes = rotate_bytes; cfg.zstd_level = zstd_level;
        writer.reset(new SegmentWriter(cfg, rb));
        enc.buf.reserve(1<<16);
    }
    ~LoggerCore(){ close(); }

    // Register a stream; returns stream_id. Scales are required here.
    uint32_t register_stream(double epoch_scale, double value_scale) {
        uint32_t sid = next_stream_id++;
        StreamMeta m; m.epoch_scale = epoch_scale; m.value_scale = value_scale;
        streams.emplace(sid, m);
        return sid;
    }

    // record(stream_id, epoch_double, idx:uint32[], vals:double[])
    void record(uint32_t stream_id, double epoch,
                py::array_t<uint32_t, py::array::c_style | py::array::forcecast> idx,
                py::array_t<double,   py::array::c_style | py::array::forcecast> vals)
    {
        if (idx.size() != vals.size()) throw std::runtime_error("indices/values length mismatch");
        auto it = streams.find(stream_id);
        if (it == streams.end()) throw std::runtime_error("unknown stream_id");
        StreamMeta& sm = it->second;

        // Convert epoch to integer ticks, then delta vs last
        int64_t tick = (int64_t) llround(epoch / sm.epoch_scale);
        int64_t d = sm.has_last ? (tick - sm.last_epoch_tick) : tick; // first record stores absolute tick as delta from 0
        sm.last_epoch_tick = tick; sm.has_last = true;

        // Optional: ensure indices are sorted strictly increasing (cheap check, or sort if needed)
        const uint32_t* pidx = idx.data();
        #ifndef NDEBUG
        for (ssize_t i=1;i<idx.size();++i) if (pidx[i] <= pidx[i-1]) throw std::runtime_error("indices must be strictly increasing per record");
        #endif

        py::gil_scoped_release nogil;
        enc.reset();
        enc.add_record(stream_id, d, pidx, vals.data(), idx.size(), sm.value_scale);

        // push frame into ring; block on backpressure
        while (!rb.push(enc.buf.data(), (uint32_t)enc.buf.size()))
            std::this_thread::sleep_for(std::chrono::microseconds(50));
    }

    void close(){
        if (writer){ writer->stop_and_join(); writer.reset(); }
    }
};

// ---- Decoder that needs per-stream scales to reconstruct doubles & epochs ----

struct ScalePair { double epoch_scale; double value_scale; };

py::list decode_segment_file_with_scales(const std::string& path,
                                         const std::unordered_map<uint32_t, ScalePair>& scales)
{
    // Keep last tick per stream while decoding to reconstruct epoch
    std::unordered_map<uint32_t, int64_t> last_tick;
    std::unordered_map<uint32_t, bool>    has_last;

    std::ifstream f(path, std::ios::binary);
    if (!f) throw std::runtime_error("open failed: " + path);
    std::vector<uint8_t> data((std::istreambuf_iterator<char>(f)), {});
    const uint8_t* p   = data.data();
    const uint8_t* end = p + data.size();

    py::list out;

    while (p < end) {
        size_t comp_size = ZSTD_findFrameCompressedSize(p, end - p);
        if (ZSTD_isError(comp_size)) throw std::runtime_error(ZSTD_getErrorName(comp_size));

        // Decompress this frame (streaming to avoid unknown size issues)
        ZSTD_DCtx* dctx = ZSTD_createDCtx();
        std::vector<uint8_t> dst(1<<20);
        ZSTD_inBuffer inb{p, comp_size, 0};
        std::vector<uint8_t> frame_out; frame_out.reserve(1<<20);

        while (inb.pos < inb.size) {
            ZSTD_outBuffer outb{dst.data(), dst.size(), 0};
            size_t ret = ZSTD_decompressStream(dctx, &outb, &inb);
            if (ZSTD_isError(ret)) { ZSTD_freeDCtx(dctx); throw std::runtime_error(ZSTD_getErrorName(ret)); }
            frame_out.insert(frame_out.end(), dst.data(), dst.data()+outb.pos);
            if (ret == 0 && inb.pos == inb.size) break;
            if (outb.pos == outb.size) dst.resize(dst.size()*2);
        }
        ZSTD_freeDCtx(dctx);

        // Parse all records in this frame (records may be from different streams)
        FrameDecoder dec(frame_out.data(), frame_out.size());
        uint32_t sid; int64_t d_tick;
        std::vector<uint32_t> idx; std::vector<double> vals;

        while (true) {
            // We need the value_scale for this sid to parse the values
            // But sid is read inside dec.next() -> refactor: peek sid
            const uint8_t* q = dec.p;
            if (q >= dec.end) break;
            uint64_t tmp;
            const uint8_t* np = data_logger::read_varu64(q, dec.end, tmp);
            if (!np) break;
            sid = (uint32_t)tmp;

            auto it = scales.find(sid);
            if (it == scales.end()) throw std::runtime_error("missing scales for stream_id="+std::to_string(sid));
            double vscale = it->second.value_scale;

            // Now actually consume the record
            if (!dec.next(sid, d_tick, idx, vals, vscale)) break;

            int64_t tick = d_tick;
            if (has_last[sid]) tick = last_tick[sid] + d_tick;
            last_tick[sid] = tick; has_last[sid] = true;

            double epoch = (double)tick * it->second.epoch_scale;

            // Return tuple: (stream_id, epoch, idx array, val array)
            out.append(py::make_tuple(
                sid, epoch,
                py::array(idx.size(), idx.data()),
                py::array(vals.size(), vals.data())
            ));
        }

        p += comp_size; // advance to next compressed frame
    }

    return out;
}

PYBIND11_MODULE(_core, m){
    py::class_<LoggerCore>(m, "LoggerCore")
        .def(py::init<const std::string&, size_t, size_t, int>(),
             py::arg("dir"), py::arg("ring_bytes")=(1u<<26),
             py::arg("rotate_bytes")=(256u<<20), py::arg("zstd_level")=1)
        .def("register_stream", &LoggerCore::register_stream,
             py::arg("epoch_scale"), py::arg("value_scale"),
             "Register a stream; returns stream_id (uint32).")
        .def("record", &LoggerCore::record,
             py::arg("stream_id"), py::arg("epoch"),
             py::arg("indices"), py::arg("values"),
             "Record one sparse vector for a stream.")
        .def("close", &LoggerCore::close);

    // Bind a tiny helper struct for passing scales from Python
    py::class_<ScalePair>(m, "ScalePair")
        .def(py::init<>())
        .def_readwrite("epoch_scale", &ScalePair::epoch_scale)
        .def_readwrite("value_scale", &ScalePair::value_scale);

    m.def("decode_segment_file_with_scales", &decode_segment_file_with_scales,
          py::arg("path"), py::arg("scales"),
          "Decode a segment file into (stream_id, epoch, idx, vals) tuples using per-stream scales.");
}
