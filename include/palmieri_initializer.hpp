#pragma once

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace pf {

// The metadata name records sequential placement, high-53-bit uniforms,
// periodic radius exclusion, and binary32 coordinate storage.
inline constexpr const char* kPalmieriInitializerMethod =
    "palmieri_sequential_mt19937_64_high53_periodic_minsep_R_store_f32_v1";
// This stable algorithm identifier is written to run metadata. Its suffix is
// unrelated to the checkpoint format.

static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53,
              "Palmieri placement requires IEEE-754 binary64");
static_assert(std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "Palmieri placement requires IEEE-754 binary32");

struct PalmieriInitDiagnostics {
    std::uint64_t candidates_drawn = 0;
    std::uint64_t candidates_rejected = 0;
    double minimum_periodic_distance = std::numeric_limits<double>::infinity();
};

struct PalmieriCentresCsvDiagnostics {
    std::size_t accepted_count = 0;
    double minimum_periodic_distance = std::numeric_limits<double>::infinity();
    std::uint64_t table_fnv1a64 = 0;
};

inline void palmieri_set_error(std::string* error, const std::string& message) {
    if (error) *error = message;
}

inline double palmieri_min_image(double d, double side) {
    return d - side * std::floor(d / side + 0.5);
}

// Library-independent [0,1) map: retain the high 53 engine bits and scale by
// exactly 2^-53. std::mt19937_64 has a standardized output sequence, whereas
// std::uniform_real_distribution does not specify an identical bit mapping
// across C++ standard-library implementations.
inline constexpr double palmieri_u01_from_u64(std::uint64_t raw) noexcept {
    return static_cast<double>(raw >> 11) * 0x1.0p-53;
}

inline double palmieri_uniform53(std::mt19937_64& rng) noexcept {
    return palmieri_u01_from_u64(static_cast<std::uint64_t>(rng()));
}

// Accepted coordinates are stored as binary32 before subsequent distance tests;
// this rounding point is part of the reproducible placement algorithm.
inline void palmieri_sequential_centres(int n, double side, double radius,
                                        std::uint64_t seed,
                                        std::vector<float>* x,
                                        std::vector<float>* y,
                                        PalmieriInitDiagnostics* diag) {
    if (!x || !y || n < 1 || !(side > 0.0) || !(radius > 0.0))
        throw std::runtime_error("invalid Palmieri initializer parameters");
    x->clear(); y->clear(); x->reserve((std::size_t)n); y->reserve((std::size_t)n);
    x->push_back((float)(0.5 * side));
    y->push_back((float)(0.5 * side));
    PalmieriInitDiagnostics d{};
    std::mt19937_64 rng(seed);
    constexpr std::uint64_t kMaxAttempts = 1000000000ull;
    while ((int)x->size() < n) {
        if (++d.candidates_drawn > kMaxAttempts)
            throw std::runtime_error("Palmieri sequential placement did not terminate");
        const double cx = side * palmieri_uniform53(rng);
        const double cy = side * palmieri_uniform53(rng);
        bool accept = true;
        for (std::size_t j = 0; j < x->size(); ++j) {
            const double dx = palmieri_min_image(cx - (*x)[j], side);
            const double dy = palmieri_min_image(cy - (*y)[j], side);
            if (dx * dx + dy * dy < radius * radius) {
                accept = false;
                break;
            }
        }
        if (!accept) { ++d.candidates_rejected; continue; }
        x->push_back((float)cx); y->push_back((float)cy);
    }
    for (std::size_t i = 1; i < x->size(); ++i)
        for (std::size_t j = 0; j < i; ++j)
            d.minimum_periodic_distance = std::min(
                d.minimum_periodic_distance,
                std::hypot(palmieri_min_image((*x)[i] - (*x)[j], side),
                           palmieri_min_image((*y)[i] - (*y)[j], side)));
    if (diag) *diag = d;
}

// FNV-1a over little-endian IEEE-754 words in x0,y0,x1,y1,... order.
inline std::uint64_t palmieri_centre_table_fnv1a64(
    const std::vector<float>& x, const std::vector<float>& y) {
    if (x.size() != y.size()) return 0;
    std::uint64_t hash = 0xcbf29ce484222325ull;
    for (std::size_t i = 0; i < x.size(); ++i) {
        const float values[2] = {x[i], y[i]};
        for (float value : values) {
            std::uint32_t bits = 0;
            std::memcpy(&bits, &value, sizeof(bits));
            for (unsigned shift = 0; shift < 32; shift += 8) {
                hash ^= (bits >> shift) & 0xffu;
                hash *= 0x00000100000001b3ull;
            }
        }
    }
    return hash;
}

inline bool palmieri_validate_centres(const std::vector<float>& x,
                                       const std::vector<float>& y,
                                       int expected_n, double side,
                                       double radius,
                                       PalmieriCentresCsvDiagnostics* diag,
                                       std::string* error) {
    if (expected_n < 1 || !(side > 0.0) || !(radius > 0.0)) {
        palmieri_set_error(error, "invalid centre-table validation parameters");
        return false;
    }
    if (x.size() != (std::size_t)expected_n || y.size() != x.size()) {
        palmieri_set_error(error, "centre-table row count does not equal expected N");
        return false;
    }
    const float expected_centre = (float)(0.5 * side);
    if (x[0] != expected_centre || y[0] != expected_centre) {
        palmieri_set_error(error, "global_id 0 is not exactly at the box centre");
        return false;
    }
    for (int i = 0; i < expected_n; ++i) {
        if (!std::isfinite(x[(std::size_t)i]) ||
            !std::isfinite(y[(std::size_t)i]) ||
            !(x[(std::size_t)i] >= 0.0f && x[(std::size_t)i] < side) ||
            !(y[(std::size_t)i] >= 0.0f && y[(std::size_t)i] < side)) {
            palmieri_set_error(error, "centre is non-finite or outside [0,L)");
            return false;
        }
    }
    double minimum = std::numeric_limits<double>::infinity();
    const double r2 = radius * radius;
    for (int i = 1; i < expected_n; ++i) {
        for (int j = 0; j < i; ++j) {
            const double dx = palmieri_min_image(
                (double)x[(std::size_t)i] - (double)x[(std::size_t)j], side);
            const double dy = palmieri_min_image(
                (double)y[(std::size_t)i] - (double)y[(std::size_t)j], side);
            const double d2 = dx * dx + dy * dy;
            minimum = std::min(minimum, std::sqrt(d2));
            if (d2 < r2) {
                palmieri_set_error(error,
                    "centre table violates the periodic minimum-separation rule");
                return false;
            }
        }
    }
    if (diag) {
        diag->accepted_count = x.size();
        diag->minimum_periodic_distance = minimum;
        diag->table_fnv1a64 = palmieri_centre_table_fnv1a64(x, y);
    }
    if (error) error->clear();
    return true;
}

inline bool palmieri_write_float(std::ofstream* out, float value,
                                  std::string* error) {
    char buffer[64];
    const auto converted = std::to_chars(
        buffer, buffer + sizeof(buffer), value, std::chars_format::general,
        std::numeric_limits<float>::max_digits10);
    if (converted.ec != std::errc{}) {
        palmieri_set_error(error, "failed to format a centre coordinate");
        return false;
    }
    out->write(buffer, converted.ptr - buffer);
    return (bool)*out;
}

inline bool palmieri_write_centres_csv(const std::string& path,
                                        const std::vector<float>& x,
                                        const std::vector<float>& y,
                                        std::string* error) {
    if (x.empty() || x.size() != y.size()) {
        palmieri_set_error(error, "cannot write an empty or mismatched centre table");
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        palmieri_set_error(error, "cannot open centre CSV for writing: " + path);
        return false;
    }
    out.write("global_id,x,y\n", 14);
    for (std::size_t i = 0; i < x.size(); ++i) {
        char id[32];
        const auto converted = std::to_chars(id, id + sizeof(id), i);
        if (converted.ec != std::errc{}) {
            palmieri_set_error(error, "failed to format global_id");
            return false;
        }
        out.write(id, converted.ptr - id);
        out.put(',');
        if (!palmieri_write_float(&out, x[i], error)) return false;
        out.put(',');
        if (!palmieri_write_float(&out, y[i], error)) return false;
        out.put('\n');
        if (!out) {
            palmieri_set_error(error, "failed while writing centre CSV: " + path);
            return false;
        }
    }
    out.close();
    if (!out) {
        palmieri_set_error(error, "failed to close centre CSV: " + path);
        return false;
    }
    if (error) error->clear();
    return true;
}

inline bool palmieri_parse_int(const char* first, const char* last, int* value) {
    const auto parsed = std::from_chars(first, last, *value, 10);
    return parsed.ec == std::errc{} && parsed.ptr == last;
}

inline bool palmieri_parse_float(const char* first, const char* last,
                                  float* value) {
    const auto parsed = std::from_chars(first, last, *value,
                                        std::chars_format::general);
    return parsed.ec == std::errc{} && parsed.ptr == last;
}

inline bool palmieri_read_centres_csv(const std::string& path, int expected_n,
                                       double side, double radius,
                                       std::vector<float>* x,
                                       std::vector<float>* y,
                                       PalmieriCentresCsvDiagnostics* diag,
                                       std::string* error) {
    if (!x || !y || expected_n < 1) {
        palmieri_set_error(error, "invalid centre CSV reader parameters");
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        palmieri_set_error(error, "cannot open centre CSV: " + path);
        return false;
    }
    std::string line;
    if (!std::getline(in, line) || line != "global_id,x,y") {
        palmieri_set_error(error, "CSV header must be exactly global_id,x,y with LF line endings");
        return false;
    }
    x->clear(); y->clear();
    x->reserve((std::size_t)expected_n); y->reserve((std::size_t)expected_n);
    for (int expected_id = 0; expected_id < expected_n; ++expected_id) {
        if (!std::getline(in, line)) {
            palmieri_set_error(error, "centre CSV ended before expected N rows");
            return false;
        }
        const std::size_t comma1 = line.find(',');
        const std::size_t comma2 = comma1 == std::string::npos
            ? std::string::npos : line.find(',', comma1 + 1);
        if (comma1 == std::string::npos || comma2 == std::string::npos ||
            line.find(',', comma2 + 1) != std::string::npos) {
            palmieri_set_error(error, "centre CSV row must contain exactly three fields");
            return false;
        }
        int id = -1;
        float xv = 0.0f, yv = 0.0f;
        if (!palmieri_parse_int(line.data(), line.data() + comma1, &id) ||
            !palmieri_parse_float(line.data() + comma1 + 1,
                                  line.data() + comma2, &xv) ||
            !palmieri_parse_float(line.data() + comma2 + 1,
                                  line.data() + line.size(), &yv)) {
            palmieri_set_error(error, "centre CSV contains a malformed field");
            return false;
        }
        if (id != expected_id) {
            palmieri_set_error(error,
                               "global_id must be the ordered sequence 0..N-1");
            return false;
        }
        x->push_back(xv); y->push_back(yv);
    }
    if (std::getline(in, line)) {
        palmieri_set_error(error, "centre CSV contains more than expected N rows");
        return false;
    }
    if (in.bad()) {
        palmieri_set_error(error, "I/O failure while reading centre CSV");
        return false;
    }
    return palmieri_validate_centres(*x, *y, expected_n, side, radius,
                                      diag, error);
}

}  // namespace pf
