#pragma once

// Reproducible periodic 3D placement. Continuous proposals are accepted
// sequentially at the required minimum separation. Cell zero is centered;
// accepted coordinates are rounded to binary32 before later tests.

#include "initializer.hpp"

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

namespace pf3d {

inline constexpr const char* kPalmieriInitializer3DMethod =
    "palmieri_sequential_mt19937_64_high53_periodic_minsep_R_store_f32_3d_v1";
inline constexpr const char* kChannelInitializer3DMethod =
    "sequential_mt19937_64_periodic_xy_resolved_wall_physical_z_"
    "minsep_R_store_f32_v2";
// These strings are stable algorithm identifiers written to run metadata;
// their suffixes distinguish placement algorithms, not checkpoint formats.

static_assert(std::numeric_limits<double>::is_iec559 &&
                  std::numeric_limits<double>::digits == 53,
              "Palmieri placement requires IEEE-754 binary64");
static_assert(std::numeric_limits<float>::is_iec559 &&
                  std::numeric_limits<float>::digits == 24,
              "Palmieri placement requires IEEE-754 binary32");

struct PalmieriInit3DDiagnostics {
    std::uint64_t candidates_drawn = 0;
    std::uint64_t candidates_rejected = 0;
    double minimum_periodic_distance = std::numeric_limits<double>::infinity();
};

struct PalmieriCentres3DCsvDiagnostics {
    std::size_t accepted_count = 0;
    double minimum_periodic_distance = std::numeric_limits<double>::infinity();
    std::uint64_t table_fnv1a64 = 0;
};

inline void palmieri3d_set_error(std::string* error,
                                 const std::string& message) {
    if (error) *error = message;
}

inline double palmieri3d_min_image(double d, double side) {
    return d - side * std::floor(d / side + 0.5);
}

// Library-independent [0,1) map; see the two-dimensional generator.
inline constexpr double palmieri3d_u01_from_u64(std::uint64_t raw) noexcept {
    return static_cast<double>(raw >> 11) * 0x1.0p-53;
}

inline double palmieri3d_uniform53(std::mt19937_64& rng) noexcept {
    return palmieri3d_u01_from_u64(static_cast<std::uint64_t>(rng()));
}

// Accepted coordinates are stored as binary32 before subsequent distance
// tests; this rounding point is part of the reproducible algorithm.  Each
// candidate draws x, then y, then z.
inline void palmieri3d_sequential_centres(int n, double side, double radius,
                                          std::uint64_t seed,
                                          std::vector<float>* x,
                                          std::vector<float>* y,
                                          std::vector<float>* z,
                                          PalmieriInit3DDiagnostics* diag) {
    if (!x || !y || !z || n < 1 || !(side > 0.0) || !(radius > 0.0))
        throw std::runtime_error("invalid 3D Palmieri initializer parameters");
    x->clear(); y->clear(); z->clear();
    x->reserve((std::size_t)n); y->reserve((std::size_t)n);
    z->reserve((std::size_t)n);
    x->push_back((float)(0.5 * side));
    y->push_back((float)(0.5 * side));
    z->push_back((float)(0.5 * side));
    PalmieriInit3DDiagnostics d{};
    std::mt19937_64 rng(seed);
    constexpr std::uint64_t kMaxAttempts = 1000000000ull;
    const double r2 = radius * radius;
    while ((int)x->size() < n) {
        if (++d.candidates_drawn > kMaxAttempts)
            throw std::runtime_error(
                "3D Palmieri sequential placement did not terminate");
        const double cx = side * palmieri3d_uniform53(rng);
        const double cy = side * palmieri3d_uniform53(rng);
        const double cz = side * palmieri3d_uniform53(rng);
        bool accept = true;
        for (std::size_t j = 0; j < x->size(); ++j) {
            const double dx = palmieri3d_min_image(cx - (*x)[j], side);
            const double dy = palmieri3d_min_image(cy - (*y)[j], side);
            const double dz = palmieri3d_min_image(cz - (*z)[j], side);
            if (dx * dx + dy * dy + dz * dz < r2) {
                accept = false;
                break;
            }
        }
        if (!accept) { ++d.candidates_rejected; continue; }
        x->push_back((float)cx);
        y->push_back((float)cy);
        z->push_back((float)cz);
    }
    for (std::size_t i = 1; i < x->size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const double dx = palmieri3d_min_image((*x)[i] - (*x)[j], side);
            const double dy = palmieri3d_min_image((*y)[i] - (*y)[j], side);
            const double dz = palmieri3d_min_image((*z)[i] - (*z)[j], side);
            d.minimum_periodic_distance = std::min(
                d.minimum_periodic_distance,
                std::sqrt(dx * dx + dy * dy + dz * dz));
        }
    }
    if (diag) *diag = d;
}

// FNV-1a over the full logical table: ordered int64 global IDs followed by the
// corresponding little-endian binary32 x, y, and z coordinates.
inline std::uint64_t palmieri3d_centre_table_fnv1a64(
    const std::vector<float>& x, const std::vector<float>& y,
    const std::vector<float>& z) {
    if (x.size() != y.size() || x.size() != z.size()) return 0;
    return centre_table_fnv1a64(ordered_global_ids(x.size()), x, y, z);
}

inline bool palmieri3d_validate_centres(
    const std::vector<float>& x, const std::vector<float>& y,
    const std::vector<float>& z, int expected_n, double side, double radius,
    PalmieriCentres3DCsvDiagnostics* diag, std::string* error) {
    if (expected_n < 1 || !(side > 0.0) || !(radius > 0.0)) {
        palmieri3d_set_error(error,
                             "invalid centre-table validation parameters");
        return false;
    }
    if (x.size() != (std::size_t)expected_n || y.size() != x.size() ||
        z.size() != x.size()) {
        palmieri3d_set_error(error,
                             "centre-table row count does not equal expected N");
        return false;
    }
    const float expected_centre = (float)(0.5 * side);
    if (x[0] != expected_centre || y[0] != expected_centre ||
        z[0] != expected_centre) {
        palmieri3d_set_error(error,
                             "global_id 0 is not exactly at the box centre");
        return false;
    }
    for (int i = 0; i < expected_n; ++i) {
        const std::size_t k = (std::size_t)i;
        if (!std::isfinite(x[k]) || !std::isfinite(y[k]) ||
            !std::isfinite(z[k]) ||
            !(x[k] >= 0.0f && x[k] < side) ||
            !(y[k] >= 0.0f && y[k] < side) ||
            !(z[k] >= 0.0f && z[k] < side)) {
            palmieri3d_set_error(error,
                                 "centre is non-finite or outside [0,L)");
            return false;
        }
    }
    double minimum = std::numeric_limits<double>::infinity();
    const double r2 = radius * radius;
    for (int i = 1; i < expected_n; ++i) {
        for (int j = 0; j < i; ++j) {
            const double dx = palmieri3d_min_image(
                (double)x[(std::size_t)i] - (double)x[(std::size_t)j], side);
            const double dy = palmieri3d_min_image(
                (double)y[(std::size_t)i] - (double)y[(std::size_t)j], side);
            const double dz = palmieri3d_min_image(
                (double)z[(std::size_t)i] - (double)z[(std::size_t)j], side);
            const double d2 = dx * dx + dy * dy + dz * dz;
            minimum = std::min(minimum, std::sqrt(d2));
            if (d2 < r2) {
                palmieri3d_set_error(error,
                    "centre table violates the periodic minimum-separation rule");
                return false;
            }
        }
    }
    if (diag) {
        diag->accepted_count = x.size();
        diag->minimum_periodic_distance = minimum;
        diag->table_fnv1a64 = palmieri3d_centre_table_fnv1a64(x, y, z);
    }
    if (error) error->clear();
    return true;
}

inline bool palmieri3d_write_float(std::ofstream* out, float value,
                                   std::string* error) {
    char buffer[64];
    const auto converted = std::to_chars(
        buffer, buffer + sizeof(buffer), value, std::chars_format::general,
        std::numeric_limits<float>::max_digits10);
    if (converted.ec != std::errc{}) {
        palmieri3d_set_error(error, "failed to format a centre coordinate");
        return false;
    }
    out->write(buffer, converted.ptr - buffer);
    return (bool)*out;
}

inline bool palmieri3d_write_centres_csv(const std::string& path,
                                         const std::vector<float>& x,
                                         const std::vector<float>& y,
                                         const std::vector<float>& z,
                                         std::string* error) {
    if (x.empty() || x.size() != y.size() || x.size() != z.size()) {
        palmieri3d_set_error(error,
            "cannot write an empty or mismatched centre table");
        return false;
    }
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    if (!out) {
        palmieri3d_set_error(error,
                             "cannot open centre CSV for writing: " + path);
        return false;
    }
    out.write("global_id,x,y,z\n", 16);
    for (std::size_t i = 0; i < x.size(); ++i) {
        char id[32];
        const auto converted = std::to_chars(id, id + sizeof(id), i);
        if (converted.ec != std::errc{}) {
            palmieri3d_set_error(error, "failed to format global_id");
            return false;
        }
        out.write(id, converted.ptr - id);
        out.put(',');
        if (!palmieri3d_write_float(&out, x[i], error)) return false;
        out.put(',');
        if (!palmieri3d_write_float(&out, y[i], error)) return false;
        out.put(',');
        if (!palmieri3d_write_float(&out, z[i], error)) return false;
        out.put('\n');
        if (!out) {
            palmieri3d_set_error(error,
                                 "failed while writing centre CSV: " + path);
            return false;
        }
    }
    out.close();
    if (!out) {
        palmieri3d_set_error(error, "failed to close centre CSV: " + path);
        return false;
    }
    if (error) error->clear();
    return true;
}

inline bool palmieri3d_parse_int(const char* first, const char* last,
                                 int* value) {
    const auto parsed = std::from_chars(first, last, *value, 10);
    return parsed.ec == std::errc{} && parsed.ptr == last;
}

inline bool palmieri3d_parse_float(const char* first, const char* last,
                                   float* value) {
    const auto parsed = std::from_chars(first, last, *value,
                                        std::chars_format::general);
    return parsed.ec == std::errc{} && parsed.ptr == last;
}

inline bool palmieri3d_read_centres_csv(
    const std::string& path, int expected_n, double side, double radius,
    std::vector<float>* x, std::vector<float>* y, std::vector<float>* z,
    PalmieriCentres3DCsvDiagnostics* diag, std::string* error) {
    if (!x || !y || !z || expected_n < 1) {
        palmieri3d_set_error(error, "invalid centre CSV reader parameters");
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        palmieri3d_set_error(error, "cannot open centre CSV: " + path);
        return false;
    }
    std::string line;
    if (!std::getline(in, line) || line != "global_id,x,y,z") {
        palmieri3d_set_error(error,
            "CSV header must be exactly global_id,x,y,z with LF line endings");
        return false;
    }
    x->clear(); y->clear(); z->clear();
    x->reserve((std::size_t)expected_n);
    y->reserve((std::size_t)expected_n);
    z->reserve((std::size_t)expected_n);
    for (int expected_id = 0; expected_id < expected_n; ++expected_id) {
        if (!std::getline(in, line)) {
            palmieri3d_set_error(error,
                                 "centre CSV ended before expected N rows");
            return false;
        }
        const std::size_t comma1 = line.find(',');
        const std::size_t comma2 = comma1 == std::string::npos
            ? std::string::npos : line.find(',', comma1 + 1);
        const std::size_t comma3 = comma2 == std::string::npos
            ? std::string::npos : line.find(',', comma2 + 1);
        if (comma1 == std::string::npos || comma2 == std::string::npos ||
            comma3 == std::string::npos ||
            line.find(',', comma3 + 1) != std::string::npos) {
            palmieri3d_set_error(error,
                "centre CSV row must contain exactly four fields");
            return false;
        }
        int id = -1;
        float xv = 0.0f, yv = 0.0f, zv = 0.0f;
        if (!palmieri3d_parse_int(line.data(), line.data() + comma1, &id) ||
            !palmieri3d_parse_float(line.data() + comma1 + 1,
                                    line.data() + comma2, &xv) ||
            !palmieri3d_parse_float(line.data() + comma2 + 1,
                                    line.data() + comma3, &yv) ||
            !palmieri3d_parse_float(line.data() + comma3 + 1,
                                    line.data() + line.size(), &zv)) {
            palmieri3d_set_error(error,
                                 "centre CSV contains a malformed field");
            return false;
        }
        if (id != expected_id) {
            palmieri3d_set_error(error,
                "global_id must be the ordered sequence 0..N-1");
            return false;
        }
        x->push_back(xv); y->push_back(yv); z->push_back(zv);
    }
    if (std::getline(in, line)) {
        palmieri3d_set_error(error,
                             "centre CSV contains more than expected N rows");
        return false;
    }
    if (in.bad()) {
        palmieri3d_set_error(error, "I/O failure while reading centre CSV");
        return false;
    }
    return palmieri3d_validate_centres(*x, *y, *z, expected_n, side, radius,
                                       diag, error);
}

// Hard-wall channel placement uses the same reproducible random stream and
// minimum centre separation as the periodic initializer. x/y use minimum-image
// distances; z is height above the lower physical wall and lies between the
// wall mid-surfaces at 0 and H. The nominal-radius clearance defines placement;
// the calibrated diffuse seed can extend slightly into the smooth wall field.
inline void channel3d_sequential_centres(
    int n, double side, int height, double radius, std::uint64_t seed,
    std::vector<float>* x, std::vector<float>* y, std::vector<float>* z,
    PalmieriInit3DDiagnostics* diag) {
    const double z_min = radius;
    const double z_max = static_cast<double>(height) - radius;
    if (!x || !y || !z || n < 1 || !(side > 0.0) || !(radius > 0.0) ||
        height <= 0 || z_max < z_min)
        throw std::runtime_error("invalid hard-wall channel initializer parameters");

    x->clear(); y->clear(); z->clear();
    x->reserve(static_cast<std::size_t>(n));
    y->reserve(static_cast<std::size_t>(n));
    z->reserve(static_cast<std::size_t>(n));
    x->push_back(static_cast<float>(0.5 * side));
    y->push_back(static_cast<float>(0.5 * side));
    z->push_back(static_cast<float>(0.5 * height));

    PalmieriInit3DDiagnostics d{};
    std::mt19937_64 rng(seed);
    constexpr std::uint64_t kMaxAttempts = 1000000000ull;
    const double r2 = radius * radius;
    while (static_cast<int>(x->size()) < n) {
        if (++d.candidates_drawn > kMaxAttempts)
            throw std::runtime_error(
                "hard-wall channel sequential placement did not terminate");
        const float cx = static_cast<float>(side * palmieri3d_uniform53(rng));
        const float cy = static_cast<float>(side * palmieri3d_uniform53(rng));
        const float cz = static_cast<float>(
            z_min + (z_max - z_min) * palmieri3d_uniform53(rng));
        bool accept = true;
        for (std::size_t j = 0; j < x->size(); ++j) {
            const double dx = palmieri3d_min_image(
                static_cast<double>(cx) - (*x)[j], side);
            const double dy = palmieri3d_min_image(
                static_cast<double>(cy) - (*y)[j], side);
            const double dz = static_cast<double>(cz) - (*z)[j];
            if (dx * dx + dy * dy + dz * dz < r2) {
                accept = false;
                break;
            }
        }
        if (!accept) {
            ++d.candidates_rejected;
            continue;
        }
        x->push_back(cx); y->push_back(cy); z->push_back(cz);
    }

    for (std::size_t i = 1; i < x->size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            const double dx = palmieri3d_min_image((*x)[i] - (*x)[j], side);
            const double dy = palmieri3d_min_image((*y)[i] - (*y)[j], side);
            const double dz = static_cast<double>((*z)[i]) - (*z)[j];
            d.minimum_periodic_distance = std::min(
                d.minimum_periodic_distance,
                std::sqrt(dx * dx + dy * dy + dz * dz));
        }
    }
    if (diag) *diag = d;
}

inline bool channel3d_validate_centres(
    const std::vector<float>& x, const std::vector<float>& y,
    const std::vector<float>& z, int expected_n, double side, int height,
    double radius, PalmieriCentres3DCsvDiagnostics* diag,
    std::string* error) {
    if (expected_n < 1 || !(side > 0.0) || height <= 0 || !(radius > 0.0)) {
        palmieri3d_set_error(error,
                             "invalid channel centre-table validation parameters");
        return false;
    }
    if (x.size() != static_cast<std::size_t>(expected_n) ||
        y.size() != x.size() || z.size() != x.size()) {
        palmieri3d_set_error(error,
                             "centre-table row count does not equal expected N");
        return false;
    }
    const float expected_xy = static_cast<float>(0.5 * side);
    const float expected_z = static_cast<float>(0.5 * height);
    if (x[0] != expected_xy || y[0] != expected_xy || z[0] != expected_z) {
        palmieri3d_set_error(error,
                             "global_id 0 is not exactly at the channel centre");
        return false;
    }
    const double z_min = radius;
    const double z_max = static_cast<double>(height) - radius;
    if (z_max < z_min) {
        palmieri3d_set_error(error,
                             "channel height is smaller than one cell diameter");
        return false;
    }
    for (int i = 0; i < expected_n; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        if (!std::isfinite(x[k]) || !std::isfinite(y[k]) ||
            !std::isfinite(z[k]) || x[k] < 0.0f || x[k] >= side ||
            y[k] < 0.0f || y[k] >= side || z[k] < z_min || z[k] > z_max) {
            palmieri3d_set_error(error,
                "channel centre is non-finite, outside x/y, or lacks wall clearance");
            return false;
        }
    }
    double minimum = std::numeric_limits<double>::infinity();
    const double r2 = radius * radius;
    for (int i = 1; i < expected_n; ++i) {
        for (int j = 0; j < i; ++j) {
            const double dx = palmieri3d_min_image(
                static_cast<double>(x[static_cast<std::size_t>(i)]) -
                x[static_cast<std::size_t>(j)], side);
            const double dy = palmieri3d_min_image(
                static_cast<double>(y[static_cast<std::size_t>(i)]) -
                y[static_cast<std::size_t>(j)], side);
            const double dz =
                static_cast<double>(z[static_cast<std::size_t>(i)]) -
                z[static_cast<std::size_t>(j)];
            const double d2 = dx * dx + dy * dy + dz * dz;
            minimum = std::min(minimum, std::sqrt(d2));
            if (d2 < r2) {
                palmieri3d_set_error(error,
                    "centre table violates the channel minimum-separation rule");
                return false;
            }
        }
    }
    if (diag) {
        diag->accepted_count = x.size();
        diag->minimum_periodic_distance = minimum;
        diag->table_fnv1a64 = palmieri3d_centre_table_fnv1a64(x, y, z);
    }
    if (error) error->clear();
    return true;
}

inline bool channel3d_read_centres_csv(
    const std::string& path, int expected_n, double side, int height,
    double radius, std::vector<float>* x, std::vector<float>* y,
    std::vector<float>* z, PalmieriCentres3DCsvDiagnostics* diag,
    std::string* error) {
    if (!x || !y || !z || expected_n < 1) {
        palmieri3d_set_error(error, "invalid channel centre CSV reader parameters");
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        palmieri3d_set_error(error, "cannot open centre CSV: " + path);
        return false;
    }
    std::string line;
    if (!std::getline(in, line) || line != "global_id,x,y,z") {
        palmieri3d_set_error(error,
            "CSV header must be exactly global_id,x,y,z with LF line endings");
        return false;
    }
    x->clear(); y->clear(); z->clear();
    x->reserve(static_cast<std::size_t>(expected_n));
    y->reserve(static_cast<std::size_t>(expected_n));
    z->reserve(static_cast<std::size_t>(expected_n));
    for (int expected_id = 0; expected_id < expected_n; ++expected_id) {
        if (!std::getline(in, line)) {
            palmieri3d_set_error(error, "centre CSV ended before expected N rows");
            return false;
        }
        const std::size_t c1 = line.find(',');
        const std::size_t c2 = c1 == std::string::npos
            ? std::string::npos : line.find(',', c1 + 1);
        const std::size_t c3 = c2 == std::string::npos
            ? std::string::npos : line.find(',', c2 + 1);
        if (c1 == std::string::npos || c2 == std::string::npos ||
            c3 == std::string::npos ||
            line.find(',', c3 + 1) != std::string::npos) {
            palmieri3d_set_error(error,
                                 "centre CSV row must contain exactly four fields");
            return false;
        }
        int id = -1;
        float xv = 0.0f, yv = 0.0f, zv = 0.0f;
        if (!palmieri3d_parse_int(line.data(), line.data() + c1, &id) ||
            !palmieri3d_parse_float(line.data() + c1 + 1,
                                    line.data() + c2, &xv) ||
            !palmieri3d_parse_float(line.data() + c2 + 1,
                                    line.data() + c3, &yv) ||
            !palmieri3d_parse_float(line.data() + c3 + 1,
                                    line.data() + line.size(), &zv) ||
            id != expected_id) {
            palmieri3d_set_error(error,
                "centre CSV must contain ordered global_id 0..N-1 and finite numbers");
            return false;
        }
        x->push_back(xv); y->push_back(yv); z->push_back(zv);
    }
    if (std::getline(in, line) || in.bad()) {
        palmieri3d_set_error(error,
                             "centre CSV has extra rows or an I/O failure");
        return false;
    }
    return channel3d_validate_centres(*x, *y, *z, expected_n, side, height,
                                      radius, diag, error);
}

}  // namespace pf3d
