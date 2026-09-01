#pragma once

// Strict input contract for a realized two-dimensional configuration used to
// seed a substrate slab. This is intentionally separate from Palmieri's
// initial-proposal validation: passive relaxation can displace cell zero and
// bring centroids closer than the proposal exclusion distance.

#include "palmieri_initializer.hpp"

#include <charconv>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <limits>
#include <string>
#include <system_error>
#include <vector>

namespace pf {

struct RelaxedCentresCsvDiagnostics {
    std::size_t accepted_count = 0;
    double minimum_periodic_distance = std::numeric_limits<double>::infinity();
    int declared_source_side = 0;
};

inline bool relaxed_parse_int64(const char* first, const char* last,
                                std::int64_t* value) {
    const auto parsed = std::from_chars(first, last, *value, 10);
    return parsed.ec == std::errc{} && parsed.ptr == last;
}

inline void relaxed_normalize_csv_line(std::string* line,
                                       bool first_line = false) {
    if (!line) return;
    if (!line->empty() && line->back() == '\r') line->pop_back();
    if (first_line && line->size() >= 3 &&
        static_cast<unsigned char>((*line)[0]) == 0xefu &&
        static_cast<unsigned char>((*line)[1]) == 0xbbu &&
        static_cast<unsigned char>((*line)[2]) == 0xbfu)
        line->erase(0, 3);
}

// Coordinates must be wrapped into the target box. Rows retain the simulator's
// canonical global IDs 0..N-1 so identity-based parameter assignment is not
// changed by table order. Close centroids are handled by slab relaxation.
inline bool relaxed_read_centres_csv(
    const std::string& path, int expected_n, double side,
    std::vector<std::int64_t>* global_id, std::vector<float>* x,
    std::vector<float>* y, RelaxedCentresCsvDiagnostics* diag,
    std::string* error, bool require_source_side = false) {
    if (!global_id || !x || !y || expected_n < 1 || !(side > 0.0)) {
        palmieri_set_error(error,
                           "invalid relaxed-centre CSV reader parameters");
        return false;
    }
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        palmieri_set_error(error,
                           "cannot open relaxed-centre CSV: " + path);
        return false;
    }
    std::string line;
    if (!std::getline(in, line)) {
        palmieri_set_error(error, "centre CSV is empty");
        return false;
    }
    relaxed_normalize_csv_line(&line, true);
    if (line != "global_id,x,y") {
        palmieri_set_error(
            error, "CSV header must be exactly global_id,x,y");
        return false;
    }

    if (!std::getline(in, line)) {
        palmieri_set_error(error, "centre CSV has no data rows");
        return false;
    }
    relaxed_normalize_csv_line(&line);
    int declared_source_side = 0;
    bool buffered_first_row = true;
    constexpr const char* kSourceSide = "# source_L=";
    if (line.rfind(kSourceSide, 0) == 0) {
        const char* first = line.data() + std::char_traits<char>::length(kSourceSide);
        const char* last = line.data() + line.size();
        std::int64_t parsed_side = 0;
        if (!relaxed_parse_int64(first, last, &parsed_side) ||
            parsed_side <= 0 ||
            parsed_side > std::numeric_limits<int>::max()) {
            palmieri_set_error(error, "centre CSV has an invalid source_L");
            return false;
        }
        declared_source_side = static_cast<int>(parsed_side);
        if (static_cast<double>(declared_source_side) != side) {
            palmieri_set_error(
                error, "centre CSV source_L does not match the target box");
            return false;
        }
        buffered_first_row = false;
    } else if (require_source_side) {
        palmieri_set_error(
            error, "one-layer channel centre CSV requires # source_L=L");
        return false;
    }

    global_id->clear();
    x->clear();
    y->clear();
    global_id->reserve((std::size_t)expected_n);
    x->reserve((std::size_t)expected_n);
    y->reserve((std::size_t)expected_n);
    for (int row = 0; row < expected_n; ++row) {
        if ((!buffered_first_row || row != 0) && !std::getline(in, line)) {
            palmieri_set_error(error,
                               "centre CSV ended before expected N rows");
            return false;
        }
        relaxed_normalize_csv_line(&line);
        const std::size_t comma1 = line.find(',');
        const std::size_t comma2 = comma1 == std::string::npos
            ? std::string::npos : line.find(',', comma1 + 1);
        if (comma1 == std::string::npos || comma2 == std::string::npos ||
            line.find(',', comma2 + 1) != std::string::npos) {
            palmieri_set_error(
                error, "centre CSV row must contain exactly three fields");
            return false;
        }
        std::int64_t id = 0;
        float xv = 0.0f;
        float yv = 0.0f;
        if (!relaxed_parse_int64(line.data(), line.data() + comma1, &id) ||
            !palmieri_parse_float(line.data() + comma1 + 1,
                                  line.data() + comma2, &xv) ||
            !palmieri_parse_float(line.data() + comma2 + 1,
                                  line.data() + line.size(), &yv)) {
            palmieri_set_error(error,
                               "centre CSV contains a malformed field");
            return false;
        }
        if (id != static_cast<std::int64_t>(row)) {
            palmieri_set_error(
                error, "centre CSV rows must have global_id 0..N-1 in order");
            return false;
        }
        if (!std::isfinite(xv) || !std::isfinite(yv) ||
            !(xv >= 0.0f && xv < side) ||
            !(yv >= 0.0f && yv < side)) {
            palmieri_set_error(
                error, "centre is non-finite or outside the target [0,L) box");
            return false;
        }
        global_id->push_back(id);
        x->push_back(xv);
        y->push_back(yv);
    }
    if (std::getline(in, line)) {
        relaxed_normalize_csv_line(&line);
        palmieri_set_error(error,
                           "centre CSV contains more than expected N rows");
        return false;
    }
    if (in.bad()) {
        palmieri_set_error(error, "I/O failure while reading centre CSV");
        return false;
    }

    double minimum = std::numeric_limits<double>::infinity();
    for (std::size_t i = 1; i < x->size(); ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            minimum = std::min(
                minimum,
                std::hypot(
                    palmieri_min_image((double)(*x)[i] - (double)(*x)[j], side),
                    palmieri_min_image((double)(*y)[i] - (double)(*y)[j], side)));
        }
    }
    if (diag) {
        diag->accepted_count = x->size();
        diag->minimum_periodic_distance = minimum;
        diag->declared_source_side = declared_source_side;
    }
    if (error) error->clear();
    return true;
}

}  // namespace pf
