#pragma once
#include <cstdio>
#include <initializer_list>
#include <string>

namespace pf {
// Create a new binary stream atomically. Never truncate an existing output;
// each recorder invocation/restart segment must use its own path.
std::FILE* open_new_binary_file(const std::string& path);
bool distinct_output_paths(std::initializer_list<std::string> paths);
} // namespace pf
