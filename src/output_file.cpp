#include "output_file.hpp"
#include <fcntl.h>
#include <filesystem>
#include <vector>
#ifdef _WIN32
#include <io.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif

namespace pf {
bool distinct_output_paths(std::initializer_list<std::string> paths) {
    std::vector<std::filesystem::path> resolved;
    for (const auto& path : paths) {
        if (path.empty())
            continue;
        std::error_code error;
        const auto absolute = std::filesystem::absolute(path, error);
        const auto canonical =
            error ? absolute : std::filesystem::weakly_canonical(absolute, error);
        if (error) {
            std::fprintf(stderr, "[fatal] cannot resolve output path %s: %s\n", path.c_str(),
                         error.message().c_str());
            return false;
        }
        for (const auto& previous : resolved) {
            const bool alias = std::filesystem::equivalent(canonical, previous, error);
            if (canonical == previous || (!error && alias)) {
                std::fprintf(stderr, "[fatal] output files must use distinct paths\n");
                return false;
            }
        }
        resolved.push_back(canonical);
    }
    return true;
}

std::FILE* open_new_binary_file(const std::string& path) {
#ifdef _WIN32
    const int fd =
        _open(path.c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
    if (fd < 0)
        return nullptr;
    std::FILE* file = _fdopen(fd, "wb");
    if (!file)
        _close(fd);
#else
    const int fd = ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0)
        return nullptr;
    std::FILE* file = fdopen(fd, "wb");
    if (!file)
        ::close(fd);
#endif
    return file;
}
} // namespace pf
