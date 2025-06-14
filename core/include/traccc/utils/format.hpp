/** TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2025 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#pragma once

#if __has_include(<format>) && defined(__cpp_lib_format)
#include <format>
namespace traccc {
using std::format;
}  // namespace traccc
#else
#include <fmt/format.h>
namespace traccc {
using fmt::format;
}  // namespace traccc
#endif

