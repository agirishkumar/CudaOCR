#pragma once

#include <gtest/gtest.h>

#include <cstdlib>

#include "cudaocr/core/host_image.hpp"

namespace cudaocr::testutil {

// Byte-exact comparison with a useful first-mismatch message.
inline ::testing::AssertionResult imagesEqual(const HostImage& a, const HostImage& b) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels)
        return ::testing::AssertionFailure() << "shape mismatch";
    for (int y = 0; y < a.height; ++y)
        for (int x = 0; x < a.width; ++x)
            for (int c = 0; c < a.channels; ++c)
                if (a.at(x, y, c) != b.at(x, y, c))
                    return ::testing::AssertionFailure()
                           << "first mismatch at (" << x << "," << y << ") ch " << c << ": "
                           << int(a.at(x, y, c)) << " vs " << int(b.at(x, y, c));
    return ::testing::AssertionSuccess();
}

// Tolerance comparison: at most maxOutlierFrac of pixels may differ by more than tol.
inline ::testing::AssertionResult imagesClose(const HostImage& a, const HostImage& b, int tol,
                                              double maxOutlierFrac) {
    if (a.width != b.width || a.height != b.height || a.channels != b.channels)
        return ::testing::AssertionFailure() << "shape mismatch";
    size_t outliers = 0;
    for (size_t i = 0; i < a.data.size(); ++i)
        if (std::abs(int(a.data[i]) - int(b.data[i])) > tol) ++outliers;
    const double frac = double(outliers) / double(a.data.size());
    if (frac > maxOutlierFrac)
        return ::testing::AssertionFailure()
               << outliers << " pixels (" << frac * 100 << "%) differ by more than " << tol;
    return ::testing::AssertionSuccess();
}

}  // namespace cudaocr::testutil
