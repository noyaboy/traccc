/**
 * TRACCC library, part of the ACTS project (R&D line)
 *
 * (c) 2022 CERN for the benefit of the ACTS project
 *
 * Mozilla Public License Version 2.0
 */

#include <cuda_runtime.h>
#include <gtest/gtest.h>

int main(int argc, char** argv) {
    testing::InitGoogleTest(&argc, argv);

    int r = RUN_ALL_TESTS();

    // When Google Test is only listing the available tests, avoid calling
    // ``cudaDeviceReset``. Some CUDA setups report an error code on this call
    // without a device having been initialised, causing the test discovery
    // phase in CMake to fail. Skipping the reset in this case prevents
    // spurious errors while leaving normal test execution unchanged.
    if (!testing::GTEST_FLAG(list_tests)) {
        cudaError_t cErr = cudaDeviceReset();

        if (cErr == cudaSuccess || cErr == cudaErrorNoDevice ||
            cErr == cudaErrorInsufficientDriver) {
            return r;
        } else {
            return static_cast<int>(cErr);
        }
    }

    return r;
}
