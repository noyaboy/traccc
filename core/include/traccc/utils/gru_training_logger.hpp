#pragma once
#include <fstream>
#include <mutex>
#include <vector>
#include "traccc/definitions/primitives.hpp"

namespace traccc {

class gru_training_logger {
public:
    static void write_row(const std::vector<traccc::scalar>& values);

private:
    static std::ofstream& stream();
    static std::mutex& mutex();
};

}  // namespace traccc
