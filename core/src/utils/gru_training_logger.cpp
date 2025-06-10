#include "traccc/utils/gru_training_logger.hpp"

namespace traccc {

std::ofstream& gru_training_logger::stream() {
    static std::ofstream s("gru_training_data.csv");
    return s;
}

std::mutex& gru_training_logger::mutex() {
    static std::mutex m;
    return m;
}

void gru_training_logger::write_row(const std::vector<traccc::scalar>& values) {
    std::lock_guard<std::mutex> lock(mutex());
    std::ofstream& out = stream();
    for (std::size_t i = 0; i < values.size(); ++i) {
        out << values[i];
        if (i + 1 != values.size()) {
            out << ',';
        }
    }
    out << '\n';
}

}  // namespace traccc
