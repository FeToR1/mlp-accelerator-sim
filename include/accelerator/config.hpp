#pragma once

#include <stdexcept>

namespace accelerator {

// Set before starting a simulation. All components use the same configuration.
struct AcceleratorConfig {
    int pe_count = 4;
    int microbatch_size = 8;
    int output_tile_size = 4;
    int k_tile_size = 64;

    void validate() const {
        if (pe_count <= 0) {
            throw std::invalid_argument("pe_count must be positive");
        }
        if (microbatch_size <= 0) {
            throw std::invalid_argument("microbatch_size must be positive");
        }
        if (output_tile_size <= 0) {
            throw std::invalid_argument("output_tile_size must be positive");
        }
        if (k_tile_size <= 0) {
            throw std::invalid_argument("k_tile_size must be positive");
        }
    }
};

}  // namespace accelerator
