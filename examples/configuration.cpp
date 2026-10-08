#include <iostream>

#include "accelerator/config.hpp"

int main() {
    for (int pe_count : {1, 2, 4, 8}) {
        accelerator::AcceleratorConfig config;
        config.pe_count = pe_count;
        config.validate();

        std::cout << "PE=" << config.pe_count
                  << ", microbatch=" << config.microbatch_size
                  << ", output_tile=" << config.output_tile_size
                  << ", K_tile=" << config.k_tile_size << '\n';
    }
}
