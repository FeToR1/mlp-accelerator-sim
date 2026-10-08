#include <iostream>

#include "accelerator/command.hpp"
#include "accelerator/config.hpp"
#include "accelerator/ram.hpp"

int main() {
    accelerator::AcceleratorConfig config;
    config.validate();

    accelerator::FCCommand command;
    command.m = 2;
    command.n = 2;
    command.k = 3;

    accelerator::ExternalRAM ram;
    const auto input_count = std::size_t(command.m) * command.k;
    command.x_addr = ram.allocate(input_count);
    ram.write(command.x_addr, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    std::cout << "PE=" << config.pe_count
              << ", microbatch=" << config.microbatch_size
              << ", output_tile=" << config.output_tile_size
              << ", K_tile=" << config.k_tile_size << '\n';
    std::cout << "FC: M=" << command.m << ", N=" << command.n
              << ", K=" << command.k << '\n';
    std::cout << "X address=" << command.x_addr
              << ", RAM bytes=" << ram.size_bytes() << '\n';
    std::cout << "X:";
    for (float value : ram.read(command.x_addr, input_count)) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
}
