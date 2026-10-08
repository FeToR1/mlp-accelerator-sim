#include <iostream>

#include "accelerator/command.hpp"
#include "accelerator/config.hpp"

int main() {
    accelerator::AcceleratorConfig config;
    config.validate();

    accelerator::FCCommand command;
    command.m = 8;
    command.n = 128;
    command.k = 784;
    command.x_addr = 0x1000;
    command.w_addr = 0x10000;
    command.bias_addr = 0x80000;
    command.y_addr = 0x90000;

    std::cout << "PE=" << config.pe_count
              << ", microbatch=" << config.microbatch_size
              << ", output_tile=" << config.output_tile_size
              << ", K_tile=" << config.k_tile_size << '\n';
    std::cout << "FC: M=" << command.m << ", N=" << command.n
              << ", K=" << command.k << '\n';
    std::cout << std::hex << "RAM offsets: X=0x" << command.x_addr << ", W=0x"
              << command.w_addr << ", bias=0x" << command.bias_addr << ", Y=0x"
              << command.y_addr << '\n';
}
