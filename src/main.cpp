#include <iostream>
#include <systemc>

#include "accelerator/command.hpp"
#include "accelerator/config.hpp"
#include "accelerator/ram.hpp"

SC_MODULE(BlockTransferDemo) {
    SC_CTOR(BlockTransferDemo, accelerator::ExternalRAM& ram,
            std::uint64_t address, std::size_t count)
        : ram(ram), address(address), count(count) {
        SC_THREAD(run);
    }

   private:
    accelerator::ExternalRAM& ram;
    const std::uint64_t address;
    const std::size_t count;
    const sc_core::sc_time transfer_time{1, sc_core::SC_NS};

    void run() {
        std::cout << "Transfer begin: " << sc_core::sc_time_stamp() << '\n';

        sc_core::wait(transfer_time);
        const auto input = ram.read(address, count);

        std::cout << "Transfer end: " << sc_core::sc_time_stamp()
                  << ", model time=" << sc_core::sc_time_stamp() / transfer_time
                  << '\n';
        std::cout << "X:";
        for (float value : input) {
            std::cout << ' ' << value;
        }
        std::cout << '\n';
    }
};

int sc_main(int, char*[]) {
    accelerator::AcceleratorConfig config;
    config.validate();

    accelerator::FCCommand command;
    command.m = 2;
    command.n = 2;
    command.k = 3;

    accelerator::ExternalRAM ram("ram");
    const auto input_count = std::size_t(command.m) * command.k;
    command.x_addr = ram.allocate(input_count);
    ram.write(command.x_addr, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    BlockTransferDemo transfer("transfer", ram, command.x_addr, input_count);

    std::cout << "PE=" << config.pe_count
              << ", microbatch=" << config.microbatch_size
              << ", output_tile=" << config.output_tile_size
              << ", K_tile=" << config.k_tile_size << '\n';
    std::cout << "FC: M=" << command.m << ", N=" << command.n
              << ", K=" << command.k << '\n';
    std::cout << "X address=" << command.x_addr
              << ", RAM bytes=" << ram.size_bytes() << '\n';

    sc_core::sc_start();

    std::cout << "SystemC time=" << sc_core::sc_time_stamp() << '\n';
    return 0;
}
