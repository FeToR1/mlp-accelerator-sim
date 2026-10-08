#include <iostream>
#include <systemc>

#include "accelerator/command.hpp"
#include "accelerator/config.hpp"
#include "accelerator/dma.hpp"
#include "accelerator/ram.hpp"
#include "accelerator/sram.hpp"
#include "simulator/transactions.hpp"

SC_MODULE(BlockTransferDemo) {
    SC_CTOR(BlockTransferDemo, accelerator::DMA& dma, accelerator::SRAM& input,
            std::uint64_t address, std::uint64_t copy_address, std::size_t count)
        : dma(dma), input(input), address(address), copy_address(copy_address),
          count(count) {
        SC_THREAD(run);
    }

   private:
    accelerator::DMA& dma;
    accelerator::SRAM& input;
    const std::uint64_t address;
    const std::uint64_t copy_address;
    const std::size_t count;

    void run() {
        std::cout << "Transfer begin: " << sc_core::sc_time_stamp() << '\n';

        dma.load(address, count, input);

        std::cout << "Load end: " << sc_core::sc_time_stamp() << '\n';
        std::cout << "Input SRAM:";
        for (float value : input.read()) {
            std::cout << ' ' << value;
        }
        std::cout << '\n';

        dma.store(copy_address, count, input);
        std::cout << "Store end: " << sc_core::sc_time_stamp() << '\n';
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
    const auto copy_addr = ram.allocate(input_count);
    ram.write(command.x_addr, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});

    simulator::Transactions transactions;
    accelerator::DMA dma("dma", ram, transactions);
    accelerator::SRAM input("input_sram", input_count);
    BlockTransferDemo transfer("transfer", dma, input, command.x_addr, copy_addr,
                               input_count);

    std::cout << "PE=" << config.pe_count
              << ", microbatch=" << config.microbatch_size
              << ", output_tile=" << config.output_tile_size
              << ", K_tile=" << config.k_tile_size << '\n';
    std::cout << "FC: M=" << command.m << ", N=" << command.n
              << ", K=" << command.k << '\n';
    std::cout << "X address=" << command.x_addr
              << ", copy address=" << copy_addr
              << ", RAM bytes=" << ram.size_bytes()
              << ", Input SRAM bytes=" << input.size_bytes() << '\n';

    sc_core::sc_start();

    std::cout << "RAM copy:";
    for (float value : ram.read(copy_addr, input_count)) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';

    std::cout << "SystemC time=" << sc_core::sc_time_stamp() << '\n';
    std::cout << "Model time=" << transactions.model_time()
              << ", transactions=" << transactions.total.count
              << ", transferred bytes=" << transactions.total.bytes << '\n';

    const auto& ram_dma = transactions.stats(simulator::TransactionType::RamToDma);
    const auto& dma_sram = transactions.stats(simulator::TransactionType::DmaToSram);
    const auto& sram_dma = transactions.stats(simulator::TransactionType::SramToDma);
    const auto& dma_ram = transactions.stats(simulator::TransactionType::DmaToRam);
    std::cout << "RAM -> DMA: count=" << ram_dma.count
              << ", bytes=" << ram_dma.bytes << '\n';
    std::cout << "DMA -> SRAM: count=" << dma_sram.count
              << ", bytes=" << dma_sram.bytes << '\n';
    std::cout << "SRAM -> DMA: count=" << sram_dma.count
              << ", bytes=" << sram_dma.bytes << '\n';
    std::cout << "DMA -> RAM: count=" << dma_ram.count
              << ", bytes=" << dma_ram.bytes << '\n';
    return 0;
}
