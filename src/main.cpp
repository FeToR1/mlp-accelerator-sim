#include <iostream>
#include <systemc>

#include "accelerator/command.hpp"
#include "accelerator/config.hpp"
#include "accelerator/dma.hpp"
#include "accelerator/pe.hpp"
#include "accelerator/ram.hpp"
#include "accelerator/sram.hpp"
#include "simulator/transactions.hpp"

SC_MODULE(PEDemo) {
    SC_CTOR(PEDemo, accelerator::DMA& dma, accelerator::SRAM& input,
            accelerator::SRAM& weights, accelerator::PE& pe,
            const accelerator::FCCommand& command)
        : dma(dma), input(input), weights(weights), pe(pe), command(command) {
        SC_THREAD(run);
    }

   private:
    accelerator::DMA& dma;
    accelerator::SRAM& input;
    accelerator::SRAM& weights;
    accelerator::PE& pe;
    const accelerator::FCCommand& command;

    void run() {
        std::cout << "Load begin: " << sc_core::sc_time_stamp() << '\n';

        dma.load(command.x_addr, command.k, input);
        dma.load(command.w_addr, command.k, weights);
        std::cout << "DMA loads end: " << sc_core::sc_time_stamp() << '\n';

        pe.load(input.read(), weights.read());
        const auto compute_begin = sc_core::sc_time_stamp();
        pe.compute();
        std::cout << "Compute: " << compute_begin << " -> "
                  << sc_core::sc_time_stamp() << '\n';
        std::cout << "PE acc=" << pe.acc << ", MAC=" << pe.mac_count << '\n';
    }
};

int sc_main(int, char*[]) {
    accelerator::AcceleratorConfig config;
    config.pe_count = 1;
    config.validate();

    accelerator::FCCommand command;
    command.m = 1;
    command.n = 1;
    command.k = 3;

    accelerator::ExternalRAM ram("ram");
    command.x_addr = ram.allocate(command.k);
    command.w_addr = ram.allocate(command.k);
    ram.write(command.x_addr, {1.0f, 2.0f, 3.0f});
    ram.write(command.w_addr, {4.0f, 5.0f, 6.0f});

    simulator::Transactions transactions;
    accelerator::DMA dma("dma", ram, transactions);
    accelerator::SRAM input("input_sram", command.k);
    accelerator::SRAM weights("weight_sram", command.k);
    accelerator::PE pe("pe", transactions);
    PEDemo demo("demo", dma, input, weights, pe, command);

    std::cout << "PE=" << config.pe_count
              << ", microbatch=" << config.microbatch_size
              << ", output_tile=" << config.output_tile_size
              << ", K_tile=" << config.k_tile_size << '\n';
    std::cout << "FC: M=" << command.m << ", N=" << command.n
              << ", K=" << command.k << '\n';
    std::cout << "X address=" << command.x_addr
              << ", W address=" << command.w_addr
              << ", RAM bytes=" << ram.size_bytes()
              << ", Input SRAM bytes=" << input.size_bytes()
              << ", Weight SRAM bytes=" << weights.size_bytes() << '\n';

    sc_core::sc_start();

    std::cout << "SystemC time=" << sc_core::sc_time_stamp() << '\n';
    std::cout << "Model time=" << transactions.model_time()
              << ", transactions=" << transactions.total.count
              << ", transferred bytes=" << transactions.total.bytes << '\n';

    const auto& ram_dma = transactions.stats(simulator::TransactionType::RamToDma);
    const auto& dma_sram = transactions.stats(simulator::TransactionType::DmaToSram);
    const auto& sram_dma = transactions.stats(simulator::TransactionType::SramToDma);
    const auto& dma_ram = transactions.stats(simulator::TransactionType::DmaToRam);
    const auto& input_pe =
        transactions.stats(simulator::TransactionType::InputSramToPe);
    const auto& weight_pe =
        transactions.stats(simulator::TransactionType::WeightSramToPe);
    std::cout << "RAM -> DMA: count=" << ram_dma.count
              << ", bytes=" << ram_dma.bytes << '\n';
    std::cout << "DMA -> SRAM: count=" << dma_sram.count
              << ", bytes=" << dma_sram.bytes << '\n';
    std::cout << "SRAM -> DMA: count=" << sram_dma.count
              << ", bytes=" << sram_dma.bytes << '\n';
    std::cout << "DMA -> RAM: count=" << dma_ram.count
              << ", bytes=" << dma_ram.bytes << '\n';
    std::cout << "Input SRAM -> PE: count=" << input_pe.count
              << ", bytes=" << input_pe.bytes << '\n';
    std::cout << "Weight SRAM -> PE: count=" << weight_pe.count
              << ", bytes=" << weight_pe.bytes << '\n';
    return 0;
}
