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
        pe.clear_acc();

        dma.load(command.x_addr, std::size_t(command.m) * command.k, input);
        dma.load(command.w_addr, std::size_t(command.k) * command.n, weights);
        std::cout << "DMA loads end: " << sc_core::sc_time_stamp() << '\n';

        pe.load(input.read(), weights.read(), command.m);
        const auto compute_begin = sc_core::sc_time_stamp();
        pe.compute();
        std::cout << "Compute: " << compute_begin << " -> "
                  << sc_core::sc_time_stamp() << '\n';
        for (std::size_t m = 0; m < command.m; ++m) {
            std::cout << "ACC[" << m << "]:";
            for (std::size_t lane = 0; lane < command.n; ++lane) {
                std::cout << ' ' << pe.acc[m * command.n + lane];
            }
            std::cout << '\n';
        }
        std::cout << "MAC ops=" << pe.mac_count << '\n';
    }
};

int sc_main(int, char*[]) {
    accelerator::AcceleratorConfig config;
    config.pe_count = 1;
    config.validate();

    accelerator::FCCommand command;
    command.m = 2;
    command.n = config.output_tile_size;
    command.k = 3;

    accelerator::ExternalRAM ram("ram");
    const auto input_count = std::size_t(command.m) * command.k;
    const auto weight_count = std::size_t(command.k) * command.n;
    command.x_addr = ram.allocate(input_count);
    command.w_addr = ram.allocate(weight_count);
    ram.write(command.x_addr, {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f});
    ram.write(command.w_addr, {
        4.0f, 1.0f, -1.0f, 0.0f,
        5.0f, 0.0f, -1.0f, 1.0f,
        6.0f, 1.0f, -1.0f, 0.0f,
    });

    simulator::Transactions transactions;
    accelerator::DMA dma("dma", ram, transactions);
    accelerator::SRAM input("input_sram", input_count);
    accelerator::SRAM weights("weight_sram", weight_count);
    accelerator::PE pe("pe", transactions, config.output_tile_size,
                       config.microbatch_size);
    PEDemo demo("demo", dma, input, weights, pe, command);

    std::cout << "PE=" << config.pe_count
              << ", microbatch=" << config.microbatch_size
              << ", output_tile=" << config.output_tile_size
              << ", K_tile=" << config.k_tile_size << '\n';
    std::cout << "MAC lanes per PE=" << config.output_tile_size
              << ", ACC bytes=" << pe.acc.size() * sizeof(float) << '\n';
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
