#include <iostream>
#include <stdexcept>
#include <string>
#include <systemc>
#include <vector>

#include "accelerator/command.hpp"
#include "accelerator/config.hpp"
#include "accelerator/dma.hpp"
#include "accelerator/packed_weights.hpp"
#include "accelerator/pe_array.hpp"
#include "accelerator/ram.hpp"
#include "accelerator/reduction.hpp"
#include "accelerator/scheduler.hpp"
#include "simulator/transactions.hpp"

SC_MODULE(PEArrayDemo) {
    SC_CTOR(PEArrayDemo, accelerator::Scheduler & scheduler,
            accelerator::PEArray & array, const accelerator::FCCommand& command)
        : scheduler(scheduler), array(array), command(command) {
        SC_THREAD(run);
    }

   private:
    accelerator::Scheduler& scheduler;
    accelerator::PEArray& array;
    const accelerator::FCCommand& command;

    void run() {
        std::cout << "Microbatches + N/K tiles begin: "
                  << sc_core::sc_time_stamp() << '\n';
        const auto result = scheduler.execute(command);
        std::cout << "Microbatches + N/K tiles end: "
                  << sc_core::sc_time_stamp() << '\n';
        for (std::size_t pe_id = 0; pe_id < array.pes.size(); ++pe_id) {
            const auto& pe = array.pes[pe_id];
            std::cout << "PE[" << pe_id << "] MAC ops=" << pe.mac_count << '\n';
        }

        for (std::size_t m = 0; m < command.m; ++m) {
            std::cout << "Reduced[" << m << "]:";
            for (std::size_t lane = 0; lane < command.n; ++lane) {
                std::cout << ' ' << result[m * command.n + lane];
            }
            std::cout << '\n';
        }
    }
};

int sc_main(int argc, char* argv[]) {
    accelerator::AcceleratorConfig config;
    if (argc > 1) {
        config.pe_count = std::stoi(argv[1]);
    }
    config.validate();

    accelerator::FCCommand command;
    const auto k_size = argc > 2 ? std::stoi(argv[2]) : config.k_tile_size + 1;
    const auto n_size =
        argc > 3 ? std::stoi(argv[3]) : config.output_tile_size + 1;
    const auto m_size =
        argc > 4 ? std::stoi(argv[4]) : config.microbatch_size + 1;
    if (k_size <= 0 || n_size <= 0 || m_size <= 0) {
        throw std::invalid_argument("M, N and K must be positive");
    }
    command.k = k_size;
    command.n = n_size;
    command.m = m_size;

    accelerator::WeightMatrix weights(command.k, command.n);
    const std::vector<float> regular_weights{1.0f, 2.0f, -1.0f, 0.0f};
    const std::vector<float> last_weights{4.0f, 1.0f, -1.0f, 1.0f};
    for (std::size_t n = 0; n < command.n; ++n) {
        weights.col(n).setConstant(regular_weights[n % regular_weights.size()]);
        weights(command.k - 1, n) = last_weights[n % last_weights.size()];
    }
    const auto packed = accelerator::pack_weights(weights, config);
    std::cout << "Packed weights: N tiles=" << packed.n_tiles
              << ", K tiles=" << packed.k_tiles
              << ", local K=" << packed.local_k_size
              << ", bytes=" << packed.values.size() * sizeof(float) << '\n';
    for (std::size_t pe = 0; pe < packed.pe_count; ++pe) {
        const auto base = packed.offset(0, 0, pe);
        std::cout << "Wpacked[0][0][" << pe << "][0]:";
        for (std::size_t lane = 0; lane < packed.lane_count; ++lane) {
            std::cout << ' ' << packed.values[base + lane];
        }
        std::cout << '\n';
    }

    accelerator::ExternalRAM ram("ram");
    const auto input_count = std::size_t(command.m) * command.k;
    std::vector<float> input(input_count);
    for (std::size_t m = 0; m < command.m; ++m) {
        for (std::size_t k = 0; k < command.k; ++k) {
            input[m * command.k + k] = float(m + 1);
        }
    }
    command.x_addr = ram.allocate(input_count);
    command.w_addr = ram.allocate(packed.values.size());
    ram.write(command.x_addr, input);
    ram.write(command.w_addr, packed.values);

    simulator::Transactions transactions;
    accelerator::DMA dma("dma", ram, transactions);
    accelerator::PEArray array("array", config, transactions);
    accelerator::Reduction reduction("reduction", transactions);
    accelerator::Scheduler scheduler("scheduler", config, dma, array,
                                     reduction);
    PEArrayDemo demo("demo", scheduler, array, command);

    std::cout << "PE=" << config.pe_count
              << ", microbatch=" << config.microbatch_size
              << ", output_tile=" << config.output_tile_size
              << ", K_tile=" << config.k_tile_size << '\n';
    std::cout << "MAC lanes per PE=" << config.output_tile_size
              << ", ACC bytes per PE="
              << array.pes[0].acc.size() * sizeof(float) << '\n';
    std::cout << "FC: M=" << command.m << ", N=" << command.n
              << ", K=" << command.k << '\n';
    std::cout << "X address=" << command.x_addr
              << ", W address=" << command.w_addr
              << ", RAM bytes=" << ram.size_bytes() << '\n';

    sc_core::sc_start();

    std::cout << "SystemC time=" << sc_core::sc_time_stamp() << '\n';
    std::cout << "Model time=" << transactions.model_time()
              << ", transactions=" << transactions.total.count
              << ", transferred bytes=" << transactions.total.bytes << '\n';

    const auto& ram_dma =
        transactions.stats(simulator::TransactionType::RamToDma);
    const auto& dma_sram =
        transactions.stats(simulator::TransactionType::DmaToSram);
    const auto& sram_dma =
        transactions.stats(simulator::TransactionType::SramToDma);
    const auto& dma_ram =
        transactions.stats(simulator::TransactionType::DmaToRam);
    const auto& input_pe =
        transactions.stats(simulator::TransactionType::InputSramToPe);
    const auto& weight_pe =
        transactions.stats(simulator::TransactionType::WeightSramToPe);
    const auto& pe_reduction =
        transactions.stats(simulator::TransactionType::PeToReduction);
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
    std::cout << "PE -> reduction: count=" << pe_reduction.count
              << ", bytes=" << pe_reduction.bytes << '\n';
    return 0;
}
