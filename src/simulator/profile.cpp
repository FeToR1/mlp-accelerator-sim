#include "simulator/profile.hpp"

#include <array>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <utility>

#include "accelerator/accelerator.hpp"

namespace simulator {

void write_profile(const accelerator::Accelerator& accel,
                   const std::string& path) {
    std::ofstream out(path);
    if (!out) throw std::runtime_error("Cannot write profile: " + path);
    std::size_t input_bytes = 0, weight_bytes = 0, acc_bytes = 0;
    for (const auto& pe : accel.array.pes) {
        input_bytes += pe.input_sram.size_bytes();
        weight_bytes += pe.weight_sram.size_bytes();
        acc_bytes += pe.acc.size() * sizeof(float);
    }
    const std::array<std::pair<const char*, std::size_t>, 8> memory{{
        {"ram", accel.ram.size_bytes()},
        {"input_sram", input_bytes},
        {"weight_sram", weight_bytes},
        {"acc", acc_bytes},
        {"output_sram", accel.output_sram.size_bytes()},
        {"bias", accel.bias.size_bytes()},
        {"lut_rom", accel.lut.size_bytes()},
        {"command_fifo", accel.queue.size_bytes()},
    }};
    out << std::setprecision(17)
        << "{\n  \"model_time\": " << accel.transactions.model_time()
        << ",\n  \"transactions\": " << accel.transactions.total.count
        << ",\n  \"bytes\": " << accel.transactions.total.bytes
        << ",\n  \"completed_commands\": " << accel.queue.completed
        << ",\n  \"config\": {\"pe_count\": " << accel.config.pe_count
        << ", \"microbatch\": " << accel.config.microbatch_size
        << ", \"output_tile\": " << accel.config.output_tile_size
        << ", \"k_tile\": " << accel.config.k_tile_size
        << "},\n  \"memory\": {";
    std::size_t total_bytes = 0;
    for (const auto& entry : memory) {
        total_bytes += entry.second;
        out << "\n    \"" << entry.first << "\": " << entry.second << ',';
    }
    out << "\n    \"total\": " << total_bytes << "\n  },\n  \"pe\": [";
    for (std::size_t i = 0; i < accel.array.pes.size(); ++i) {
        const auto& pe = accel.array.pes[i];
        if (i) out << ',';
        out << "\n    {\"id\": " << i << ", \"macs\": " << pe.mac_count
            << ", \"capacity_macs\": " << pe.mac_capacity
            << ", \"utilization\": " << pe.utilization() << '}';
    }
    constexpr std::array<const char*,
                         static_cast<std::size_t>(TransactionType::Count)>
        names{"RamToDma",        "DmaToSram",       "SramToDma",
              "DmaToRam",        "InputSramToPe",   "WeightSramToPe",
              "PeToReduction",   "DmaToBias",       "ReductionToBias",
              "BiasToLut",       "LutToOutputSram", "CpuToQueue",
              "QueueToScheduler"};
    out << "\n  ],\n  \"by_type\": {";
    for (std::size_t i = 0; i < names.size(); ++i) {
        if (i) out << ',';
        const auto& stats =
            accel.transactions.stats(static_cast<TransactionType>(i));
        out << "\n    \"" << names[i] << "\": {\"count\": " << stats.count
            << ", \"bytes\": " << stats.bytes << '}';
    }
    out << "\n  }\n}\n";
    out.close();
    if (!out) throw std::runtime_error("Cannot finish profile: " + path);
}

}
