#include "accelerator/scheduler.hpp"

#include <algorithm>
#include <stdexcept>

namespace accelerator {

std::vector<float> Scheduler::execute(const FCCommand& command) {
    if (command.m == 0 || command.m > std::size_t(config.microbatch_size) ||
        command.n == 0 || command.k == 0) {
        throw std::invalid_argument(
            "Scheduler needs one microbatch and positive N/K");
    }
    const auto k_size = std::size_t(command.k);
    const auto k_tile_size = std::size_t(config.k_tile_size);
    const auto n_size = std::size_t(command.n);
    const auto lane_count = std::size_t(config.output_tile_size);
    const auto k_tiles = (k_size + k_tile_size - 1) / k_tile_size;
    const auto local_k_capacity =
        (k_tile_size + config.pe_count - 1) / config.pe_count;
    const auto weight_tile_count =
        local_k_capacity * config.pe_count * config.output_tile_size;

    std::vector<float> result(std::size_t(command.m) * n_size);
    for (std::size_t n0 = 0; n0 < n_size; n0 += lane_count) {
        const auto output_count = std::min(lane_count, n_size - n0);
        array.clear_acc();
        for (std::size_t k0 = 0; k0 < k_size; k0 += k_tile_size) {
            const auto k_count = std::min(k_tile_size, k_size - k0);
            std::vector<float> input;
            input.reserve(std::size_t(command.m) * k_count);
            for (std::size_t m = 0; m < command.m; ++m) {
                const auto address =
                    command.x_addr + (m * k_size + k0) * sizeof(float);
                const auto row = dma.read(address, k_count);
                input.insert(input.end(), row.begin(), row.end());
            }
            const auto tile_index =
                (n0 / lane_count) * k_tiles + k0 / k_tile_size;
            const auto tile_offset = tile_index * weight_tile_count;
            const auto weight_tile_addr =
                command.w_addr + tile_offset * sizeof(float);
            array.load(dma, input, weight_tile_addr, command.m, output_count);
            array.compute();
        }
        const auto reduced =
            reduction.reduce(array.pes, std::size_t(command.m) * lane_count);
        for (std::size_t m = 0; m < command.m; ++m) {
            for (std::size_t lane = 0; lane < output_count; ++lane) {
                result[m * n_size + n0 + lane] = reduced[m * lane_count + lane];
            }
        }
    }
    return result;
}

}
