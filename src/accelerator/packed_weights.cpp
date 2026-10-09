#include "accelerator/packed_weights.hpp"

#include <stdexcept>

namespace accelerator {

std::size_t PackedWeights::offset(std::size_t n_tile, std::size_t k_tile,
                                  std::size_t pe) const {
    return ((n_tile * k_tiles + k_tile) * pe_count + pe) * local_k_size *
           lane_count;
}

PackedWeights pack_weights(const Eigen::Ref<const WeightMatrix>& weights,
                           const AcceleratorConfig& config) {
    config.validate();
    if (weights.rows() == 0 || weights.cols() == 0) {
        throw std::invalid_argument("Weights must be nonempty");
    }
    const auto k_size = std::size_t(weights.rows());
    const auto n_size = std::size_t(weights.cols());
    const auto k_tile_size = std::size_t(config.k_tile_size);
    const auto lane_count = std::size_t(config.output_tile_size);
    const auto pe_count = std::size_t(config.pe_count);
    PackedWeights packed{
        (n_size + lane_count - 1) / lane_count,
        (k_size + k_tile_size - 1) / k_tile_size,
        pe_count,
        (k_tile_size + pe_count - 1) / pe_count,
        lane_count,
        {},
    };
    packed.values.resize(packed.n_tiles * packed.k_tiles * pe_count *
                             packed.local_k_size * lane_count,
                         0.0f);

    for (std::size_t k = 0; k < k_size; ++k) {
        const auto k_tile = k / k_tile_size;
        const auto tile_k = k % k_tile_size;
        const auto pe = tile_k % pe_count;
        const auto local_k = tile_k / pe_count;
        for (std::size_t n = 0; n < n_size; ++n) {
            const auto n_tile = n / lane_count;
            const auto lane = n % lane_count;
            const auto base = packed.offset(n_tile, k_tile, pe);
            packed.values[base + local_k * lane_count + lane] = weights(k, n);
        }
    }
    return packed;
}

}
