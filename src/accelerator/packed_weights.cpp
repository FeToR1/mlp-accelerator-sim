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

    for (std::size_t n_tile = 0; n_tile < packed.n_tiles; ++n_tile) {
        for (std::size_t k_tile = 0; k_tile < packed.k_tiles; ++k_tile) {
            for (std::size_t pe = 0; pe < pe_count; ++pe) {
                const auto base = packed.offset(n_tile, k_tile, pe);
                for (std::size_t local_k = 0; local_k < packed.local_k_size;
                     ++local_k) {
                    const auto tile_k = local_k * pe_count + pe;
                    const auto k = k_tile * k_tile_size + tile_k;
                    for (std::size_t lane = 0; lane < lane_count; ++lane) {
                        const auto n = n_tile * lane_count + lane;
                        if (tile_k < k_tile_size && k < k_size && n < n_size) {
                            packed.values[base + local_k * lane_count + lane] =
                                weights(k, n);
                        }
                    }
                }
            }
        }
    }
    return packed;
}

}
