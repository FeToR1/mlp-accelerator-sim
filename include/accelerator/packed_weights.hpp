#pragma once

#include <Eigen/Core>
#include <cstddef>
#include <vector>

#include "accelerator/config.hpp"

namespace accelerator {

using WeightMatrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

struct PackedWeights {
    std::size_t n_tiles;
    std::size_t k_tiles;
    std::size_t pe_count;
    std::size_t local_k_size;
    std::size_t lane_count;
    std::vector<float> values;

    std::size_t offset(std::size_t n_tile, std::size_t k_tile,
                       std::size_t pe) const;
};

PackedWeights pack_weights(const Eigen::Ref<const WeightMatrix>& weights,
                           const AcceleratorConfig& config);

}
