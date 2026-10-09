#pragma once

#include <vector>

#include "accelerator/command.hpp"
#include "accelerator/packed_weights.hpp"
#include "accelerator/ram.hpp"

namespace accelerator {

struct FCLayer {
    WeightMatrix weights;
    std::vector<float> bias;
};

struct MLPProgram {
    std::vector<FCCommand> commands;
};

MLPProgram load_mlp(ExternalRAM& ram, const WeightMatrix& input,
                    const std::vector<FCLayer>& layers,
                    const AcceleratorConfig& config);

}
