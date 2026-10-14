#pragma once

#include <string>
#include <vector>

#include "accelerator/program.hpp"

namespace accelerator {

std::vector<FCLayer> load_weights(const std::string& path);
WeightMatrix load_input(const std::string& path, int width);

}
