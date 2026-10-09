#pragma once

#include <string>

namespace accelerator {
struct Accelerator;
}

namespace simulator {

void write_profile(const accelerator::Accelerator& accel,
                   const std::string& path);

}
