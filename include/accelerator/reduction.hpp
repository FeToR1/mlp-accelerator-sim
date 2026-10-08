#pragma once

#include <sysc/utils/sc_vector.h>

#include <cstddef>
#include <vector>

#include "accelerator/pe.hpp"
#include "simulator/transactions.hpp"

namespace accelerator {

SC_MODULE(Reduction) {
    SC_CTOR(Reduction, simulator::Transactions& transactions)
        : transactions(transactions) {}

    std::vector<float> reduce(const sc_core::sc_vector<PE>& pes,
                              std::size_t count);

   private:
    simulator::Transactions& transactions;
};

}
