#include "simulator/transactions.hpp"

#include <sysc/kernel/sc_simcontext.h>
#include <sysc/kernel/sc_wait.h>

namespace simulator {

void Transactions::transfer(TransactionType type, std::size_t bytes) {
    auto& entry = by_type.at(static_cast<std::size_t>(type));
    sc_core::wait(transfer_time);
    ++entry.count;
    entry.bytes += bytes;
    ++total.count;
    total.bytes += bytes;
}

const TransactionStats& Transactions::stats(TransactionType type) const {
    return by_type.at(static_cast<std::size_t>(type));
}

double Transactions::model_time() const {
    return sc_core::sc_time_stamp() / transfer_time;
}

}
