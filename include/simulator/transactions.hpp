#pragma once

#include <sysc/kernel/sc_time.h>

#include <array>
#include <cstddef>
#include <cstdint>

namespace simulator {

enum class TransactionType {
    RamToDma,
    DmaToSram,
    SramToDma,
    DmaToRam,
    InputSramToPe,
    WeightSramToPe,
    PeToReduction,
    Count
};

struct TransactionStats {
    std::uint64_t count = 0;
    std::uint64_t bytes = 0;
};

class Transactions {
   public:
    void transfer(TransactionType type, std::size_t bytes);
    const TransactionStats& stats(TransactionType type) const;
    double model_time() const;

    TransactionStats total;

   private:
    const sc_core::sc_time transfer_time{1, sc_core::SC_NS};
    std::array<TransactionStats,
               static_cast<std::size_t>(TransactionType::Count)>
        by_type{};
};

}
