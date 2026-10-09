#pragma once

#include <systemc>

#include "accelerator/command.hpp"
#include "accelerator/scheduler.hpp"
#include "simulator/transactions.hpp"

namespace accelerator {

SC_MODULE(CommandQueue) {
    static constexpr int capacity = 16;
    SC_CTOR(CommandQueue, Scheduler & scheduler,
            simulator::Transactions & transactions)
        : fifo("fifo", capacity),
          scheduler(scheduler),
          transactions(transactions) {
        SC_THREAD(run);
    }

    void submit(const FCCommand& command);
    std::size_t size_bytes() const {
        return capacity * FCCommand::descriptor_bytes;
    }
    std::uint64_t completed = 0;

   private:
    sc_core::sc_fifo<FCCommand> fifo;
    Scheduler & scheduler;
    simulator::Transactions& transactions;
    void run();
};

}
