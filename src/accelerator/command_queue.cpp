#include "accelerator/command_queue.hpp"

namespace accelerator {

void CommandQueue::submit(const FCCommand& command) {
    while (fifo.num_free() == 0) {
        sc_core::wait(fifo.data_read_event());
    }
    transactions.transfer(simulator::TransactionType::CpuToQueue,
                          FCCommand::descriptor_bytes);
    fifo.write(command);
}

void CommandQueue::run() {
    while (true) {
        const auto command = fifo.read();
        transactions.transfer(simulator::TransactionType::QueueToScheduler,
                              FCCommand::descriptor_bytes);
        scheduler.execute(command);
        ++completed;
    }
}

}
