#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <systemc>
#include <vector>

#include "accelerator/accelerator.hpp"
#include "simulator/profile.hpp"

SC_MODULE(PEArrayDemo) {
    SC_CTOR(PEArrayDemo, accelerator::CommandQueue & queue,
            const accelerator::FCCommand& command)
        : queue(queue), command(command) {
        SC_THREAD(run);
    }

   private:
    accelerator::CommandQueue& queue;
    const accelerator::FCCommand& command;

    void run() { queue.submit(command); }
};

SC_MODULE(MLPDemo) {
    SC_CTOR(MLPDemo, accelerator::CommandQueue & queue,
            const accelerator::MLPProgram& program)
        : queue(queue), program(program) {
        SC_THREAD(run);
    }

   private:
    accelerator::CommandQueue& queue;
    const accelerator::MLPProgram& program;
    void run() {
        for (const auto& command : program.commands) {
            queue.submit(command);
        }
    }
};

int run_mlp(const std::vector<int>& sizes, int pe_count, int batch_size,
            const std::string& profile_path) {
    if (sizes.size() < 2 || batch_size <= 0) {
        throw std::invalid_argument(
            "MLP needs at least two widths and positive batch");
    }
    for (int size : sizes) {
        if (size <= 0)
            throw std::invalid_argument("MLP widths must be positive");
    }
    accelerator::AcceleratorConfig config;
    config.pe_count = pe_count;
    config.validate();
    std::mt19937 generator(42);
    std::uniform_real_distribution<float> input_distribution(-1.0f, 1.0f);
    accelerator::WeightMatrix input(batch_size, sizes.front());
    for (Eigen::Index i = 0; i < input.size(); ++i) {
        input.data()[i] = input_distribution(generator);
    }
    std::vector<accelerator::FCLayer> layers;
    for (std::size_t i = 1; i < sizes.size(); ++i) {
        const auto limit = std::sqrt(6.0f / (sizes[i - 1] + sizes[i]));
        std::uniform_real_distribution<float> distribution(-limit, limit);
        accelerator::WeightMatrix weights(sizes[i - 1], sizes[i]);
        for (Eigen::Index j = 0; j < weights.size(); ++j) {
            weights.data()[j] = distribution(generator);
        }
        layers.push_back(
            {std::move(weights), std::vector<float>(sizes[i], 0.0f)});
    }
    accelerator::Accelerator accel("accelerator", config);
    const auto program = accel.load(input, layers);
    MLPDemo demo("demo", accel.queue, program);
    sc_core::sc_start();
    const auto& last = program.commands.back();
    const auto output =
        accel.ram.read(last.y_addr, std::size_t(last.m) * last.n);
    std::cout << "Completed commands=" << accel.queue.completed
              << ", PE=" << pe_count << ", batch=" << batch_size
              << ", RAM bytes=" << accel.ram.size_bytes() << '\n';
    for (std::size_t m = 0; m < last.m; ++m) {
        std::cout << "MLP Y[" << m << "]:";
        for (std::size_t n = 0; n < last.n; ++n) {
            std::cout << ' ' << output[m * last.n + n];
        }
        std::cout << '\n';
    }
    for (std::size_t pe = 0; pe < accel.array.pes.size(); ++pe) {
        std::cout << "PE[" << pe
                  << "] MAC ops=" << accel.array.pes[pe].mac_count
                  << ", utilization="
                  << 100.0 * accel.array.pes[pe].utilization() << '%' << '\n';
    }
    std::cout << "Model time=" << accel.transactions.model_time()
              << ", transactions=" << accel.transactions.total.count
              << ", transferred bytes=" << accel.transactions.total.bytes
              << '\n';
    if (!profile_path.empty()) simulator::write_profile(accel, profile_path);
    return 0;
}

int sc_main(int argc, char* argv[]) {
    std::string profile_path;
    if (argc > 5) {
        if (argc != 7 || std::string(argv[5]) != "--profile")
            throw std::invalid_argument(
                "Append --profile path.json after PE/dimensions");
        profile_path = argv[6];
    }
    if (argc > 1 && std::string(argv[1]) == "--mlp") {
        if (argc < 3) throw std::invalid_argument("Use --mlp widths PE batch");
        std::vector<int> sizes;
        std::stringstream widths(argv[2]);
        std::string value;
        while (std::getline(widths, value, ','))
            sizes.push_back(std::stoi(value));
        return run_mlp(sizes, argc > 3 ? std::stoi(argv[3]) : 4,
                       argc > 4 ? std::stoi(argv[4]) : 8, profile_path);
    }
    accelerator::AcceleratorConfig config;
    if (argc > 1) {
        config.pe_count = std::stoi(argv[1]);
    }
    config.validate();

    accelerator::FCCommand command;
    const auto k_size = argc > 2 ? std::stoi(argv[2]) : config.k_tile_size + 1;
    const auto n_size =
        argc > 3 ? std::stoi(argv[3]) : config.output_tile_size + 1;
    const auto m_size =
        argc > 4 ? std::stoi(argv[4]) : config.microbatch_size + 1;
    if (k_size <= 0 || n_size <= 0 || m_size <= 0) {
        throw std::invalid_argument("M, N and K must be positive");
    }
    command.k = k_size;
    command.n = n_size;
    command.m = m_size;

    accelerator::WeightMatrix weights(command.k, command.n);
    const std::vector<float> regular_weights{1.0f, 2.0f, -1.0f, 0.0f};
    const std::vector<float> last_weights{4.0f, 1.0f, -1.0f, 1.0f};
    for (std::size_t n = 0; n < command.n; ++n) {
        weights.col(n).setConstant(regular_weights[n % regular_weights.size()]);
        weights(command.k - 1, n) = last_weights[n % last_weights.size()];
    }
    const auto packed = accelerator::pack_weights(weights, config);
    std::cout << "Packed weights: N tiles=" << packed.n_tiles
              << ", K tiles=" << packed.k_tiles
              << ", local K=" << packed.local_k_size
              << ", bytes=" << packed.values.size() * sizeof(float) << '\n';
    for (std::size_t pe = 0; pe < packed.pe_count; ++pe) {
        const auto base = packed.offset(0, 0, pe);
        std::cout << "Wpacked[0][0][" << pe << "][0]:";
        for (std::size_t lane = 0; lane < packed.lane_count; ++lane) {
            std::cout << ' ' << packed.values[base + lane];
        }
        std::cout << '\n';
    }

    accelerator::Accelerator accel("accelerator", config);
    auto& ram = accel.ram;
    const auto input_count = std::size_t(command.m) * command.k;
    std::vector<float> input(input_count);
    for (std::size_t m = 0; m < command.m; ++m) {
        for (std::size_t k = 0; k < command.k; ++k) {
            input[m * command.k + k] = float(m + 1);
        }
    }
    std::vector<float> bias_values(command.n);
    for (std::size_t n = 0; n < command.n; ++n) {
        bias_values[n] = float(n + 1);
    }
    command.x_addr = ram.allocate(input_count);
    command.w_addr = ram.allocate(packed.values.size());
    command.bias_addr = ram.allocate(bias_values.size());
    command.y_addr = ram.allocate(std::size_t(command.m) * command.n);
    ram.write(command.x_addr, input);
    ram.write(command.w_addr, packed.values);
    ram.write(command.bias_addr, bias_values);

    auto& transactions = accel.transactions;
    auto& array = accel.array;
    auto& bias = accel.bias;
    auto& lut = accel.lut;
    auto& output_sram = accel.output_sram;
    auto& queue = accel.queue;
    PEArrayDemo demo("demo", queue, command);

    std::cout << "PE=" << config.pe_count
              << ", microbatch=" << config.microbatch_size
              << ", output_tile=" << config.output_tile_size
              << ", K_tile=" << config.k_tile_size << '\n';
    std::cout << "MAC lanes per PE=" << config.output_tile_size
              << ", ACC bytes per PE="
              << array.pes[0].acc.size() * sizeof(float) << '\n';
    std::cout << "Bias buffer bytes=" << bias.size_bytes() << ", bias:";
    for (const auto value : bias_values) {
        std::cout << ' ' << value;
    }
    std::cout << '\n';
    std::cout << "Sigmoid LUT: entries=" << accelerator::SigmoidLUT::table_size
              << ", bytes=" << lut.size_bytes() << '\n';
    for (const float value : {-16.0f, -1.1f, 0.0f, 1.1f, 16.0f}) {
        std::cout << "LUT(" << value << ")=" << lut.lookup(value) << '\n';
    }
    std::cout << "Output SRAM bytes=" << output_sram.size_bytes() << '\n';
    std::cout << "FC: M=" << command.m << ", N=" << command.n
              << ", K=" << command.k << '\n';
    std::cout << "X address=" << command.x_addr
              << ", W address=" << command.w_addr
              << ", bias address=" << command.bias_addr
              << ", Y address=" << command.y_addr
              << ", RAM bytes=" << ram.size_bytes() << '\n';

    sc_core::sc_start();

    std::cout << "Completed commands=" << queue.completed
              << ", FIFO bytes=" << queue.size_bytes() << '\n';
    for (std::size_t pe_id = 0; pe_id < array.pes.size(); ++pe_id) {
        std::cout << "PE[" << pe_id
                  << "] MAC ops=" << array.pes[pe_id].mac_count
                  << ", utilization=" << 100.0 * array.pes[pe_id].utilization()
                  << '%' << '\n';
    }
    const auto result =
        ram.read(command.y_addr, std::size_t(command.m) * command.n);
    for (std::size_t m = 0; m < command.m; ++m) {
        std::cout << "Y RAM[" << m << "]:";
        for (std::size_t n = 0; n < command.n; ++n) {
            std::cout << ' ' << result[m * command.n + n];
        }
        std::cout << '\n';
    }

    std::cout << "SystemC time=" << sc_core::sc_time_stamp() << '\n';
    std::cout << "Model time=" << transactions.model_time()
              << ", transactions=" << transactions.total.count
              << ", transferred bytes=" << transactions.total.bytes << '\n';

    const auto& ram_dma =
        transactions.stats(simulator::TransactionType::RamToDma);
    const auto& dma_sram =
        transactions.stats(simulator::TransactionType::DmaToSram);
    const auto& sram_dma =
        transactions.stats(simulator::TransactionType::SramToDma);
    const auto& dma_ram =
        transactions.stats(simulator::TransactionType::DmaToRam);
    const auto& input_pe =
        transactions.stats(simulator::TransactionType::InputSramToPe);
    const auto& weight_pe =
        transactions.stats(simulator::TransactionType::WeightSramToPe);
    const auto& pe_reduction =
        transactions.stats(simulator::TransactionType::PeToReduction);
    const auto& dma_bias =
        transactions.stats(simulator::TransactionType::DmaToBias);
    const auto& reduction_bias =
        transactions.stats(simulator::TransactionType::ReductionToBias);
    const auto& bias_lut =
        transactions.stats(simulator::TransactionType::BiasToLut);
    const auto& lut_sram =
        transactions.stats(simulator::TransactionType::LutToOutputSram);
    const auto& cpu_queue =
        transactions.stats(simulator::TransactionType::CpuToQueue);
    const auto& queue_scheduler =
        transactions.stats(simulator::TransactionType::QueueToScheduler);
    std::cout << "RAM -> DMA: count=" << ram_dma.count
              << ", bytes=" << ram_dma.bytes << '\n';
    std::cout << "DMA -> SRAM: count=" << dma_sram.count
              << ", bytes=" << dma_sram.bytes << '\n';
    std::cout << "SRAM -> DMA: count=" << sram_dma.count
              << ", bytes=" << sram_dma.bytes << '\n';
    std::cout << "DMA -> RAM: count=" << dma_ram.count
              << ", bytes=" << dma_ram.bytes << '\n';
    std::cout << "Input SRAM -> PE: count=" << input_pe.count
              << ", bytes=" << input_pe.bytes << '\n';
    std::cout << "Weight SRAM -> PE: count=" << weight_pe.count
              << ", bytes=" << weight_pe.bytes << '\n';
    std::cout << "PE -> reduction: count=" << pe_reduction.count
              << ", bytes=" << pe_reduction.bytes << '\n';
    std::cout << "DMA -> Bias: count=" << dma_bias.count
              << ", bytes=" << dma_bias.bytes << '\n';
    std::cout << "Reduction -> Bias: count=" << reduction_bias.count
              << ", bytes=" << reduction_bias.bytes << '\n';
    std::cout << "Bias -> LUT: count=" << bias_lut.count
              << ", bytes=" << bias_lut.bytes << '\n';
    std::cout << "LUT -> Output SRAM: count=" << lut_sram.count
              << ", bytes=" << lut_sram.bytes << '\n';
    std::cout << "CPU -> FIFO: count=" << cpu_queue.count
              << ", bytes=" << cpu_queue.bytes << '\n';
    std::cout << "FIFO -> Scheduler: count=" << queue_scheduler.count
              << ", bytes=" << queue_scheduler.bytes << '\n';
    if (!profile_path.empty()) simulator::write_profile(accel, profile_path);
    return 0;
}
