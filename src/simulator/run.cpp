#include "simulator/run.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <systemc>
#include <utility>
#include <vector>

#include "accelerator/accelerator.hpp"
#include "accelerator/model_file.hpp"
#include "simulator/profile.hpp"

namespace simulator {
namespace {

SC_MODULE(CPU) {
    SC_CTOR(CPU, accelerator::CommandQueue & queue,
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

void check_output(const std::vector<float>& output,
                  accelerator::WeightMatrix expected,
                  const std::vector<accelerator::FCLayer>& layers) {
    // Обычный расчёт на CPU, без упаковки весов и разбиения на плитки.
    for (const auto& layer : layers) {
        accelerator::WeightMatrix next(expected.rows(), layer.weights.cols());
        for (Eigen::Index m = 0; m < next.rows(); ++m) {
            for (Eigen::Index n = 0; n < next.cols(); ++n) {
                double sum = layer.bias[n];
                for (Eigen::Index k = 0; k < expected.cols(); ++k) {
                    sum += double(expected(m, k)) * layer.weights(k, n);
                }
                next(m, n) = 1.0 / (1.0 + std::exp(-sum));
            }
        }
        expected = std::move(next);
    }
    if (output.size() != std::size_t(expected.size())) {
        throw std::runtime_error("Output size does not match CPU result");
    }
    double max_error = 0.0;
    for (std::size_t i = 0; i < output.size(); ++i) {
        if (!std::isfinite(output[i])) {
            throw std::runtime_error("Output contains NaN or infinity");
        }
        max_error = std::max(max_error,
                             std::abs(double(output[i]) - expected.data()[i]));
    }
    std::cout << "CPU comparison: max error=" << max_error << '\n';
    if (max_error > 1e-4) {
        throw std::runtime_error(
            "Output does not match CPU result (tolerance 1e-4)");
    }
}

void print_pe_stats(const accelerator::PEArray& array) {
    for (std::size_t i = 0; i < array.pes.size(); ++i) {
        const auto& pe = array.pes[i];
        std::cout << "PE[" << i << "] MAC ops=" << pe.mac_count
                  << ", utilization=" << 100.0 * pe.utilization() << '%'
                  << '\n';
    }
}

void print_transactions(const simulator::Transactions& transactions) {
    std::cout << "Model time=" << transactions.model_time()
              << ", transactions=" << transactions.total.count
              << ", transferred bytes=" << transactions.total.bytes << '\n';
}

int run_mlp(std::vector<int> sizes, int pe_count, int batch_size,
            const std::string& profile_path, const std::string& weights_path,
            const std::string& input_path) {
    std::vector<accelerator::FCLayer> layers;
    if (!weights_path.empty()) {
        layers = accelerator::load_weights(weights_path);
        sizes = {int(layers.front().weights.rows())};
        for (const auto& layer : layers) {
            sizes.push_back(int(layer.weights.cols()));
        }
    }
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
    accelerator::WeightMatrix input;
    if (input_path.empty()) {
        input.resize(batch_size, sizes.front());
        for (Eigen::Index i = 0; i < input.size(); ++i) {
            input.data()[i] = input_distribution(generator);
        }
    } else {
        input = accelerator::load_input(input_path, sizes.front());
        batch_size = input.rows();
    }
    if (weights_path.empty()) {
        for (std::size_t i = 1; i < sizes.size(); ++i) {
            const auto limit = std::sqrt(6.0f / (sizes[i - 1] + sizes[i]));
            std::uniform_real_distribution<float> distribution(-limit, limit);
            accelerator::WeightMatrix weights(sizes[i - 1], sizes[i]);
            for (Eigen::Index j = 0; j < weights.size(); ++j) {
                weights.data()[j] = distribution(generator);
            }
            layers.push_back(
                {std::move(weights), std::vector<float>(sizes[i], 0.0f)});
            for (int n = 0; n < sizes[i]; ++n) {
                layers.back().bias[n] = 0.1f * (n % 5 - 2);
            }
        }
    }
    accelerator::Accelerator accel("accelerator", config);
    const auto program = accel.load(input, layers);
    CPU cpu("cpu", accel.queue, program);
    sc_core::sc_start();
    const auto& last = program.commands.back();
    const auto output =
        accel.ram.read(last.y_addr, std::size_t(last.m) * last.n);
    if (accel.queue.completed != layers.size()) {
        throw std::runtime_error("Not all FC commands completed");
    }
    check_output(output, input, layers);
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
    print_pe_stats(accel.array);
    print_transactions(accel.transactions);
    if (!profile_path.empty()) simulator::write_profile(accel, profile_path);
    return 0;
}

}

int run(int argc, char* argv[]) {
    std::string profile_path, input_path;
    std::vector<std::string> args;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--profile" || arg == "--input") {
            if (++i == argc)
                throw std::invalid_argument("Missing path after " + arg);
            if (arg == "--profile") {
                profile_path = argv[i];
            } else {
                input_path = argv[i];
            }
        } else {
            args.push_back(arg);
        }
    }
    if (args.size() > 4) throw std::invalid_argument("Too many arguments");
    if (!args.empty() && args[0] == "--weights") {
        if (args.size() < 2)
            throw std::invalid_argument("Use --weights model.txt PE batch");
        return run_mlp({}, args.size() > 2 ? std::stoi(args[2]) : 4,
                       args.size() > 3 ? std::stoi(args[3]) : 8, profile_path,
                       args[1], input_path);
    }
    if (!args.empty() && args[0] == "--mlp") {
        if (args.size() < 2)
            throw std::invalid_argument("Use --mlp widths PE batch");
        std::vector<int> sizes;
        std::stringstream widths(args[1]);
        std::string value;
        while (std::getline(widths, value, ','))
            sizes.push_back(std::stoi(value));
        return run_mlp(sizes, args.size() > 2 ? std::stoi(args[2]) : 4,
                       args.size() > 3 ? std::stoi(args[3]) : 8, profile_path,
                       "", input_path);
    }
    const auto pe_count = !args.empty() ? std::stoi(args[0]) : 4;
    const auto k = args.size() > 1 ? std::stoi(args[1]) : 65;
    const auto n = args.size() > 2 ? std::stoi(args[2]) : 5;
    const auto batch = args.size() > 3 ? std::stoi(args[3]) : 9;
    return run_mlp({k, n}, pe_count, batch, profile_path, "", input_path);
}

}
