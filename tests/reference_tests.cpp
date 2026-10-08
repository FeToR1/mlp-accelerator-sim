#include <cmath>
#include <iomanip>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "mlp/reference.hpp"

namespace {

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

template <typename Function>
void expect_invalid(Function function) {
    try {
        function();
    } catch (const std::invalid_argument&) {
        return;
    }
    throw std::runtime_error(
        "Expected invalid_argument for incompatible shapes");
}

// Independent scalar calculation to check layout, bias and layer composition.
mlp::Matrix scalar_dense(const mlp::Matrix& x, const mlp::Layer& layer) {
    mlp::Matrix result(x.rows(), layer.weights.cols());
    for (Eigen::Index m = 0; m < x.rows(); ++m) {
        for (Eigen::Index n = 0; n < layer.weights.cols(); ++n) {
            double sum = layer.bias[n];
            for (Eigen::Index k = 0; k < x.cols(); ++k) {
                sum += double(x(m, k)) * layer.weights(k, n);
            }
            result(m, n) = static_cast<float>(1.0 / (1.0 + std::exp(-sum)));
        }
    }
    return result;
}

void test_numerics() {
    mlp::Matrix x(2, 3);
    x << 1.0f, -2.0f, 0.5f, 0.25f, 3.0f, -1.0f;
    mlp::Layer first{mlp::Matrix(3, 2), mlp::Vector(2)};
    first.weights << 0.5f, -0.25f, 1.0f, 0.75f, -2.0f, 0.125f;
    first.bias << 0.3f, -0.7f;
    mlp::Layer second{mlp::Matrix(2, 3), mlp::Vector(3)};
    second.weights << 0.4f, -0.2f, 0.8f, -0.5f, 1.2f, 0.1f;
    second.bias << 0.1f, -0.3f, 0.2f;
    const mlp::Matrix input_copy = x;
    const mlp::Matrix weights_copy = first.weights;
    const mlp::Vector bias_copy = first.bias;
    const auto single = mlp::dense(x, first);
    require((single - scalar_dense(x, first)).cwiseAbs().maxCoeff() < 1e-6f,
            "FC disagrees with scalar dot products");
    const auto result = mlp::predict(x, {first, second});
    const auto expected = scalar_dense(scalar_dense(x, first), second);
    require(result.rows() == 2 && result.cols() == 3,
            "Wrong network output shape");
    require((result - expected).cwiseAbs().maxCoeff() < 1e-6f,
            "Network disagrees with scalar reference");
    require((x.array() == input_copy.array()).all() &&
                (first.weights.array() == weights_copy.array()).all() &&
                (first.bias.array() == bias_copy.array()).all(),
            "Inference mutated input or layer parameters");

    mlp::Matrix extremes(5, 1);
    extremes << -1000.0f, 0.0f, 1000.0f,
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity();
    mlp::Layer identity{mlp::Matrix::Ones(1, 1), mlp::Vector::Zero(1)};
    const auto saturated = mlp::dense(extremes, identity);
    require(saturated(0, 0) == 0.0f && saturated(1, 0) == 0.5f &&
                saturated(2, 0) == 1.0f && saturated(3, 0) == 0.0f &&
                saturated(4, 0) == 1.0f,
            "Sigmoid failed for large values or infinities");
    mlp::Matrix nan_input(1, 1);
    nan_input(0, 0) = std::numeric_limits<float>::quiet_NaN();
    require(std::isnan(mlp::dense(nan_input, identity)(0, 0)),
            "Sigmoid must propagate NaN");

    expect_invalid([&] { mlp::predict(x, {}); });
    expect_invalid([&] { mlp::dense(mlp::Matrix(0, 3), first); });
    expect_invalid([&] { mlp::dense(mlp::Matrix(2, 0), first); });
    expect_invalid([&] { mlp::dense(mlp::Matrix::Zero(2, 4), first); });
    expect_invalid(
        [&] { mlp::dense(x, {first.weights, mlp::Vector::Zero(1)}); });
    expect_invalid([&] { mlp::dense(x, {mlp::Matrix(3, 0), mlp::Vector(0)}); });
    expect_invalid([&] { mlp::predict(x, {first, first}); });
}

void write_matrix(const mlp::Matrix& matrix) {
    std::cout << '[';
    for (Eigen::Index m = 0; m < matrix.rows(); ++m) {
        if (m) std::cout << ',';
        std::cout << '[';
        for (Eigen::Index n = 0; n < matrix.cols(); ++n) {
            if (n) std::cout << ',';
            std::cout << matrix(m, n);
        }
        std::cout << ']';
    }
    std::cout << ']';
}

// Match the old constructor only in the compatibility fixture. The new API
// accepts explicit parameters and has no random initialization of its own.
void write_legacy_fixtures() {
    const std::vector<std::vector<int>> topologies = {
        {3, 4, 2}, {7, 5, 3, 2}, {784, 128, 10}};
    std::cout << std::setprecision(std::numeric_limits<float>::max_digits10);
    std::cout << '[';
    for (std::size_t fixture = 0; fixture < topologies.size(); ++fixture) {
        if (fixture) std::cout << ',';
        const auto& sizes = topologies[fixture];
        std::mt19937 generator(42);
        std::vector<mlp::Layer> layers;
        for (std::size_t i = 1; i < sizes.size(); ++i) {
            const float limit =
                std::sqrt(6.0 / (double(sizes[i - 1]) + sizes[i]));
            std::uniform_real_distribution<float> distribution(-limit, limit);
            mlp::Layer layer{mlp::Matrix(sizes[i - 1], sizes[i]),
                             mlp::Vector::Zero(sizes[i])};
            for (Eigen::Index j = 0; j < layer.weights.size(); ++j) {
                layer.weights.data()[j] = distribution(generator);
            }
            layers.push_back(std::move(layer));
        }
        mlp::Matrix x(8, sizes.front());
        for (Eigen::Index j = 0; j < x.size(); ++j) {
            x.data()[j] = static_cast<float>(int(j % 17) - 8) / 8.0f;
        }
        std::cout << "{\"sizes\":[";
        for (std::size_t i = 0; i < sizes.size(); ++i) {
            if (i) std::cout << ',';
            std::cout << sizes[i];
        }
        std::cout << "],\"input\":";
        write_matrix(x);
        std::cout << ",\"output\":";
        write_matrix(mlp::predict(x, layers));
        std::cout << '}';
    }
    std::cout << "]\n";
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc == 2 && std::string(argv[1]) == "--legacy-fixtures") {
            write_legacy_fixtures();
        } else {
            test_numerics();
            std::cout << "Reference FC, network, sigmoid and shape checks OK\n";
        }
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
