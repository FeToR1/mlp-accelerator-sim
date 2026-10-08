#include "mlp/reference.hpp"

#include <Eigen/Dense>
#include <cmath>
#include <stdexcept>

namespace mlp {

Matrix dense(const Matrix& input, const Layer& layer) {
    if (input.rows() <= 0 || input.cols() <= 0 || layer.weights.rows() <= 0 ||
        layer.weights.cols() <= 0) {
        throw std::invalid_argument("Input and weights must be nonempty");
    }
    if (input.cols() != layer.weights.rows()) {
        throw std::invalid_argument("Input width does not match the weights");
    }
    if (layer.bias.size() != layer.weights.cols()) {
        throw std::invalid_argument(
            "Bias size does not match the output width");
    }

    Matrix output(input.rows(), layer.weights.cols());
    output.noalias() = input * layer.weights;
    output.rowwise() += layer.bias;
    // Same numerically stable sigmoid as the original Eigen implementation.
    output = output.unaryExpr([](float x) {
        const float e = std::exp(-std::abs(x));
        return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    });
    return output;
}

Matrix predict(const Matrix& input, const std::vector<Layer>& layers) {
    if (layers.empty()) {
        throw std::invalid_argument("At least one FC layer is required");
    }
    Matrix current = input;
    for (const auto& layer : layers) {
        current = dense(current, layer);
    }
    return current;
}

}  // namespace mlp
