#pragma once

#include <Eigen/Core>
#include <vector>

namespace mlp {

using Matrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::RowVectorXf;

// CPU layout: weights[K, N], bias[N]. The accelerator will pack these weights.
struct Layer {
    Matrix weights;
    Vector bias;
};

// X[M, K] -> sigmoid(X * weights + bias)[M, N].
// Inputs must be nonempty and have compatible dimensions.
Matrix dense(const Matrix& input, const Layer& layer);

// Run a nonempty sequence of FC layers without training state or gradients.
Matrix predict(const Matrix& input, const std::vector<Layer>& layers);

}  // namespace mlp
