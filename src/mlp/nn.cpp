#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <Eigen/Dense>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <utility>
#include <vector>

namespace py = pybind11;
using Matrix =
    Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using Vector = Eigen::RowVectorXf;
using Array = py::array_t<float, py::array::c_style>;

Eigen::Map<const Matrix> matrix_view(const Array& array) {
    if (array.ndim() != 2 || array.shape(0) <= 0 || array.shape(1) <= 0) {
        throw std::invalid_argument(
            "Expected a nonempty 2D C-contiguous float32 array");
    }
    return {array.data(), array.shape(0), array.shape(1)};
}

Array numpy_copy(const Matrix& matrix) {
    Array result({static_cast<py::ssize_t>(matrix.rows()),
                  static_cast<py::ssize_t>(matrix.cols())});
    std::memcpy(result.mutable_data(), matrix.data(),
                matrix.size() * sizeof(float));
    return result;
}

void sigmoid(Matrix& matrix) {
    matrix = matrix.unaryExpr([](float x) {
        const float e = std::exp(-std::abs(x));
        return x >= 0.0f ? 1.0f / (1.0f + e) : e / (1.0f + e);
    });
}

struct Layer {
    Matrix weights, weight_gradient;
    Vector bias, bias_gradient;

    Matrix apply(const Matrix& input) const {
        Matrix output(input.rows(), weights.cols());
        output.noalias() = input * weights;
        output.rowwise() += bias;
        sigmoid(output);
        return output;
    }
};

struct Parameter {
    Eigen::Map<Eigen::VectorXf> value, gradient;
};

struct Parameters {
    std::vector<Parameter> entries;
};

class MLP {
    std::vector<Layer> layers;
    std::vector<Matrix> activations;

   public:
    MLP(const std::vector<int>& sizes, std::uint32_t seed = 42) {
        if (sizes.size() < 2) {
            throw std::invalid_argument(
                "At least two layer sizes are required");
        }
        for (int size : sizes)
            if (size <= 0) {
                throw std::invalid_argument("Layer sizes must be positive");
            }
        std::mt19937 generator(seed);
        for (std::size_t i = 1; i < sizes.size(); ++i) {
            const float limit =
                std::sqrt(6.0 / (double(sizes[i - 1]) + sizes[i]));
            std::uniform_real_distribution<float> distribution(-limit, limit);
            Matrix weight(sizes[i - 1], sizes[i]);
            for (Eigen::Index j = 0; j < weight.size(); ++j) {
                weight.data()[j] = distribution(generator);
            }
            layers.push_back({std::move(weight),
                              Matrix::Zero(sizes[i - 1], sizes[i]),
                              Vector::Zero(sizes[i]), Vector::Zero(sizes[i])});
        }
    }

    Parameters parameters() {
        Parameters result;
        result.entries.reserve(2 * layers.size());
        for (auto& layer : layers) {
            result.entries.push_back(
                {{layer.weights.data(), layer.weights.size()},
                 {layer.weight_gradient.data(), layer.weight_gradient.size()}});
            result.entries.push_back(
                {{layer.bias.data(), layer.bias.size()},
                 {layer.bias_gradient.data(), layer.bias_gradient.size()}});
        }
        return result;
    }

    Array forward(const Array& input) {
        const auto x = matrix_view(input);
        if (x.cols() != layers.front().weights.rows()) {
            throw std::invalid_argument("Input width does not match the model");
        }
        activations.clear();
        activations.reserve(layers.size() + 1);
        activations.emplace_back(x);
        for (const auto& layer : layers) {
            activations.push_back(layer.apply(activations.back()));
        }
        return numpy_copy(activations.back());
    }

    void backward(const Array& gradient) {
        const auto g = matrix_view(gradient);
        if (activations.size() != layers.size() + 1) {
            throw std::logic_error("Call forward before backward");
        }
        if (g.rows() != activations.back().rows() ||
            g.cols() != layers.back().weights.cols()) {
            throw std::invalid_argument(
                "Gradient shape does not match the output");
        }
        Matrix delta = g;
        for (std::size_t i = layers.size(); i-- > 0;) {
            auto& layer = layers[i];
            const Matrix& output = activations[i + 1];
            delta.array() *= output.array() * (1.0f - output.array());
            layer.weight_gradient.noalias() +=
                activations[i].transpose() * delta;
            layer.bias_gradient += delta.colwise().sum();
            if (i > 0) {
                Matrix previous(delta.rows(), layer.weights.rows());
                previous.noalias() = delta * layer.weights.transpose();
                delta.swap(previous);
            }
        }
        activations.clear();
    }

    Array predict(const Array& input) const {
        const auto x = matrix_view(input);
        if (x.cols() != layers.front().weights.rows()) {
            throw std::invalid_argument("Input width does not match the model");
        }
        Matrix current = x;
        for (const auto& layer : layers) {
            current = layer.apply(current);
        }
        return numpy_copy(current);
    }
};

class SGD {
    Parameters parameters;
    float lr;

   public:
    SGD(Parameters parameters, float lr = 0.1f)
        : parameters(std::move(parameters)), lr(lr) {
        if (!std::isfinite(lr) || lr <= 0.0f) {
            throw std::invalid_argument(
                "Learning rate must be finite and positive");
        }
    }

    void zero_grad() {
        for (auto& parameter : parameters.entries) {
            parameter.gradient.setZero();
        }
    }

    void step() {
        for (auto& parameter : parameters.entries) {
            parameter.value -= lr * parameter.gradient;
        }
    }
};

PYBIND11_MODULE(nn, module) {
    py::class_<Parameters>(module, "Parameters");
    py::class_<MLP>(module, "MLP")
        .def(py::init<const std::vector<int>&, std::uint32_t>(),
             py::arg("sizes"), py::arg("seed") = 42)
        .def("parameters", &MLP::parameters, py::keep_alive<0, 1>())
        .def("forward", &MLP::forward, py::arg("x").noconvert())
        .def("backward", &MLP::backward, py::arg("gradient").noconvert())
        .def("predict", &MLP::predict, py::arg("x").noconvert());
    py::class_<SGD>(module, "SGD")
        .def(py::init<Parameters, float>(), py::arg("parameters"),
             py::arg("lr") = 0.1f, py::keep_alive<1, 2>())
        .def("zero_grad", &SGD::zero_grad)
        .def("step", &SGD::step);
}
