#pragma once
#include <algorithm>
#include <atomic>
#include <cassert>
#include <cmath>
#include <concepts>
#include <future>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>
//
#include <Eigen/Eigen>

template <std::floating_point real>
constexpr real sigmoid(real x) {
  return 1 / (1 + std::exp(-x));
}

template <std::floating_point real>
constexpr real d_sigmoid(real x) {
  const auto tmp = sigmoid(x);
  return tmp * (1 - tmp);
}

struct neural_network {
  using real = float;
  using vector = Eigen::Matrix<real, Eigen::Dynamic, 1>;
  using matrix = Eigen::Matrix<real, Eigen::Dynamic, Eigen::Dynamic>;

  neural_network(std::initializer_list<size_t> list) : sizes{list} {
    if (size(list) < 2)
      throw std::runtime_error("Neural network needs at least two layers!");

    weights.resize(size(sizes) - 1);
    biases.resize(size(sizes) - 1);

    std::uniform_real_distribution<real> distribution{-1, 1};
    for (size_t i = 0; i < sizes.size() - 1; ++i) {
      weights[i].resize(sizes[i + 1], sizes[i]);
      for (size_t j = 0; j < weights[i].size(); ++j)
        weights[i].data()[j] = distribution(rng);
      biases[i].resize(sizes[i + 1]);
      for (size_t j = 0; j < biases[i].size(); ++j)
        biases[i].data()[j] = distribution(rng);
    }

    layer_inputs.resize(size(sizes) - 1);
    layer_outputs.resize(size(sizes) - 1);
    weight_gradients.resize(size(sizes) - 1);
    bias_gradients.resize(size(sizes) - 1);
    buffers.resize(size(sizes) - 1);
  }

  auto simple_forward_feed(const vector& input) {
    assert(input.size() == sizes[0]);
    const auto activation = sigmoid<real>;
    vector result = (weights[0] * input + biases[0]).unaryExpr(activation);
    for (size_t i = 1; i < size(sizes) - 1; ++i)
      result = (weights[i] * result + biases[i]).unaryExpr(activation);
    return result;
  }

  auto forward_feed(const vector& input) {
    assert(input.size() == sizes[0]);
    const auto activation = sigmoid<real>;
    layer_inputs[0] = weights[0] * input + biases[0];
    layer_outputs[0] = layer_inputs[0].unaryExpr(activation);
    for (size_t i = 1; i < size(sizes) - 1; ++i) {
      layer_inputs[i] = weights[i] * layer_outputs[i - 1] + biases[i];
      layer_outputs[i] = layer_inputs[i].unaryExpr(activation);
    }
  }

  auto squared_error(const vector& input, const vector& label) {
    assert(input.size() == sizes.front());
    assert(label.size() == sizes.back());

    forward_feed(input);
    return (layer_outputs.back() - label).squaredNorm();
  }

  auto mean_squared_error(const std::vector<vector>& inputs,
                          const std::vector<vector>& labels) {
    assert(inputs.size() == labels.size());

    real result{};
    for (size_t i = 0; i < inputs.size(); ++i)
      result += squared_error(inputs[i], labels[i]);
    return result / inputs.size();
  }

  auto classification_rate(const std::vector<vector>& inputs,
                           const std::vector<vector>& labels) {
    assert(inputs.size() == labels.size());

    std::atomic<size_t> result{};
    // constexpr size_t thread_count = 8;
    // const size_t thread_elements = inputs.size() / thread_count;

    // std::future<void> tasks[thread_count];

    // for (size_t t = 0; t < thread_count; ++t) {
    //   tasks[t] = std::async(std::launch::async, [&]() {
    // for (size_t i = t * thread_elements; i < ((t + 1) * thread_elements);
    // ++i) {
    for (size_t i = 0; i < inputs.size(); ++i) {
      // forward_feed(inputs[i]);
      vector output = simple_forward_feed(inputs[i]);
      size_t label_maxarg = 0;
      size_t output_maxarg = 0;
      for (size_t j = 1; j < labels[i].size(); ++j) {
        label_maxarg =
            (labels[i][label_maxarg] < labels[i][j]) ? (j) : (label_maxarg);
        output_maxarg =
            (output[output_maxarg] < output[j]) ? (j) : (output_maxarg);
      }
      result += (label_maxarg == output_maxarg);
    }
    //   });
    // }
    // for (size_t t = 0; t < thread_count; ++t) tasks[t].wait();
    return static_cast<real>(result) / inputs.size();
  }

  // auto simple_backprop(const vector& input, const vector& label) {
  //   std::vector<vector> inputs(size(sizes) - 1);
  //   std::vector<vector> outputs(size(sizes) - 1);

  //   const auto activation = sigmoid<real>;
  //   const auto d_activation = d_sigmoid<real>;

  //   inputs[0] = weights[0] * input + biases[0];
  //   outputs[0] = inputs[0].unaryExpr(activation);
  //   for (size_t i = 1; i < size(sizes) - 1; ++i) {
  //     inputs[i] = weights[i] * outputs[i - 1] + biases[i];
  //     outputs[i] = inputs[i].unaryExpr(activation);
  //   }

  //   vector buffer = (label - outputs.back()).array() *
  //                   inputs.back().array().unaryExpr(d_activation);

  //   for (auto i = size(sizes) - 2; i > 0; --i) {
  //     buffers = (weights[i].transpose() * buffers).array() *
  //               inputs[i - 1].array().unaryExpr(d_activation);
  //     while (wait)
  //       ;
  //     wait = true;
  //     // negative gradient
  //     weight_gradients[i] += buffers[i] * outputs[i - 1].transpose();
  //     bias_gradients[i] += buffers[i];
  //     wait = false;
  //   }
  //   while (wait)
  //     ;
  //   wait = true;
  //   // negative gradient
  //   weight_gradients[0] += buffers[0] * input.transpose();
  //   bias_gradients[0] += buffers[0];
  //   wait = false;
  // }

  void backprop(const vector& input, const vector& label) {
    const auto activation = sigmoid<real>;
    const auto d_activation = d_sigmoid<real>;

    buffers.back() = (label - layer_outputs.back()).array() *
                     layer_inputs.back().array().unaryExpr(d_activation);

    for (auto i = size(sizes) - 2; i > 0; --i) {
      buffers[i - 1] = (weights[i].transpose() * buffers[i]).array() *
                       layer_inputs[i - 1].array().unaryExpr(d_activation);
      // negative gradient
      weight_gradients[i] += buffers[i] * layer_outputs[i - 1].transpose();
      bias_gradients[i] += buffers[i];
    }
    // negative gradient
    weight_gradients[0] += buffers[0] * input.transpose();
    bias_gradients[0] += buffers[0];
  }

  void train(const std::vector<vector>& inputs,
             const std::vector<vector>& labels, size_t epochs,
             size_t batch_size, real learning_rate) {
    assert(inputs.size() == labels.size());
    std::vector<size_t> indices(inputs.size());
    std::iota(begin(indices), end(indices), 0);
    for (size_t e = 0; e < epochs; ++e) {
      std::shuffle(begin(indices), end(indices), rng);

      constexpr size_t thread_count = 8;

      for (size_t i = 0; i < indices.size(); i += batch_size) {
        for (auto j = 0; j < size(sizes) - 1; ++j) {
          weight_gradients[j] =
              matrix::Zero(weights[j].rows(), weights[j].cols());
          bias_gradients[j] = vector::Zero(biases[j].size());
        }
        for (size_t k = 0; k < batch_size; ++k) {
          forward_feed(inputs[indices[i + k]]);
          backprop(inputs[indices[i + k]], labels[indices[i + k]]);
        }
        for (auto j = 0; j < size(sizes) - 1; ++j) {
          weights[j] += learning_rate * weight_gradients[j] / batch_size;
          biases[j] += learning_rate * bias_gradients[j] / batch_size;
        }
      }
    }
  }

  std::vector<size_t> sizes{};
  std::vector<matrix> weights{};
  std::vector<vector> biases{};

  std::vector<vector> layer_inputs{};
  std::vector<vector> layer_outputs{};
  std::vector<matrix> weight_gradients{};
  std::vector<vector> bias_gradients{};
  std::vector<vector> buffers{};

  std::atomic<bool> wait = false;

  std::mt19937 rng{std::random_device{}()};
};
