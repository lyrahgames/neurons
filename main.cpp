#include <array>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <thread>
#include <vector>
//
#include <lyrahgames/gpp.hpp>
#include <neural_network.hpp>

#define CAPTURE(X) cout << #X << " = " << (X) << '\n'

int main() {
  using namespace std;
  using namespace chrono_literals;
  using namespace lyrahgames;

  vector<int> train_labels{};
  vector<neural_network::vector> train_images{};

  {
    fstream file{"train-labels.idx1-ubyte"};

    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic) / sizeof(char));
    magic = __builtin_bswap32(magic);
    assert(magic == 2049);

    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size) / sizeof(char));
    size = __builtin_bswap32(size);

    train_labels.resize(size);
    for (size_t i = 0; i < size; ++i) {
      uint8_t label;
      file.read(reinterpret_cast<char*>(&label), sizeof(label) / sizeof(char));
      train_labels[i] = static_cast<int>(label);
    }
  }

  vector<neural_network::vector> train_label_vectors(train_labels.size());
  for (size_t i = 0; i < train_label_vectors.size(); ++i) {
    train_label_vectors[i] = neural_network::vector::Zero(10);
    train_label_vectors[i][train_labels[i]] = 1.0f;
  }

  {
    fstream file{"train-images.idx3-ubyte"};

    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic) / sizeof(char));
    magic = __builtin_bswap32(magic);
    assert(magic == 2051);

    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size) / sizeof(char));
    size = __builtin_bswap32(size);
    assert(size == train_labels.size());

    uint32_t rows;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows) / sizeof(char));
    rows = __builtin_bswap32(rows);
    assert(rows == 28);

    uint32_t cols;
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols) / sizeof(char));
    cols = __builtin_bswap32(cols);
    assert(cols == 28);

    train_images.resize(size);
    for (size_t i = 0; i < size; ++i) {
      array<uint8_t, 28 * 28> image;
      file.read(reinterpret_cast<char*>(&image), sizeof(image) / sizeof(char));

      train_images[i] = neural_network::vector::Zero(28 * 28);
      for (size_t j = 0; j < 28 * 28; ++j)
        train_images[i][j] = image[j] / 255.0f;
    }
  }

  vector<int> test_labels{};
  vector<neural_network::vector> test_images{};
  {
    fstream file{"t10k-labels.idx1-ubyte"};

    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic) / sizeof(char));
    magic = __builtin_bswap32(magic);
    assert(magic == 2049);

    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size) / sizeof(char));
    size = __builtin_bswap32(size);

    test_labels.resize(size);
    for (size_t i = 0; i < size; ++i) {
      uint8_t label;
      file.read(reinterpret_cast<char*>(&label), sizeof(label) / sizeof(char));
      test_labels[i] = static_cast<int>(label);
    }
  }

  vector<neural_network::vector> test_label_vectors(test_labels.size());
  for (size_t i = 0; i < test_label_vectors.size(); ++i) {
    test_label_vectors[i] = neural_network::vector::Zero(10);
    test_label_vectors[i][test_labels[i]] = 1.0f;
  }

  {
    fstream file{"t10k-images.idx3-ubyte"};

    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic) / sizeof(char));
    magic = __builtin_bswap32(magic);
    assert(magic == 2051);

    uint32_t size;
    file.read(reinterpret_cast<char*>(&size), sizeof(size) / sizeof(char));
    size = __builtin_bswap32(size);
    assert(size == test_labels.size());

    uint32_t rows;
    file.read(reinterpret_cast<char*>(&rows), sizeof(rows) / sizeof(char));
    rows = __builtin_bswap32(rows);
    assert(rows == 28);

    uint32_t cols;
    file.read(reinterpret_cast<char*>(&cols), sizeof(cols) / sizeof(char));
    cols = __builtin_bswap32(cols);
    assert(cols == 28);

    test_images.resize(size);
    for (size_t i = 0; i < size; ++i) {
      array<uint8_t, 28 * 28> image;
      file.read(reinterpret_cast<char*>(&image), sizeof(image) / sizeof(char));
      test_images[i] = neural_network::vector::Zero(28 * 28);
      for (size_t j = 0; j < 28 * 28; ++j)
        test_images[i][j] = image[j] / 255.0f;
    }
  }

  // gpp plot{};
  // for (size_t n = 0; n < 2; ++n) {
  //   fstream file{"digit.dat", ios::out};
  //   for (int i = 0; i < 28; ++i) {
  //     for (int j = 0; j < 28; ++j) {
  //       file << train_images[n][28 * (28 - 1 - i) + j] << '\t';
  //     }
  //     file << '\n';
  //   }
  //   file << flush;
  //   plot << "plot 'digit.dat' matrix with image\n";
  //   this_thread::sleep_for(5ms);
  // }

  neural_network network{28 * 28, 30, 10};
  auto classification_rate =
      network.classification_rate(test_images, test_label_vectors);
  CAPTURE(classification_rate);
  for (size_t i = 0; i < 30; ++i) {
    network.train(train_images, train_label_vectors, 1, 100, 3.0);
    classification_rate =
        network.classification_rate(test_images, test_label_vectors);
    CAPTURE(classification_rate);
  }
}