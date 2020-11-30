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

#define CAPTURE(X) cout << #X << " = " << (X) << '\n'

int main() {
  using namespace std;
  using namespace chrono_literals;
  using namespace lyrahgames;

  vector<int> train_labels{};
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

  vector<array<float, 28 * 28>> train_images{};
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
      for (size_t j = 0; j < 28 * 28; ++j)
        train_images[i][j] = image[j] / 255.0f;
    }
  }

  gpp plot{};
  for (size_t n = 0; n < train_images.size(); ++n) {
    fstream file{"digit.dat", ios::out};
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        file << train_images[n][28 * (28 - 1 - i) + j] << '\t';
      }
      file << '\n';
    }
    file << flush;
    plot << "plot 'digit.dat' matrix with image\n";
    this_thread::sleep_for(500ms);
  }
}