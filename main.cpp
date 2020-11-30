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

  fstream learn_file{"learning.dat", ios::out};

  neural_network network{28 * 28, 30, 10};
  auto classification_rate =
      network.classification_rate(test_images, test_label_vectors);
  CAPTURE(classification_rate);
  for (size_t i = 0; i < 30; ++i) {
    network.train(train_images, train_label_vectors, 1, 10, 3.0);
    classification_rate =
        network.classification_rate(test_images, test_label_vectors);
    CAPTURE(classification_rate);
    auto training_rate =
        network.classification_rate(train_images, train_label_vectors);
    // CAPTURE(training_rate);
    auto test_error =
        network.mean_squared_error(test_images, test_label_vectors);
    auto train_error =
        network.mean_squared_error(train_images, train_label_vectors);
    learn_file << i << '\t' << training_rate << '\t' << train_error << '\t'
               << classification_rate << '\t' << test_error << '\n';
  }
  learn_file << flush;
  gpp classification_plot{};
  classification_plot << "plot 'learning.dat' u 1:2 w l title 'training', "  //
                         "'' u 1:4 w l title 'test'\n";
  gpp error_plot{};
  error_plot << "plot 'learning.dat' u 1:3 w l title 'training', "  //
                "'' u 1:5 w l title 'test'\n";

  gpp plot{};
  gpp pplot{};
  for (size_t n = 0; n < test_images.size(); ++n) {
    fstream file{"digit.dat", ios::out};
    for (int i = 0; i < 28; ++i) {
      for (int j = 0; j < 28; ++j) {
        file << test_images[n][28 * (28 - 1 - i) + j] << '\t';
      }
      file << '\n';
    }
    file << flush;
    plot << "plot 'digit.dat' matrix with image\n";

    neural_network::vector output = network(test_images[n]);
    fstream pfile{"output.dat", ios::out};
    for (int i = 0; i < 10; ++i) pfile << i << '\t' << output[i] << '\n';
    pfile << flush;
    pplot << "plot 'output.dat' using 1:2 w l\n";

    cout << test_labels[n] << '\t' << network.classification(test_images[n])
         << endl;

    // this_thread::sleep_for(1s);
    string str;
    getline(cin, str);
  }
}