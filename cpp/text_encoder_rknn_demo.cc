#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "rknn_api.h"

std::vector<unsigned char> ReadBytes(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("failed to open file");
  const auto size = file.tellg();
  std::vector<unsigned char> data(static_cast<size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: text_encoder_rknn_demo <model.rknn> <input_ids.bin> <output.bin>\n";
    return 2;
  }
  try {
    const auto model = ReadBytes(argv[1]);
    const auto ids = ReadBytes(argv[2]);
    if (ids.size() != 77 * sizeof(int32_t)) throw std::runtime_error("input_ids must contain 77 int32 values");
    rknn_context context = 0;
    if (rknn_init(&context, const_cast<unsigned char*>(model.data()), model.size(), 0, nullptr) < 0) {
      throw std::runtime_error("text encoder init failed");
    }
    rknn_input input{};
    input.index = 0;
    input.buf = const_cast<unsigned char*>(ids.data());
    input.size = ids.size();
    input.pass_through = 1;
    input.type = RKNN_TENSOR_INT32;
    input.fmt = RKNN_TENSOR_UNDEFINED;
    if (rknn_inputs_set(context, 1, &input) < 0) throw std::runtime_error("text encoder input failed");
    if (rknn_run(context, nullptr) < 0) throw std::runtime_error("text encoder run failed");
    rknn_output output{};
    output.want_float = 1;
    if (rknn_outputs_get(context, 1, &output, nullptr) < 0) throw std::runtime_error("text encoder output failed");
    std::ofstream file(argv[3], std::ios::binary);
    file.write(static_cast<const char*>(output.buf), 77 * 768 * sizeof(float));
    rknn_outputs_release(context, 1, &output);
    rknn_destroy(context);
    std::cout << "prompt embedding saved\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
