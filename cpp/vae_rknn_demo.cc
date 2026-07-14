#include <cstdint>
#include <fstream>
#include <iostream>
#include <vector>

#include "rknn_api.h"

namespace {

std::vector<uint8_t> ReadBytes(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("failed to open file");
  const auto size = file.tellg();
  std::vector<uint8_t> data(static_cast<size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

std::vector<float> ReadFloats(const char* path, size_t count) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("failed to open input");
  std::vector<float> data(count);
  file.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
  if (!file) throw std::runtime_error("input size is too small");
  return data;
}

void WriteFloats(const char* path, const float* data, size_t count) {
  std::ofstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("failed to open output");
  file.write(reinterpret_cast<const char*>(data), count * sizeof(float));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage: vae_rknn_demo <model.rknn> <input.bin> <output.bin>\n";
    return 2;
  }

  try {
    const auto model = ReadBytes(argv[1]);
    constexpr size_t kInputCount = 1 * 4 * 64 * 64;
    constexpr size_t kOutputCount = 1 * 3 * 512 * 512;
    const auto input_data = ReadFloats(argv[2], kInputCount);

    rknn_context context = 0;
    int ret = rknn_init(&context, const_cast<uint8_t*>(model.data()), model.size(), 0, nullptr);
    if (ret < 0) throw std::runtime_error("rknn_init failed");
    std::cerr << "rknn_init ok\n";

    rknn_tensor_attr input_attr{};
    input_attr.index = 0;
    ret = rknn_query(context, RKNN_QUERY_INPUT_ATTR, &input_attr, sizeof(input_attr));
    if (ret < 0) throw std::runtime_error("rknn_query input failed");
    std::cerr << "input type=" << input_attr.type << " fmt=" << input_attr.fmt
              << " dims=" << input_attr.n_dims << " size=" << input_attr.size << "\n";

    rknn_tensor_attr output_attr{};
    output_attr.index = 0;
    ret = rknn_query(context, RKNN_QUERY_OUTPUT_ATTR, &output_attr, sizeof(output_attr));
    if (ret < 0) throw std::runtime_error("rknn_query output failed");
    std::cerr << "output type=" << output_attr.type << " fmt=" << output_attr.fmt
              << " dims=" << output_attr.n_dims << " size=" << output_attr.size << "\n";

    std::vector<__fp16> input_fp16(kInputCount);
    for (size_t h = 0; h < 64; ++h) {
      for (size_t w = 0; w < 64; ++w) {
        for (size_t c = 0; c < 4; ++c) {
          const size_t nchw_index = c * 64 * 64 + h * 64 + w;
          const size_t nhwc_index = h * 64 * 4 + w * 4 + c;
          input_fp16[nhwc_index] = static_cast<__fp16>(input_data[nchw_index]);
        }
      }
    }

    rknn_input input{};
    input.index = 0;
    input.buf = input_fp16.data();
    input.size = input_fp16.size() * sizeof(__fp16);
    input.pass_through = 1;
    input.type = RKNN_TENSOR_FLOAT16;
    input.fmt = RKNN_TENSOR_NHWC;

    ret = rknn_inputs_set(context, 1, &input);
    if (ret < 0) throw std::runtime_error("rknn_inputs_set failed");
    ret = rknn_run(context, nullptr);
    if (ret < 0) throw std::runtime_error("rknn_run failed");

    rknn_output output{};
    output.want_float = 1;
    ret = rknn_outputs_get(context, 1, &output, nullptr);
    if (ret < 0) throw std::runtime_error("rknn_outputs_get failed");
    WriteFloats(argv[3], static_cast<const float*>(output.buf), kOutputCount);
    rknn_outputs_release(context, 1, &output);
    rknn_destroy(context);
    std::cout << "VAE inference completed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
