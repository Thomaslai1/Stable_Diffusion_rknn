#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "rknn_api.h"

namespace {

std::vector<float> ReadFloats(const char* path, size_t count) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("failed to open input");
  std::vector<float> data(count);
  file.read(reinterpret_cast<char*>(data.data()), count * sizeof(float));
  if (!file) throw std::runtime_error("input size is too small");
  return data;
}

std::vector<unsigned char> ReadBytes(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("failed to open model");
  const auto size = file.tellg();
  std::vector<unsigned char> data(static_cast<size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

void WriteFloats(const char* path, const float* data, size_t count) {
  std::ofstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("failed to open output");
  file.write(reinterpret_cast<const char*>(data), count * sizeof(float));
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 6) {
    std::cerr << "usage: unet_rknn_demo <model> <sample.bin> <timestep.bin> "
                 "<hidden_states.bin> <output.bin>\n";
    return 2;
  }

  try {
    const auto model = ReadBytes(argv[1]);
    const auto sample = ReadFloats(argv[2], 1 * 4 * 64 * 64);
    const auto timestep = ReadFloats(argv[3], 1);
    const auto hidden_states = ReadFloats(argv[4], 1 * 77 * 768);

    std::vector<__fp16> sample_fp16(sample.size());
    for (size_t h = 0; h < 64; ++h) {
      for (size_t w = 0; w < 64; ++w) {
        for (size_t c = 0; c < 4; ++c) {
          const size_t nchw = c * 64 * 64 + h * 64 + w;
          const size_t nhwc = h * 64 * 4 + w * 4 + c;
          sample_fp16[nhwc] = static_cast<__fp16>(sample[nchw]);
        }
      }
    }
    std::vector<__fp16> timestep_fp16(1);
    timestep_fp16[0] = static_cast<__fp16>(timestep[0]);
    std::vector<__fp16> hidden_states_fp16(hidden_states.size());
    for (size_t i = 0; i < hidden_states.size(); ++i) {
      hidden_states_fp16[i] = static_cast<__fp16>(hidden_states[i]);
    }

    rknn_context context = 0;
    int ret = rknn_init(&context, const_cast<unsigned char*>(model.data()), model.size(), 0, nullptr);
    if (ret < 0) throw std::runtime_error("rknn_init failed");

    rknn_input inputs[3]{};
    inputs[0].index = 0;
    inputs[0].buf = sample_fp16.data();
    inputs[0].size = sample_fp16.size() * sizeof(__fp16);
    inputs[0].pass_through = 1;
    inputs[0].type = RKNN_TENSOR_FLOAT16;
    inputs[0].fmt = RKNN_TENSOR_NHWC;

    inputs[1].index = 1;
    inputs[1].buf = timestep_fp16.data();
    inputs[1].size = timestep_fp16.size() * sizeof(__fp16);
    inputs[1].pass_through = 1;
    inputs[1].type = RKNN_TENSOR_FLOAT16;
    inputs[1].fmt = RKNN_TENSOR_UNDEFINED;

    inputs[2].index = 2;
    inputs[2].buf = hidden_states_fp16.data();
    inputs[2].size = hidden_states_fp16.size() * sizeof(__fp16);
    inputs[2].pass_through = 1;
    inputs[2].type = RKNN_TENSOR_FLOAT16;
    inputs[2].fmt = RKNN_TENSOR_UNDEFINED;

    ret = rknn_inputs_set(context, 3, inputs);
    if (ret < 0) throw std::runtime_error("rknn_inputs_set failed");
    ret = rknn_run(context, nullptr);
    if (ret < 0) throw std::runtime_error("rknn_run failed");

    rknn_output output{};
    output.want_float = 1;
    ret = rknn_outputs_get(context, 1, &output, nullptr);
    if (ret < 0) throw std::runtime_error("rknn_outputs_get failed");
    WriteFloats(argv[5], static_cast<const float*>(output.buf), 1 * 4 * 64 * 64);
    rknn_outputs_release(context, 1, &output);
    rknn_destroy(context);
    std::cout << "UNet inference completed\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
