#include <algorithm>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

#include "rknn_api.h"

namespace {

std::vector<unsigned char> ReadBytes(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("failed to open model");
  const auto size = file.tellg();
  std::vector<unsigned char> data(static_cast<size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

template <typename T>
std::vector<T> ReadValues(const char* path, size_t count) {
  std::ifstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("failed to open artifact");
  std::vector<T> data(count);
  file.read(reinterpret_cast<char*>(data.data()), count * sizeof(T));
  if (!file) throw std::runtime_error("artifact size is too small");
  return data;
}

void WritePpm(const char* path, const float* data) {
  std::ofstream file(path, std::ios::binary);
  if (!file) throw std::runtime_error("failed to open image output");
  file << "P6\n512 512\n255\n";
  for (size_t i = 0; i < 512 * 512; ++i) {
    unsigned char pixel[3];
    for (size_t c = 0; c < 3; ++c) {
      const float value = std::min(std::max(data[c * 512 * 512 + i] * 0.5f + 0.5f, 0.0f), 1.0f);
      pixel[c] = static_cast<unsigned char>(value * 255.0f + 0.5f);
    }
    file.write(reinterpret_cast<const char*>(pixel), 3);
  }
}

std::vector<float> RunUnet(
    rknn_context context,
    const std::vector<float>& sample,
    const std::vector<float>& timestep,
    const std::vector<float>& hidden_states) {
  std::vector<__fp16> sample_fp16(sample.size());
  for (size_t h = 0; h < 64; ++h) {
    for (size_t w = 0; w < 64; ++w) {
      for (size_t c = 0; c < 4; ++c) {
        sample_fp16[h * 64 * 4 + w * 4 + c] = static_cast<__fp16>(sample[c * 64 * 64 + h * 64 + w]);
      }
    }
  }
  std::vector<__fp16> timestep_fp16(1);
  timestep_fp16[0] = static_cast<__fp16>(timestep[0]);
  std::vector<__fp16> hidden_fp16(hidden_states.size());
  for (size_t i = 0; i < hidden_states.size(); ++i) hidden_fp16[i] = static_cast<__fp16>(hidden_states[i]);

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
  inputs[2].buf = hidden_fp16.data();
  inputs[2].size = hidden_fp16.size() * sizeof(__fp16);
  inputs[2].pass_through = 1;
  inputs[2].type = RKNN_TENSOR_FLOAT16;
  inputs[2].fmt = RKNN_TENSOR_UNDEFINED;
  if (rknn_inputs_set(context, 3, inputs) < 0) throw std::runtime_error("UNet inputs failed");
  if (rknn_run(context, nullptr) < 0) throw std::runtime_error("UNet run failed");

  rknn_output output{};
  output.want_float = 1;
  if (rknn_outputs_get(context, 1, &output, nullptr) < 0) throw std::runtime_error("UNet output failed");
  std::vector<float> result(static_cast<float*>(output.buf), static_cast<float*>(output.buf) + 4 * 64 * 64);
  rknn_outputs_release(context, 1, &output);
  return result;
}

std::vector<float> RunVae(rknn_context context, const std::vector<float>& latents) {
  std::vector<__fp16> input_fp16(latents.size());
  for (size_t h = 0; h < 64; ++h) {
    for (size_t w = 0; w < 64; ++w) {
      for (size_t c = 0; c < 4; ++c) {
        input_fp16[h * 64 * 4 + w * 4 + c] = static_cast<__fp16>(latents[c * 64 * 64 + h * 64 + w]);
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
  if (rknn_inputs_set(context, 1, &input) < 0) throw std::runtime_error("VAE inputs failed");
  if (rknn_run(context, nullptr) < 0) throw std::runtime_error("VAE run failed");
  rknn_output output{};
  output.want_float = 1;
  if (rknn_outputs_get(context, 1, &output, nullptr) < 0) throw std::runtime_error("VAE output failed");
  std::vector<float> result(static_cast<float*>(output.buf), static_cast<float*>(output.buf) + 3 * 512 * 512);
  rknn_outputs_release(context, 1, &output);
  return result;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    std::cerr << "usage: fixed_prompt_rknn_demo <unet.rknn> <vae.rknn> <artifact_dir> [prompt_embeds.bin] [core_mask]\n";
    return 2;
  }
  try {
    const std::string dir = argv[3];
    const auto prompt = ReadValues<float>(argc == 5 ? argv[4] : (dir + "/prompt_embeds.bin").c_str(), 77 * 768);
    auto latents = ReadValues<float>((dir + "/initial_latents.bin").c_str(), 4 * 64 * 64);
    const auto timestep_bytes = ReadBytes((dir + "/timesteps.bin").c_str());
    const size_t steps = timestep_bytes.size() / sizeof(float);
    const auto timesteps = ReadValues<float>((dir + "/timesteps.bin").c_str(), steps);
    const auto coeffs = ReadValues<float>((dir + "/scheduler_coeffs.bin").c_str(), steps * 6);
    const auto model_unet = ReadBytes(argv[1]);
    const auto model_vae = ReadBytes(argv[2]);
    rknn_context unet = 0;
    rknn_context vae = 0;
    if (rknn_init(&unet, const_cast<unsigned char*>(model_unet.data()), model_unet.size(), 0, nullptr) < 0) throw std::runtime_error("UNet init failed");
    if (rknn_init(&vae, const_cast<unsigned char*>(model_vae.data()), model_vae.size(), 0, nullptr) < 0) throw std::runtime_error("VAE init failed");
    if (argc == 6) {
      const auto core_mask = static_cast<rknn_core_mask>(std::stoi(argv[5]));
      if (rknn_set_core_mask(unet, core_mask) < 0) throw std::runtime_error("UNet core mask failed");
      if (rknn_set_core_mask(vae, core_mask) < 0) throw std::runtime_error("VAE core mask failed");
      std::cout << "core mask: " << argv[5] << "\n";
    }

    const auto unet_start = std::chrono::steady_clock::now();
    for (size_t step = 0; step < steps; ++step) {
      const auto noise_pred = RunUnet(unet, latents, {timesteps[step]}, prompt);
      const float alpha = coeffs[step * 6 + 0];
      const float beta = coeffs[step * 6 + 1];
      const float alpha_prev = coeffs[step * 6 + 2];
      const float beta_prev = coeffs[step * 6 + 3];
      const float c_skip = coeffs[step * 6 + 4];
      const float c_out = coeffs[step * 6 + 5];
      std::vector<float> denoised(latents.size());
      for (size_t i = 0; i < latents.size(); ++i) {
        const float predicted = (latents[i] - beta * noise_pred[i]) / alpha;
        denoised[i] = c_out * predicted + c_skip * latents[i];
      }
      if (step + 1 < steps) {
        const auto noise = ReadValues<float>((dir + "/scheduler_noise_" + std::to_string(step) + ".bin").c_str(), latents.size());
        for (size_t i = 0; i < latents.size(); ++i) latents[i] = alpha_prev * denoised[i] + beta_prev * noise[i];
      } else {
        latents = denoised;
      }
      std::cout << "step " << (step + 1) << "/" << steps << " completed\n";
    }
    const auto unet_end = std::chrono::steady_clock::now();
    std::cout << "UNet " << steps << " steps: " << std::chrono::duration<double, std::milli>(unet_end - unet_start).count() << " ms\n";

    const auto vae_start = std::chrono::steady_clock::now();
    const auto image = RunVae(vae, latents);
    const auto vae_end = std::chrono::steady_clock::now();
    std::cout << "VAE decode: " << std::chrono::duration<double, std::milli>(vae_end - vae_start).count() << " ms\n";
    WritePpm((dir + "/board_image.ppm").c_str(), image.data());
    rknn_destroy(unet);
    rknn_destroy(vae);
    std::cout << "image saved\n";
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
