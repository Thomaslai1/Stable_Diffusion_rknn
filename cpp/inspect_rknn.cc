#include <iostream>
#include <vector>
#include <fstream>

#include "rknn_api.h"

std::vector<unsigned char> ReadModel(const char* path) {
  std::ifstream file(path, std::ios::binary | std::ios::ate);
  if (!file) throw std::runtime_error("failed to open model");
  const auto size = file.tellg();
  std::vector<unsigned char> data(static_cast<size_t>(size));
  file.seekg(0);
  file.read(reinterpret_cast<char*>(data.data()), size);
  return data;
}

void PrintAttr(rknn_context context, rknn_query_cmd query, uint32_t index) {
  rknn_tensor_attr attr{};
  attr.index = index;
  if (rknn_query(context, query, &attr, sizeof(attr)) < 0) {
    throw std::runtime_error("rknn_query failed");
  }
  std::cout << "index=" << index << " name=" << attr.name << " type=" << attr.type
            << " fmt=" << attr.fmt << " size=" << attr.size << " dims=" << attr.n_dims;
  for (uint32_t i = 0; i < attr.n_dims; ++i) std::cout << " " << attr.dims[i];
  std::cout << "\n";
}

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: inspect_rknn <model.rknn>\n";
    return 2;
  }
  try {
    const auto model = ReadModel(argv[1]);
    rknn_context context = 0;
    if (rknn_init(&context, const_cast<unsigned char*>(model.data()), model.size(), 0, nullptr) < 0) {
      throw std::runtime_error("rknn_init failed");
    }
    rknn_input_output_num io_num{};
    if (rknn_query(context, RKNN_QUERY_IN_OUT_NUM, &io_num, sizeof(io_num)) < 0) {
      throw std::runtime_error("query io number failed");
    }
    std::cout << "inputs=" << io_num.n_input << " outputs=" << io_num.n_output << "\n";
    for (uint32_t i = 0; i < io_num.n_input; ++i) PrintAttr(context, RKNN_QUERY_INPUT_ATTR, i);
    for (uint32_t i = 0; i < io_num.n_output; ++i) PrintAttr(context, RKNN_QUERY_OUTPUT_ATTR, i);
    rknn_destroy(context);
  } catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
  }
  return 0;
}
