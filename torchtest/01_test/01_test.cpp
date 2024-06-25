#include <torch/torch.h>
#include <iostream>

int main() {
  torch::Tensor tensor = torch::rand({5, 3}, torch::kCUDA);
  std::cout << tensor << std::endl;
}