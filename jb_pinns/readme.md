2024-06-20

copied and tested code from:
https://github.com/nathanwbrei/phasm/tree/main/examples/pinn_pde_solver


Playing with torch

torch::randn({2,3} , torch::kCUDA) // makes cuda version, does this mean that it runs on the gpu?
	how would I detect if I have a gpu available?
		If I can do this, I should make a function that takes care of this for me.
		
Solution at:
https://github.com/ollewelin/libtorch-GPU-CNN-test-MNIST-with-Batchnorm/blob/main/main.cpp		
