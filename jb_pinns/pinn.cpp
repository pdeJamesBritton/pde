/**
 * @author James Britton
 * @date 2024-06-26
 * @brief taking orriginal code repository for github, completing it and making it into a class structure
 * 
 * @ref addapted code from:
 * https://github.com/nathanwbrei/phasm/tree/main/examples/pinn_pde_solver
 * and
 * https://github.com/nathanwbrei/phasm/blob/main/python/PhasmExampleHeatEquation.ipynb
 * 
 * 
 */

#include <torch/torch.h>
#include <math.h>
#include <iostream>
#include "/home/james/OneDrive/James/CompSci/004_Summer_2024/ML_PINNs/pde/jb_pinns/networkConstants.h"
#include <initializer_list>

using namespace torch::indexing;   // for tensor indexing


/////////////////////////////////
void get_whole_dataset_X(float* data) 
{
    for (int ix = 0; ix < N; ix++)
        for (int iy = 0; iy < N; iy++) {
            int idx_base = 2 * (ix * N + iy);
            data[idx_base] = ix * STEP_SIZE;
            data[idx_base + 1] = iy * STEP_SIZE;
        }
}

void get_bc_dataset_xTrain(float* data) 
{
    for (int i = 0; i < N; i++) {
        int idx_base = 2 * i;
        float num = i * STEP_SIZE;
        // x: left, right, down, top
        data[idx_base] = 0.0;
        data[idx_base + 2 * N] = 1.0;
        data[idx_base + 4 * N] = num;
        data[idx_base + 6 * N] = num;

        // y: left, right, down, top
        idx_base += 1;
        data[idx_base] = num;
        data[idx_base + 2 * N] = num;
        data[idx_base + 4 * N] = 0.0;
        data[idx_base + 6 * N] = 1.0;
    }
}

float get_pde_f_term(float x, float y) 
{
    /**
     * Get the f term of the PDE.
     * f = -2 * pi * pi * sin(pi * x) * sin(pi * y)
     */
    return sin(PI * x) * sin(PI * y);
}

void get_fterm_dataset_f(float* data) 
{
    for (int ix = 0; ix < N; ix++)
        for (int iy = 0; iy < N; iy++) {
            data[ix * N + iy] = get_pde_f_term(ix * STEP_SIZE, iy * STEP_SIZE);
        }
}

torch::Tensor get_pde_loss(torch::Tensor& u, torch::Tensor& X, torch::Device& device){
    /**
     * Get the pde loss based on the NN forward results.
     * Calculate the gradients and the pde terms.
     */

    // get the gradients
    torch::Tensor du_dX = torch::autograd::grad(
            /*output=*/{u},
            /*input=*/{X},
            /*grad_outputs=*/{torch::ones_like(u)},
            /*retain_graph=*/true,
            /*create_graph=*/true,
            /*allow_unused=*/true)[0];

    torch::Tensor du_dx = du_dX.index({"...", 0});
    torch::Tensor du_dy = du_dX.index({"...", 1});

    torch::Tensor du_dxx = torch::autograd::grad({du_dx}, {X}, {torch::ones_like(du_dx),},
                                                 true, true, true)[0].index({"...", 0});
    torch::Tensor du_dyy = torch::autograd::grad({du_dy}, {X}, {torch::ones_like(du_dy),},
                                                 true, true, true)[0].index({"...", 1});
//    std::cout << "du_dxx + du_dyy:\n" << du_dxx + du_dyy << std::endl;

    // get constant term f_X
    float f_data[WHOLE_GRID_SIZE];
    get_fterm_dataset_f(f_data);
    torch::Tensor f_X = -2.0 * PI * PI * torch::from_blob(f_data, {WHOLE_GRID_SIZE}).to(device);

    return torch::mse_loss(du_dxx + du_dyy, f_X);
};

torch::Tensor get_total_loss(
        HeatPINNetImpl& net,
        torch::Tensor& X,
        torch::Tensor& X_train,
        torch::Tensor& Y_train,
        torch::Device& device
        ) 
{
    /**
     * Calculate the loss of each step.
     * loss_train is from the training dataset. loss_pde is from the whole dataset.
     */
    //std::cout<<"IN get_total_loss"<<std::endl;
    
    torch::Tensor u = net.forward(net.get_vNetwork(), X);
    //std::cout<< "about to mse_loss"<<std::endl;
    return torch::mse_loss(net.forward(net.get_vNetwork(),X_train), Y_train) + get_pde_loss(u, X, device);
}
/*
void closure(   torch::Tensor &loss_sum, 
                torch::optim::LBFGS LBFGS_optim,
                HeatPINNetImpl& net,
                torch::Tensor& X,
                torch::Tensor& X_train,
                torch::Tensor& Y_train,
                torch::Device& device)
{
    LBFGS_optim.zero_grad();
    loss_sum = get_total_loss(net, X, X_train, Y_train, device);
    loss_sum.backward();
    return;
}
*/

std::vector<torch::nn::Linear> vLayers( int input_layer_size,
                                        int output_layer_size,
                                        int hidden_layer_size,
                                        int depth){// depth is the number of hidden layers
    // generats vector of layers with choosen depth
    std::vector<torch::nn::Linear> vLayers;
    vLayers.emplace_back(torch::nn::Linear(input_layer_size, hidden_layer_size));
    for(int i=0; i<depth; i++)
        vLayers.emplace_back(torch::nn::Linear(hidden_layer_size, hidden_layer_size));

    vLayers.emplace_back(torch::nn::Linear(hidden_layer_size, output_layer_size));
    return vLayers;
}

HeatPINNetImpl::HeatPINNetImpl(std::vector<torch::nn::Linear> initList):
            //int input_layer_size, int output_layer_size, int hidden_layer_size)
            
            vNetwork(initList)
            
        {            
    /**
     * This I have addapted
     * 
     * Declare the NN to match the Python code in
     * https://github.com/nathanwbrei/phasm/blob/main/python/PhasmExampleHeatEquation.ipynb
     * NN(
          (layers): Sequential(
            (input): Linear(in_features=2, out_features=20, bias=True)
            (input_activation): Tanh()
            (hidden_0): Linear(in_features=20, out_features=20, bias=True)
            (activation_0): Tanh()
            (hidden_1): Linear(in_features=20, out_features=20, bias=True)
            (activation_1): Tanh()
            (hidden_2): Linear(in_features=20, out_features=20, bias=True)
            (activation_2): Tanh()
            (hidden_3): Linear(in_features=20, out_features=20, bias=True)
            (activation_3): Tanh()
            (output): Linear(in_features=20, out_features=1, bias=True)
          )
        )
     */
    // register module
      {      
            std::cout<<"Registering Module"<<std::endl;
            register_module("input", vNetwork[0]);
            std::string hidden;
            int i=1;
            for(; i<(int)vNetwork.size()-1; i++){
                hidden = "hidden_";
                register_module(hidden.append(std::to_string(i)), vNetwork[i]);
            }
            register_module("output", vNetwork[i]);
      }
    //public:
    
        
        
        
    //private:
        std::vector< torch::nn::Linear > vNetwork;
        
};

torch::Tensor HeatPINNetImpl::forward(std::vector<torch::nn::Linear> Network, torch::Tensor x){
            //activation function for the input and hidden layers
            //std::cout<<" forward"<<std::endl;
            int i = 0;
            for(; i< Network.size()-1; i++)
            {
              //  std::cout<<" forward in loop at:"<< std::to_string(i)<<std::endl;
                x=torch::tanh(Network[i](x));
            }
            x = Network[i](x);
            //std::cout<<"End of forward"<<std::endl;
            return x;
}
int HeatPINNetImpl::train(torch::Tensor &loss_sum, 
                HeatPINNetImpl& net,
                torch::Tensor& X,
                torch::Tensor& X_train,
                torch::Tensor& Y_train,
                torch::Device& device,
                int options,
                int max_iter,
                int max_eval,
                int history_size)
        {
            int iter = 0;
			// optimizer declaration. All parameters are trying to match Python
			torch::optim::Adam adam_optim(net.parameters(), torch::optim::AdamOptions(1e-3));  // default Adam lr
			// Python default value ref: https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
			torch::optim::LBFGSOptions LBFGS_optim_options =
            torch::optim::LBFGSOptions(options).max_iter(max_iter).max_eval(max_eval).history_size(history_size);
			torch::optim::LBFGS LBFGS_optim(net.parameters(), LBFGS_optim_options);
            std::cout<<std::endl;
            std::cout<<"Entering training loop"<<std::endl;
            while(iter <MAX_STEPS)
            {
                auto closure = [&]() {
                    //std::cout<< "In loop"<<std::endl;
                    LBFGS_optim.zero_grad();
                    //std::cout<< "In loop"<<std::endl;
                    loss_sum = get_total_loss(net, X, X_train, Y_train, device);
                    //std::cout<< "In loop"<<std::endl;
                    loss_sum.backward();
                    //std::cout<< "In loop"<<std::endl;
                    return loss_sum;
                };
                
                if(iter < ADAM_STEPS){
                    adam_optim.step(closure);
                    
                }
                else
                    LBFGS_optim.step(closure);
                // print loss info
                if (iter % 100 == 0) 
                {
                    std::cout << "  iter=" << iter << ", loss=" << std::setprecision(7) << loss_sum.item<float>();
                    std::cout << ", loss.device().type()=" << loss_sum.device().type() << std::endl;
                }
                // stop training
                if (loss_sum.item<float>() < TARGET_LOSS)   
                {
                    iter ++;
                    break;
	            }

                iter ++;
            }
            
            std::cout << "\nTraining stopped." << std::endl;
            std::cout << "Final iter=" << iter - 1 << ", loss=" << std::setprecision(7) << loss_sum.item<float>();
            std::cout << ", loss.device().type()=" << loss_sum.device().type() << std::endl;
            
            return iter;
    
        }
std::vector< torch::nn::Linear > HeatPINNetImpl::get_vNetwork(){ return vNetwork;}
//TORCH_MODULE(HeatPINNetImpl); //? a wrapped shared_ptr, see official tutorial
//WHAT TO DO WITH ABOVE?

std::vector<torch::nn::Linear> vMake_Layers(
        int input_layer_size,
        int output_layer_size,
        int hidden_layer_size,
        int depth
){
        std::vector<torch::nn::Linear> vMake_Layers;
        vMake_Layers.push_back(torch::nn::Linear(input_layer_size, hidden_layer_size));
        for(int i=0; i<=depth; i++)
           vMake_Layers.push_back(torch::nn::Linear(hidden_layer_size, hidden_layer_size));
        vMake_Layers.push_back(torch::nn::Linear(hidden_layer_size, output_layer_size));
        
        return vMake_Layers;
}

int main() {
    std::cout << "####### A cpp torch example with PINN heat equation. #######\n" << std::endl;

    /**
     * Init NN structure.
     */
    // Device
    auto cuda_available = torch::cuda::is_available();
    auto device_str = cuda_available ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_str);
    std::cout << (cuda_available ? "CUDA available. Training on GPU.\n" : "Training on CPU.\n") << '\n';

    std::vector<torch::nn::Linear> layers= vMake_Layers(NN_INPUT_SIZE, NN_OUTPUT_SIZE, NN_HIDDEN_SIZE, NN_DEPTH_SIZE);
    auto net = HeatPINNetImpl( layers );  // init a network model
    net.to(device);

    /**
     * Init data sets.
     */
    // supervised training data set
    torch::Tensor Y_train, X_train, X;
    // TODO: seems must choose kFloat32 data type now because of the NN declaration. Check later.
    Y_train = torch::zeros({BD_SIZE, NN_OUTPUT_SIZE}, device);
    std::cout << "Y_train sizes: " << Y_train.sizes() << std::endl;
    std::cout << "Y_train.device().type(): " << Y_train.device().type() << std::endl;
    std::cout << "Y_train.requires_grad(): " << Y_train.requires_grad() << std::endl;


    float X_train_data[BD_INPUT_SIZE];
    get_bc_dataset_xTrain(X_train_data);
    X_train = torch::from_blob(X_train_data, {BD_SIZE, NN_INPUT_SIZE}).to(device);
    //X_train = torch::from_blob(X_train_data, {BD_SIZE, NN_INPUT_SIZE}, options);
    std::cout << "X_train sizes: " << X_train.sizes() << std::endl;
    std::cout << "X_train.device().type(): " << X_train.device().type() << std::endl;
    std::cout << "X_train.requires_grad(): " << X_train.requires_grad() << std::endl;

    // whole data set
    float X_data[WHOLE_INPUT_DATA_SIZE];
    get_whole_dataset_X(X_data);
    X = torch::from_blob(X_data, {WHOLE_GRID_SIZE, NN_INPUT_SIZE}, torch::requires_grad()).to(device);
    std::cout << "X sizes: " << X.sizes() << std::endl;
    std::cout << "X.device().index(): " << X.device().index() << std::endl;
    std::cout << "X.requires_grad(): " << X.requires_grad() << std::endl;

    /*
     * Training process
     *  The training steps are trying to match the Python script.
     *  First 1000 steps use Adam, and remaining steps use LBFGS.
     */
    // optimizer declaration. All parameters are trying to match Python
    torch::optim::Adam adam_optim(net.parameters(), torch::optim::AdamOptions(1e-3));  // default Adam lr
    // Python default value ref: https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
    torch::optim::LBFGSOptions LBFGS_optim_options =
            torch::optim::LBFGSOptions(1).max_iter(50000).max_eval(50000).history_size(50);
    torch::optim::LBFGS LBFGS_optim(net.parameters(), LBFGS_optim_options);

    torch::Tensor loss_sum;

    int options = 1;
    int max_iter = 50000;
    int max_eval = 50000;
    int history_size = 50;

    std::cout<<"training"<<std::endl;

    int final_inter = net.train(    loss_sum, 
                                    net,
                                    X,
                                    X_train,
                                    Y_train,
                                    device,
                                    options,
                                    max_iter,
                                    max_eval,
                                    history_size);

    return 0;
}










