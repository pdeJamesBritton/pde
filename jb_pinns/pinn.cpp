/**
 * @author James Britton
 * @date 2024-07-19 
 * @brief An implementation of a PINN in C++.  This is an expantion and reorganization of the referanced code.
 * 
 * 
 * @ref addapted code from:
 * https://github.com/nathanwbrei/phasm/tree/main/examples/pinn_pde_solver
 * and
 * https://github.com/nathanwbrei/phasm/blob/main/python/PhasmExampleHeatEquation.ipynb
 * 
 * 
 */
#include "matplotlibcpp.h"
#include <armadillo>
#include <torch/torch.h>
#include <math.h>
#include <iostream>
#include "/home/j/OneDrive/James/CompSci/004_Summer_2024/ML_PINNs/pde/jb_pinns/networkConstants.h"
#include <initializer_list>

using namespace torch::indexing;   // for tensor indexing

void get_whole_dataset_X(float* data)
{
    for (int ix = 0; ix < N; ix++)
        for (int iy = 0; iy < N; iy++) 
        {
            int idx_base = 2 * (ix * N + iy);
            data[idx_base] = ix * STEP_SIZE;
            data[idx_base + 1] = iy * STEP_SIZE;
        }
}

void get_bc_dataset_xTrain(float* data)
{
    for (int i = 0; i < N; i++)
    {
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
        for (int iy = 0; iy < N; iy++)
            data[ix * N + iy] = get_pde_f_term(ix * STEP_SIZE, iy * STEP_SIZE);
}

torch::Tensor get_pde_loss(torch::Tensor& u, torch::Tensor& X, torch::Device& device)
{
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
    // below has the form of:
    // Equation: - 2 * pi * pi * sin(pi * x) * sin(pi * y)
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
    torch::Tensor u = net.forward(net.model.vNetwork, X);
    return  torch::mse_loss(net.forward(net.model.vNetwork,X_train), Y_train) 
            +
            get_pde_loss(u, X, device);
}

std::vector<torch::nn::Linear> vLayers( int input_layer_size,
                                        int output_layer_size,
                                        int hidden_layer_size,
                                        int depth)// depth is the number of hidden layers
{
    /**
     * I took the idea and modified it below to make a vector of layers.
     * In which you declare in the function call.
     *      Declare the NN to match the Python code in
     *      https://github.com/nathanwbrei/phasm/blob/main/python/PhasmExampleHeatEquation.ipynb
     *      NN(
                (layers): Sequential(
                    (input):            Linear(in_features=2, out_features=20, bias=True)
                    (input_activation): Tanh()
                    (hidden_0):         Linear(in_features=20, out_features=20, bias=True)
                    (activation_0):     Tanh()
                    (hidden_1):         Linear(in_features=20, out_features=20, bias=True)
                    (activation_1):     Tanh()
                    (hidden_2):         Linear(in_features=20, out_features=20, bias=True)
                    (activation_2):     Tanh()
                    (hidden_3):         Linear(in_features=20, out_features=20, bias=True)
                    (activation_3):     Tanh()
                    (output):           Linear(in_features=20, out_features=1, bias=True)
          )
        )
     */
    
    // generats vector of layers with choosen depth
    std::vector<torch::nn::Linear> vLayers;
    vLayers.emplace_back(torch::nn::Linear(input_layer_size, hidden_layer_size));
    for(int i=0; i<depth; i++)
        vLayers.emplace_back(torch::nn::Linear(hidden_layer_size, hidden_layer_size));

    vLayers.emplace_back(torch::nn::Linear(hidden_layer_size, output_layer_size));
    return vLayers;
}

NN::NN(const std::vector<torch::nn::Linear> &initVector): vNetwork{initVector}
{
    // error checking to be added.
    std::cout<<"Registering Module"<<std::endl;
    //register_parameter("input", vNetwork[0]);
    register_module("input", vNetwork[0]);
    std::string hidden;
    int i=1;
    for(; i<(int)vNetwork.size()-1; i++)
    {
        hidden = "hidden_";
        //register_parameter(hidden.append(std::to_string(i)), vNetwork[i]);
        register_module(hidden.append(std::to_string(i)), vNetwork[i]);
    }
    //register_parameter("output", vNetwork[i]);
    register_module("output", vNetwork[i]);
}

std::vector<torch::nn::Linear> NN::get_Network(){ return vNetwork;}

HeatPINNetImpl::HeatPINNetImpl(const std::vector<torch::nn::Linear> &initList)
    : 
    model(initList),
    xx(arma::linspace(0, 1, N)),
    yy(arma::linspace(0, 1, N)),
    XX(N,N, arma::fill::zeros)
    //int input_layer_size, int output_layer_size, int hidden_layer_size)
{            
    
        
}


torch::Tensor HeatPINNetImpl::forward(std::vector<torch::nn::Linear> Network, torch::Tensor x)
{
            //activation function for the input and hidden layers
            int i = 0;
            for(; i< Network.size()-1; i++)
                x=torch::tanh(Network[i](x));
            x = Network[i](x);
            return x;
}

int HeatPINNetImpl::train(
                            torch::Tensor &loss_sum, 
                            HeatPINNetImpl& net,
                            torch::Tensor& X,
                            torch::Tensor& X_train,
                            torch::Tensor& Y_train,
                            torch::Device& device,
                            int options,
                            int max_iter,
                            int max_eval,
                            int history_size
                         )
{
            int iter = 0;
			// optimizer declaration.
            // using amsgrad(true) provides smoother convergence without the jumping around
            // see:
            //      https://openreview.net/forum?id=ryQu7f-RZ
            //      and
            //      https://pytorch.org/docs/stable/generated/torch.optim.Adam.html
			torch::optim::Adam adam_optim(net.model.parameters(), torch::optim::AdamOptions(1e-3).amsgrad(true));  // default Adam lr
            //adam_optim::amsgrad(true);
			
            
            // Python default value ref: https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
            torch::optim::LBFGSOptions 
                LBFGS_optim_options 
                =
                torch::optim::LBFGSOptions(options).max_iter(max_iter).max_eval(max_eval).history_size(history_size);
            torch::optim::LBFGS LBFGS_optim(net.model.parameters(), LBFGS_optim_options);            
            
            std::cout<<std::endl;
            std::cout<<"Entering training loop"<<std::endl;
            while(iter <MAX_STEPS)
            {
                auto closure = [&]() {
                    LBFGS_optim.zero_grad(); // why is this needed?
                    loss_sum = get_total_loss(net, X, X_train, Y_train, device);
                    loss_sum.backward();
                    return loss_sum;
                };
                //adam_optim.zero_grad();
                adam_optim.step(closure);
                /**
                 * if(iter < ADAM_STEPS)
                 *      adam_optim.step(closure);
                 * else
                 *      LBFGS_optim.step(closure);
                 * 
                */

                // print loss info
                if (iter % 1000 == 0)
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

void HeatPINNetImpl::forward_prediction(
        int input_layer_size,
        int output_layer_size,
        int hidden_layer_size,
        int depth
)
{
        std::vector<arma::Mat<double>> W_Layers, B_Layers, steps;
        W_Layers.emplace_back(arma::Mat<double>( hidden_layer_size, input_layer_size));
        B_Layers.emplace_back(arma::Mat<double>(hidden_layer_size, 1));
        std::cout<<"entering first loop"<<std::endl;
    
        for(int i=0; i<depth; i++){
           W_Layers.emplace_back(arma::Mat<double>(hidden_layer_size, hidden_layer_size));
           B_Layers.emplace_back(arma::Mat<double>(hidden_layer_size,1));
        }
        W_Layers.emplace_back(arma::Mat<double>( output_layer_size, hidden_layer_size));
        B_Layers.emplace_back(arma::Mat<double>(output_layer_size,1));

        int paramcount = 0;
        int w_count = 0;
        int b_count = 0;
        auto size = this->model.parameters().size();

        for ( const auto &p :this->model.parameters())//named_parameters())
        {
            if(p.dim() == 2)
            {
                for(int i = 0; i< p.size(0); i++)
                    for(int j = 0; j < p.size(1); j++)
                        W_Layers[w_count](i,j) = p[i][j].item<double>();
                w_count++;
            }else if(p.dim() ==1)
            {
                for(int i =0; i < p.size(0); i++)
                    B_Layers[b_count](i) = p[i].item<double>();
                b_count++;
            }
        }
        // now for iterating through data-grid in column-major.
        arma::Mat<double> xy_node(input_layer_size,1);

        for(int x=0; x<xx.size(); x++)
            for(int y=0; y<yy.size(); y++)
            {
                std::vector<arma::Mat<double>> copy(W_Layers);
                xy_node(0,0) = xx(x);
                xy_node(1,0) = yy(y);
                // input layer
                copy[0] = copy[0] * xy_node + B_Layers[0];

                for(int l=1; l<copy.size(); l++)
                {
                    for( auto &W : copy[l-1])
                        W = std::tanh(W);
                    copy[l] = copy[l] * copy[l-1] + B_Layers[l];
                }
                XX(x,y) = copy[copy.size()-1](0,0);
            }
        return;
}

int main()
{
    std::cout << "####### A cpp torch example with PINN heat equation. #######\n" << std::endl;

    /**
     * Init NN structure.
     */
    // Device
    auto cuda_available = torch::cuda::is_available();
    auto device_str = cuda_available ? torch::kCUDA : torch::kCPU;
    torch::Device device(device_str);
    std::cout << (cuda_available ? "CUDA available. Training on GPU.\n" : "Training on CPU.\n") << '\n';

    std::vector<torch::nn::Linear> layers= vLayers(NN_INPUT_SIZE, NN_OUTPUT_SIZE, NN_HIDDEN_SIZE, NN_DEPTH_SIZE);
    auto net = HeatPINNetImpl( layers );  // init a network model
    net.model.to(device);

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

    /**
     * 
     * 
     * 
     * Training process:
     *      After testing, Adam was used for all iterations
     */
    // optimizer declaration. All parameters are trying to match Python
    torch::optim::Adam adam_optim(net.model.parameters(), torch::optim::AdamOptions(1e-3));  // default Adam lr
    // Python default value ref: https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html
    torch::optim::LBFGSOptions LBFGS_optim_options =
            torch::optim::LBFGSOptions(1).max_iter(50000).max_eval(50000).history_size(50);
    torch::optim::LBFGS LBFGS_optim(net.model.parameters(), LBFGS_optim_options);

    torch::Tensor loss_sum;

    int options = 1;
    int max_iter = 100000;
    int max_eval = 100000;
    int history_size = 100;

    std::cout<<"Model: "<<std::endl;
    std::cout<<net.model<<std::endl;
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

    
    // Evaluation
    double h = 1.0/N;
    
    std::cout << "Evaluation and extract..." << std::endl;
    
    NN modeleval = net.model;
    modeleval.eval();
    
    torch::Tensor y_pred;
    torch::NoGradGuard no_grad;
    net.forward_prediction(
        NN_INPUT_SIZE,
            NN_OUTPUT_SIZE,
            NN_HIDDEN_SIZE,
            NN_DEPTH_SIZE);

    std::cout<< "Max: " << net.XX.max() << std::endl ;

    matplotlibcpp::plot(net.xx);
    matplotlibcpp::show();

    return 0;
}
