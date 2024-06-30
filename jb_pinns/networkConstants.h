// constants
// can vary sizes

#include <torch/torch.h>
#include <initializer_list>

constexpr float PI = 3.14159265358979323846;

const int N = 65;  // grid dimension
const float STEP_SIZE = 1.0 / (N - 1);
const int WHOLE_GRID_SIZE = N * N;
const int WHOLE_INPUT_DATA_SIZE = WHOLE_GRID_SIZE * 2;
const int BD_SIZE = 4 * N;
const int BD_INPUT_SIZE = BD_SIZE * 2;

const int ADAM_STEPS = 1000;

// max step is set based on experience. 5000 steps make loss at e-5 level.
const int MAX_STEPS = 5000;
// const int MAX_STEPS = 1;  // from ncu compiling
// criteria for stop training.
// When loss is at 1.x~e-5, it will stop degrading even the training continues
const float TARGET_LOSS = 5.0e-5;

const int NN_INPUT_SIZE = 2;
const int NN_OUTPUT_SIZE = 1;
const int NN_HIDDEN_SIZE = 20;
const int NN_DEPTH_SIZE = 3;


class NN: public torch::nn::Module{
        public:
                NN(const std::vector<torch::nn::Linear> &initVector);

                torch::Tensor forward(std::vector<torch::nn::Linear> Network,torch::Tensor x);

                std::vector<torch::nn::Linear> get_Network();

        private:
                std::vector<torch::nn::Linear> vNetwork;
};
class HeatPINNetImpl: public NN {
  
    public:
    //constructor
        HeatPINNetImpl(const std::vector<torch::nn::Linear> &init): NN(init);
        //using NN::NN;
        
        

        int train(torch::Tensor &loss_sum, 
                HeatPINNetImpl& net,
                torch::Tensor& X,
                torch::Tensor& X_train,
                torch::Tensor& Y_train,
                torch::Device& device,
                int options,
                int max_iter,
                int max_eval,
                int history_size);
        std::vector< torch::nn::Linear > get_vNetwork();
    private:
        //std::vector< torch::nn::Linear > vNetwork;
        NN model;
};

void get_whole_dataset_X(float* data);
void get_bc_dataset_xTrain(float* data);
float get_pde_f_term(float x, float y);
void get_fterm_dataset_f(float* data);
torch::Tensor get_pde_loss(torch::Tensor& u, torch::Tensor& X, torch::Device& device);
torch::Tensor get_total_loss(
        HeatPINNetImpl& net,
        torch::Tensor& X,
        torch::Tensor& X_train,
        torch::Tensor& y_train,
        torch::Device& device
        ) ;
/*
void closure(   torch::Tensor &loss_sum, 
                torch::optim::LBFGS LBFGS_optim,
                HeatPINNetImpl& net,
                torch::Tensor& X,
                torch::Tensor& X_train,
                torch::Tensor& Y_train,
                torch::Device& device);
*/

// make layers
std::vector<torch::nn::Linear> vMake_Layers(
        int input_layer_size,
        int output_layer_size,
        int hidden_layer_size,
        int depth
);