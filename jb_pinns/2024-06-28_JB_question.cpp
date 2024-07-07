/**


Hello Prof. Lorin,

In searching for the correct function, I think that I may have found a different bug,
which I believe is implementing || N(O; x,y) - f(x,y)|| in correctly in this line at the
end of this code snippet.

Thanks,
James

*/


void get_whole_dataset_X(float* data) {
    for (int ix = 0; ix < N; ix++)
        for (int iy = 0; iy < N; iy++) {
            int idx_base = 2 * (ix * N + iy);
            data[idx_base] = ix * STEP_SIZE;
            data[idx_base + 1] = iy * STEP_SIZE;
        }
}

void get_bc_dataset_xTrain(float* data) {
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

float get_pde_f_term(float x, float y) {
    /**
     * Get the f term of the PDE.
     * f = -2 * pi * pi * sin(pi * x) * sin(pi * y)
     */
    return sin(PI * x) * sin(PI * y);
}

void get_fterm_dataset_f(float* data) {
    for (int ix = 0; ix < N; ix++)
        for (int iy = 0; iy < N; iy++) {
            data[ix * N + iy] = get_pde_f_term(ix * STEP_SIZE, iy * STEP_SIZE);
        }
}

torch::Tensor get_pde_loss(torch::Tensor& u, torch::Tensor& X, torch::Device& device) {
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
}


//////////////////////////////////////////////////////////
/*
why would they have this part then?
*/
torch::Tensor get_total_loss(
        HeatPINNNet& net,
        torch::Tensor& X, torch::Tensor& X_train, torch::Tensor& y_train,
	torch::Device& device
        ) {
    /**
     * Calculate the loss of each step.
     * loss_train is from the training dataset. loss_pde is from the whole dataset.
     */
    torch::Tensor u = net->forward(X);

    return torch::mse_loss(net->forward(X_train), y_train) + get_pde_loss(u, X, device);
}