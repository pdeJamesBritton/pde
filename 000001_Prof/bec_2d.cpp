// First implementation attempt of bec_2d.py in cpp
// @date 2024-06-11
// @author James Britton
// @author Prof. Lorin
/**
 *  @description
 *  with same layout as prof.s bec_2d.py
 */


#include <mlpack.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <cmath>
#include "bec_2d.h"



//
int main(){
    
    // Initialize random seed:
    size_t set_seed;
    mlpack::math::RandomSeed(set_seed);
    int seed = mlpack::math::RandInt(0, 10000) + 1;    

    int dim, num_time_intervals, num_epochs, num_training_pts, desired_loss, nx_pts, ny_pts, nt_pts, nk_pts, x_min, x_max, y_min, y_max, t_min, t_max, k_min, k_max, t_interval, learning_rate, pring_every_x_epochs;

    dim = 4; // total dimension of the problem
    std::vector<int> nn_arch = {10,10,10,1};

    num_time_intervals = 8;
    num_epochs = 5000;
    num_training_pts = 5000;
    // desired_loss = 0.0001;

    // for calculation norm and energy
    nx_pts = 500;
    ny_pts = 500;
    nt_pts = 11;
    nk_pts = 11;

    x_min = -8.0; x_max = 8.0;
    y_min = -8.0; y_max = 8.0;
    t_min =  0.0; t_max = 0.2;
    k_min =  1.0; k_max = 2.0; // may need to change to an int
    t_interval = t_max - t_min;

    double exp_coeff = 60.0;

    std::vector<int> vKeys = {1,2,3,4,5,6};
    setkeys(vKeys);


    learning_rate = 0.002;
    pring_every_x_epochs = 5000;

    return 0;
}
