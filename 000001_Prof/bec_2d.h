// First implementation attempt of bec_2d.py in cpp
// @date 2024-06-11
// @author James Britton
// @author Prof. Lorin
/**
 * @description
 * with same layout as prof.s bec_2d.py
 * placing function (prototypes) definitions here
 */


#include <mlpack.hpp>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>


/// @brief sets seed on interval [a,b], only call once
/// @param a 
/// @param b 
/// @return int
int seed(int a, int b, size_t seed){

    // generate interger between a & b
    int seed = mlpack::math::RandInt(a,b) + 1;
    std::cout<< "Seed: " << seed << std::endl;
    return seed;
}
/// @brief fills vector with random keys on interval [a, b]
/// @param key 
/// @param a 
/// @param b 
void setkeys(std::vector<int>key, int a, int b){
    for( auto i : key)
        i = mlpack::math::RandInt(a,b)+1;
    
    for(auto i : key)
        std::cout<<"Keys " + std::to_string(i)<< std::endl;
    
    return;
}


/// @brief  makes a vector to store param file names
/// @param  int seed
/// @return std::vector<std::string>
std::vector<std::string>paramsVfn(int seed){
    std::vector<std::string> v;
    for (int i=1; i<6; i++){
        v.push_back("2d_k_params0"+ std::to_string(i)+"_"+std::to_string(seed)+ ".txt");// txt file for now, need to think of a npy equivalent
    }
    return v;
}

// functioin defn's

/// @brief
/// 
double sigmoid(double s){
    return std::tanh(s);
}
def init_params(nn_arch,key):
    num_params = (dim + 1)*nn_arch[0]
    for i in range(1,len(nn_arch)):
        num_params += nn_arch[i]*(nn_arch[i-1] + 1)
    params = key, shape=(num_params,))
    return params

def nn(params, x, y, t, k):
    nn_arch_len = len(nn_arch)
    w = params[: nn_arch[0]*dim]
    w = jnp.reshape(w, (nn_arch[0], dim))
    index = nn_arch[0]*dim
    b = params[index : index + nn_arch[0]]
    z = sigmoid(w[:,0]*x + w[:, 1]*y + w[:, 2]*t + w[:, 3]*k + b)
    index = index +  nn_arch[0]
    for i in range(1, nn_arch_len - 1):
        w = params[index:index + nn_arch[i]*nn_arch[i - 1]]
        w = jnp.reshape(w, (nn_arch[i], nn_arch[i - 1]))
        index += nn_arch[i]*nn_arch[i - 1] 
        b = params[index:index + nn_arch[i]]
        index += nn_arch[i]
        z = sigmoid(jnp.dot(w,z) + b)
    w = params[index:index + nn_arch[nn_arch_len - 1]*nn_arch[nn_arch_len - 2]]
    b = params[index + nn_arch[nn_arch_len - 1]*nn_arch[nn_arch_len - 2]]
    z = jnp.dot(w,z) + b
    return z


def V(x,y,k):
    v = 0.5*k*(x**2 + y**2)
    return v

def ic_fn(t,t_min,t_max):
    a = 1.0/( jnp.exp(-exp_coeff*t_min) - jnp.exp(-exp_coeff*t_max))
    c = jnp.exp(-exp_coeff*t_max) / ( jnp.exp(-exp_coeff*t_max) - jnp.exp(-exp_coeff*t_min))
    f = a*jnp.exp(-exp_coeff*t)+c
    compare = jnp.array([f,0.0])
    return jnp.max(compare)

def f(params,x,y,t,k,t_min):
    f = nn(params,x,y,t,k)*(1.0-ic_fn(t,t_min,t_max)) + ic_fn(t,t_min,t_max)*((2.0 / jnp.pi)**0.5 * jnp.exp(-1.0*(x**2 + y**2) ))
    return f

def loss(params,x,y,t,k,t_min):
    pde = f_t_vect(params,x,y,t,k,t_min) - 0.5*(f_xx_vect(params,x,y,t,k,t_min) + f_yy_vect(params,x,y,t,k,t_min)) + V_vect(x,y,k)*f_vect(params,x,y,t,k,t_min)

    pde_x = f_tx_vect(params,x,y,t,k,t_min) -0.5*(f_xxx_vect(params,x,y,t,k,t_min) + f_yyx_vect(params,x,y,t,k,t_min)) + V_x_vect(x,y,k)*f_vect(params,x,y,t,k,t_min) + V_vect(x,y,k)*f_x_vect(params,x,y,t,k,t_min)

    pde_y = f_ty_vect(params,x,y,t,k,t_min) - 0.5*(f_xxy_vect(params,x,y,t,k,t_min) + f_yyy_vect(params,x,y,t,k,t_min)) + V_y_vect(x,y,k)*f_vect(params,x,y,t,k,t_min) + V_vect(x,y,k)*f_y_vect(params,x,y,t,k,t_min)

    loss_pde = jnp.mean(pde**2 + pde_x**2 + pde_y**2)
    return loss_pde