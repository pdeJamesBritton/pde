#solves the (k,x,y,t) case 
#same format and structure as problem.py

import jax
import jax.numpy as jnp
import numpy as np
import time
from functools import partial
 
from jax import vmap
from jax import jit                      
from jax import grad
from jax import random

import optax
import matplotlib.pyplot as plt

#----------------------------------------------------------
#important parameters

seed = np.random.randint(0,100000)
print("seed: ", seed)

#######
# what is an appropriat file extention? .txt? .csv?
save_params = False
params1_filename = '2d_k_params1_' + str(seed) + '.npy'
params2_filename = '2d_k_params2_' + str(seed) + '.npy'
params3_filename = '2d_k_params3_' + str(seed) + '.npy'
params4_filename = '2d_k_params4_' + str(seed) + '.npy'
params5_filename = '2d_k_params5_' + str(seed) + '.npy'

path = '/'

params_filenames = [params1_filename,params2_filename,params3_filename,params4_filename,params5_filename]

dim = 4 #total dimension of the problem
nn_arch = [10,10,10,1]

num_time_intervals = 8  #number of time intervals to consider
num_epochs =  50000 #120000
num_training_pts = 5000
#desired_loss = 0.00001

#for calculating norm and energy
nx_pts = 500 
ny_pts = 500
nt_pts = 11
nk_pts = 11

x_min = -8.0; x_max = 8.0
y_min = -8.0; y_max = 8.0
t_min = 0.0; t_max = 0.2
k_min = 1; k_max = 2
t_interval = t_max - t_min

exp_coeff = 60.0 #depends on size of t_interval

key = random.PRNGKey(seed)
print('key ', key)
key,xkey,ykey,tkey,kkey,paramskey = random.split(key,num=6)

learning_rate = 0.002
print_every_x_epochs = 5000

#----------------------------------------------------------
#function definitions

def sigmoid(s):
    return jnp.tanh(s)

#what diminsion does this return? int, double?
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


#  Please clarify this function.  I cannot continue without it.
def init_params(nn_arch,key):
    num_params = (dim + 1)*nn_arch[0]
    for i in range(1,len(nn_arch)):
        num_params += nn_arch[i]*(nn_arch[i-1] + 1)
     #num_params is not returned, what is its' purpose? 
      
      # what does below do?  the brackets are unbalanced.
      
    params = key, shape=(num_params,))
    return params

def V(x,y,k):
    v = 0.5*k*(x**2 + y**2)
    return v

#returns max(f,0) where f(t_min) = 1, f(t_max) = 0
#need max(f,0) to be smooth enough - determined by value of exp_coeff
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

V_vect = vmap(V,(0,0,0))
V_x = grad(V,0)
V_y = grad(V,1)
V_x_vect = jit(vmap(V_x,(0,0,0)))
V_y_vect = jit(vmap(V_y,(0,0,0)))

f_x = grad(f, 1)
f_xx = grad(f_x,1)
f_y = grad(f, 2)
f_yy = grad(f_y, 2)
f_t = grad(f, 3)

f_tx = grad(f_t,1)
f_ty = grad(f_t,2)
f_xxx = grad(f_xx,1)
f_xxy = grad(f_xx,2)
f_yyx = grad(f_yy,1)
f_yyy = grad(f_yy,2)

f_vect = jit(vmap(f, (None, 0, 0, 0, 0,None)))
f_x_vect = jit(vmap(f_x, (None, 0, 0, 0, 0, None)))
f_xx_vect = jit(vmap(f_xx, (None, 0, 0, 0, 0, None)))
f_y_vect = jit(vmap(f_y, (None, 0, 0, 0, 0, None)))
f_yy_vect = jit(vmap(f_yy, (None, 0, 0, 0, 0, None)))
f_t_vect = jit(vmap(f_t, (None, 0, 0, 0, 0, None)))

f_tx_vect = jit(vmap(f_tx,(None, 0, 0, 0, 0, None)))
f_ty_vect = jit(vmap(f_ty,(None, 0, 0, 0, 0, None)))
f_xxx_vect = jit(vmap(f_xxx,(None, 0, 0, 0, 0, None)))
f_xxy_vect = jit(vmap(f_xxy,(None, 0, 0, 0, 0, None)))
f_yyx_vect = jit(vmap(f_yyx, (None, 0, 0, 0, 0, None)))
f_yyy_vect = jit(vmap(f_yyy, (None, 0, 0, 0, 0, None)))

loss_fast = jit(loss)
loss_grad = grad(loss,0)
loss_grad_fast = jit(loss_grad)

#redefined functions for subsequent intervals
if(num_time_intervals > 1):
	def f2(params,old_params,x,y,t,k,t_min,t_max,norm):
		f = nn(params,x,y,t,k)*(1.0-ic_fn(t,t_min,t_max)) + ic_fn(t,t_min,t_max) * nn(old_params,x,y,t_min,k) / norm
		return f

	def loss2(params,old_params,x,y,t,k,t_min,t_max,norm):

		pde = f2_t_vect(params,old_params,x,y,t,k,t_min,t_max,norm) - 0.5*(f2_xx_vect(params,old_params,x,y,t,k,t_min,t_max,norm) + f2_yy_vect(params,old_params,x,y,t,k,t_min,t_max,norm)) + V_vect(x,y,k)*f2_vect(params,old_params,x,y,t,k,t_min,t_max,norm)
		
		pde_x = f2_tx_vect(params,old_params,x,y,t,k,t_min,t_max,norm) - 0.5*(f2_xxx_vect(params,old_params,x,y,t,k,t_min,t_max,norm) + f2_yyx_vect(params,old_params,x,y,t,k,t_min,t_max,norm)) + V_x_vect(x,y,k)*f2_vect(params,old_params,x,y,t,k,t_min,t_max,norm) + V_vect(x,y,k)* f2_x_vect(params,old_params,x,y,t,k,t_min,t_max,norm)

		pde_y = f2_ty_vect(params,old_params,x,y,t,k,t_min,t_max,norm) - 0.5*(f2_xxy_vect(params,old_params,x,y,t,k,t_min,t_max,norm) + f2_yyy_vect(params,old_params,x,y,t,k,t_min,t_max,norm)) + V_y_vect(x,y,k)*f2_vect(params,old_params,x,y,t,k,t_min,t_max,norm) + V_vect(x,y,k)*f2_y_vect(params,old_params,x,y,t,k,t_min,t_max,norm)
		
		loss_pde = jnp.mean(pde**2 + pde_x**2 + pde_y**2)
		return loss_pde

	f2_x = grad(f2, 2)
	f2_xx = grad(f2_x,2)
	f2_y = grad(f2, 3)
	f2_yy = grad(f2_y, 3)
	f2_t = grad(f2, 4)

	f2_tx = grad(f2_t,2)
	f2_ty = grad(f2_t,3)
	f2_xxx = grad(f2_xx,2)
	f2_xxy = grad(f2_xx,3)
	f2_yyx = grad(f2_yy,2)
	f2_yyy = grad(f2_yy,3)

	f2_vect = jit(vmap(f2, (None, None, 0, 0, 0, 0, 0, None, None)))
	f2_x_vect = jit(vmap(f2_x, (None, None, 0, 0, 0, 0, 0, None, None)))
	f2_xx_vect = jit(vmap(f2_xx, (None, None, 0, 0, 0, 0, 0, None, None)))
	f2_y_vect = jit(vmap(f2_y, (None, None, 0, 0, 0, 0, 0, None, None)))
	f2_yy_vect = jit(vmap(f2_yy, (None, None, 0, 0, 0, 0, 0, None, None)))
	f2_t_vect = jit(vmap(f2_t, (None, None, 0, 0, 0, 0, 0, None, None)))

	f2_tx_vect = jit(vmap(f2_tx,(None, None, 0, 0, 0, 0, 0, None,None)))
	f2_ty_vect = jit(vmap(f2_ty,(None, None, 0, 0, 0, 0, 0, None,None)))
	f2_xxx_vect = jit(vmap(f2_xxx,(None, None, 0, 0, 0, 0, 0, None,None)))
	f2_xxy_vect = jit(vmap(f2_xxy,(None, None, 0, 0, 0, 0, 0, None,None)))
	f2_yyx_vect = jit(vmap(f2_yyx, (None, None, 0, 0, 0, 0, 0, None,None)))
	f2_yyy_vect = jit(vmap(f2_yyy, (None, None, 0, 0, 0, 0, 0, None,None)))

	loss2_fast = jit(loss2)
	loss2_grad_fast = jit(grad(loss2,0))

#----------------------------------------------------------
#set up loop 1

#get points
x_coll = random.uniform(xkey,(num_training_pts,),minval=x_min,maxval = x_max)
y_coll = random.uniform(ykey,(num_training_pts,),minval=y_min,maxval = y_max)
t_coll = random.uniform(tkey,(num_training_pts,),minval=t_min,maxval = t_max)
k_coll = random.uniform(kkey,(num_training_pts,),minval=k_min,maxval = k_max)

#get parameters and initialize optimizer
params = init_params(nn_arch,paramskey)
params_size = jnp.size(params)

optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(params)

#track best parameters and loss
best_params = params
best_loss = 100

total_norm = np.zeros((nk_pts,nt_pts*num_time_intervals))
total_energy = np.zeros((nk_pts,nt_pts*num_time_intervals))
total_time = np.zeros((nt_pts*num_time_intervals,1))

#----------------------------------------------------------
#run NN

start_time = time.time()
for epoch in range(num_epochs+1):
    print('key ', key)
    #update parameters
    grads = loss_grad_fast(params,x_coll,y_coll,t_coll,k_coll,t_min)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)

	#calculate loss value and compare to previous losses
    loss_val = loss_fast(params,x_coll,y_coll,t_coll,k_coll,t_min)
    if(loss_val < best_loss):
        best_params = params
        best_loss = loss_val

    #print loss
    if(epoch % print_every_x_epochs == 0):
        print("epoch: ",epoch, "  loss: ", loss_val, "  best loss: ", best_loss)

print("----------")
print("best loss: ", best_loss)
print("time to run the NN: %s seconds" % round(time.time() - start_time, 2))
trained_params = best_params

#----------------------------------------------------------
#calculate norm and energy of solution
start_time = time.time()
dx = (x_max-x_min)/(nx_pts - 1)
dy = (y_max - y_min)/(ny_pts - 1)
dt = (t_max - t_min)/(nt_pts-1)
dk = (k_max - k_min)/(nk_pts-1)

xx = jnp.linspace(x_min, x_max, nx_pts)
yy = jnp.linspace(y_min, y_max, ny_pts)
tt = jnp.linspace(t_min, t_max, nt_pts)
kk = jnp.linspace(k_min, k_max, nk_pts)

KK,TT,XX,YY = jnp.meshgrid(kk,tt,xx,yy,indexing='ij') 

KK_1d = jnp.reshape(KK,(nt_pts*nx_pts*ny_pts*nk_pts))
TT_1d = jnp.reshape(TT,(nt_pts*nx_pts*ny_pts*nk_pts))
XX_1d = jnp.reshape(XX,(nt_pts*nx_pts*ny_pts*nk_pts))
YY_1d = jnp.reshape(YY,(nt_pts*nx_pts*ny_pts*nk_pts))

FF = f_vect(trained_params,XX_1d,YY_1d,TT_1d,KK_1d,t_min).reshape(nk_pts,nt_pts,nx_pts,ny_pts)
FF_x = f_x_vect(trained_params,XX_1d,YY_1d,TT_1d,KK_1d,t_min).reshape(nk_pts,nt_pts,nx_pts,ny_pts)
FF_y = f_y_vect(trained_params,XX_1d,YY_1d,TT_1d,KK_1d,t_min).reshape(nk_pts,nt_pts,nx_pts,ny_pts)

l2norm = np.zeros((nk_pts,nt_pts))
energy = np.zeros((nk_pts,nt_pts))

#for each k, remove last row and column from each FF[0][t]
for k in range(nk_pts):
    for t in range(nt_pts):
        l2norm[k][t] = (dx*dy * np.sum(FF[k][t][0:-1,0:-1]**2))**0.5
        energy[k][t] = (dx*dy * np.sum(0.5*(FF_x[k][t][0:-1,0:-1]**2+FF_y[k][t][0:-1,0:-1]**2) + V_vect(XX[k][t][0:-1,0:-1],YY[k][t][0:-1,0:-1],KK[k][t][0:-1,0:-1])*FF[k][t][0:-1,0:-1]**2)) / l2norm[k][t]**2
print("time to calculate norm and energy: %s seconds" % round(time.time() - start_time, 2))

total_norm[:,0:nt_pts] = l2norm
total_energy[:,0:nt_pts] = energy

for t in range(nt_pts):
    total_time[t] = t_min+dt*t

#get AVERAGE norm - otherwise norm will be a vector
norm = jnp.mean(l2norm[:,nt_pts-1])
avg_energy = jnp.mean(energy[:,nt_pts-1])

print('average norm at t_max: ', norm)
print('average energy at t_max: ', avg_energy)

#save parameters
if save_params == True:
    jnp.save(path + params_filenames[0], trained_params)

loss_values = np.zeros((num_epochs+1,num_time_intervals))

#next loops
if (num_time_intervals > 1):

    for i in range(num_time_intervals - 1):
        #increase time intervals
        t_min = t_max
        t_max = t_max + t_interval

        #get points
        x_coll = random.uniform(xkey,(num_training_pts,),minval=x_min,maxval = x_max)
        y_coll = random.uniform(ykey,(num_training_pts,),minval=y_min,maxval = y_max)
        t_coll = random.uniform(tkey,(num_training_pts,),minval=t_min,maxval = t_max)
        k_coll = random.uniform(kkey,(num_training_pts,),minval=k_min,maxval = k_max)
        tmin_coll = jnp.linspace(t_min,t_min,num_training_pts)

        #get parameters and initialize optimizer
        params = init_params(nn_arch,paramskey)
        params_size = jnp.size(params)

        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        #track best parameters and loss
        best_params = params
        old_params = trained_params
        best_loss = 100

        for epoch in range(num_epochs+1):

            #update parameters
            grads = loss2_grad_fast(params,old_params,x_coll,y_coll,t_coll,k_coll,tmin_coll,t_max,norm)
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

            loss_val = loss2_fast(params,old_params,x_coll,y_coll,t_coll,k_coll,tmin_coll,t_max,norm)

            if(loss_val < best_loss):
                best_params = params
                best_loss = loss_val

            #print losses
            if(epoch % print_every_x_epochs == 0):
                print("epoch: ",epoch, "  loss: ", loss_val)
            #print('i=',i)
            #print('epoch=',epoch)
            loss_values[epoch][i] = loss_val

        print("----------")
        print("best loss: ", best_loss)

        trained_params = best_params
        
        dx = (x_max-x_min) / (nx_pts - 1)
        dy = (y_max - y_min) / (ny_pts - 1)
        dt = (t_max - t_min) / (nt_pts-1)
        dk = (k_max - k_min) / (nk_pts - 1)

        xx = jnp.linspace(x_min, x_max, nx_pts)
        yy = jnp.linspace(y_min, y_max, ny_pts)
        tt = jnp.linspace(t_min, t_max, nt_pts)
        kk = jnp.linspace(k_min, k_max, nk_pts)
        KK,TT,XX,YY = jnp.meshgrid(kk,tt,xx,yy,indexing='ij') 

        KK_1d = jnp.reshape(KK,(nt_pts*nx_pts*ny_pts*nk_pts))
        TT_1d = jnp.reshape(TT,(nt_pts*nx_pts*ny_pts*nk_pts))
        XX_1d = jnp.reshape(XX,(nt_pts*nx_pts*ny_pts*nk_pts))
        YY_1d = jnp.reshape(YY,(nt_pts*nx_pts*ny_pts*nk_pts))

        TTmin_1d = jnp.linspace(t_min,t_min,nt_pts*nx_pts*ny_pts*nk_pts)

        FF = f2_vect(trained_params,old_params,XX_1d,YY_1d,TT_1d,KK_1d,TTmin_1d,t_max,norm).reshape(nk_pts,nt_pts,nx_pts,ny_pts)
        FF_x = f2_x_vect(trained_params,old_params,XX_1d,YY_1d,TT_1d,KK_1d,TTmin_1d,t_max,norm).reshape(nk_pts,nt_pts,nx_pts,ny_pts)
        FF_y = (f2_y_vect(trained_params,old_params,XX_1d,YY_1d,TT_1d,KK_1d,TTmin_1d,t_max,norm)).reshape(nk_pts,nt_pts,nx_pts,ny_pts)

        #get norm and energy for each t
        l2norm = np.zeros((nk_pts,nt_pts))
        energy = np.zeros((nk_pts,nt_pts))

        for k in range(nk_pts):
            for t in range(nt_pts):
                l2norm[k][t] = (dx*dy * np.sum(FF[k][t][0:-1,0:-1]**2))**0.5
                energy[k][t] = (dx*dy * np.sum(0.5*(FF_x[k][t][0:-1,0:-1]**2+FF_y[k][t][0:-1,0:-1]**2) + V_vect(XX[k][t][0:-1,0:-1],YY[k][t][0:-1,0:-1],KK[k][t][0:-1,0:-1])*FF[k][t][0:-1,0:-1]**2)) / l2norm[k][t]**2

        #add results to total
        total_norm[:,(i+1)*nt_pts:(i+2)*nt_pts] = l2norm
        total_energy[:,(i+1)*nt_pts:(i+2)*nt_pts] = energy
        for t in range(nt_pts):
            total_time[(i+1)*nt_pts + t] = t_min+dt*t

        norm = jnp.mean(l2norm[:,nt_pts-1])
        avg_energy = jnp.mean(energy[:,nt_pts-1])

        print('average norm at t_max: ', norm)
        print('average energy at t_max: ', avg_energy)

        #save parameters
        if save_params == True:
            jnp.save(path + params_filenames[i+1], trained_params)


#----------------------------------------------------------
#print results

t = jnp.squeeze(total_time)
k = jnp.linspace(k_min,k_max,nk_pts)

k_mesh,t_mesh = jnp.meshgrid(k,t,indexing='ij')

#3d plot
ax = plt.axes(projection = '3d')
ax.plot_surface(t_mesh,k_mesh,total_energy)
plt.show()

#2d plot
#plot k curves
for k in range(nk_pts):
    plot_label = "k = " + str(k_min + k*dk)
    plt.plot(t, total_energy[k],label=plot_label)
plt.legend()
plt.show()
np.savetxt('t',t)
np.savetxt('loss_tot',loss_values)
np.savetxt('total_energy',total_energy)
np.savetxt('total_norm',total_norm)





