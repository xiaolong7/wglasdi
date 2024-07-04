import numpy as np
import tensorflow.compat.v1 as tf
#tf.compat.v1.enable_eager_execution
tf.compat.v1.disable_v2_behavior()
import pickle
from autoencoder_wsindy import full_network, define_loss, silu
from sindy_utils import sindy_simulate, derivative, derivative2
import wsindy_utils
from time import time
import seaborn as sns
import random
from error_utils import *
import copy
from sklearn.linear_model import LinearRegression
import seaborn as sns
import matplotlib.patches as patches
from copy import deepcopy
import matplotlib.pyplot as plt

def train_network(training_data, params):
    latent_dim = params['latent_dim']
    library_dim = params['library_dim']
    test_data = params['test_data']
    if isinstance(test_data, str):
        print('Load test data')
        test_data = pickle.load(open(test_data, "rb"))
    
    if 'save_frequency' not in params.keys():
        params['save_frequency'] = 1e8
    
    if params['retrain']:
        testing_losses = params['testing_losses']
        training_losses = params['training_losses']
        if 'err_array' in params.keys():
            err_array = params['err_array']
        else:
            err_array = []
            
        if 'max_err_idx_param' in params.keys():
            max_err_idx_param = params['max_err_idx_param']
        else:
            max_err_idx_param = [[]]
            
        if 'wsindy_idx' in params.keys():
            wsindy_idx = params['wsindy_idx']
        else:
            wsindy_idx = []
            
        if 'epoch_count' in params.keys():
            epoch_count = params['epoch_count']   # epoch counter
            params['max_epochs'] += epoch_count
        else:
            epoch_count = 0
        param_flag = False
        
    else:
        testing_losses = []
        training_losses = []
        err_array = []
        max_err_idx_param = [[]]
        wsindy_idx = []
        epoch_count = 0   # epoch counter
    wsindy_model_terms = [np.sum(params['coefficient_mask'])]
    
    train_flag = True # flag to indicate whether to perform training
    w = 1 # counter for the level of random subset evalutions
    while epoch_count < params['max_epochs'] and train_flag:
        tf.reset_default_graph() # clear graph
        tf.set_random_seed(params['seed'])
        np.random.seed(params['seed'])
        random.seed(params['seed'])
        
        if epoch_count > 0:
            if param_flag:
                print(f'* Updating graph')
            else:
                print(f'* NOT updating graph and continue Training')
        
        # Create models
        autoencoder_network = full_network(params)
        loss,losses,_ = define_loss(autoencoder_network, params)
        learning_rate = tf.placeholder(tf.float64, name='learning_rate')
        train_op = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
        saver = tf.train.Saver(var_list=tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)) 
        
        param_flag = True # flag to indicate whether to update parameters
        with tf.Session(config = params['config']) as sess:
            sess.run(tf.global_variables_initializer())
            params['coeff_exist'] = True
            wsindy_coefficients = []
            for i in range(len(autoencoder_network['wsindy_coefficients'])):
                wsindy_coefficients.append(sess.run(autoencoder_network['wsindy_coefficients'][i], feed_dict={}))
            params['model_params'] = []
            params['model_params'].append(wsindy_coefficients)
            params['model_params'].append(sess.run(autoencoder_network['encoder_weights'], feed_dict={}))
            params['model_params'].append(sess.run(autoencoder_network['encoder_biases'], feed_dict={}))
            params['model_params'].append(sess.run(autoencoder_network['decoder_weights'], feed_dict={}))
            params['model_params'].append(sess.run(autoencoder_network['decoder_biases'], feed_dict={}))
              
            if epoch_count == 0:
                print(f'* Evaluating')
                # random subset evaluation
                err_array_tmp, max_err, idx, param_tmp, wsindy_idx_tmp = err_map_subset(params, test_data=test_data,
                                                                                       err_type=params['err_type'])
                testing_losses.append(max_err)
                err_array.append(err_array_tmp)
                max_err_idx_param.append([])
                wsindy_idx.append(wsindy_idx_tmp)

            batch_overlap = params['batch_overlap']
            b = params['batch_size']
            a = 0
            index = [[a, b]]
            while b - batch_overlap + params['batch_size'] < params['epoch_size']:
                a = b - batch_overlap
                b = a + params['batch_size']
                index.append([a, b])
                
            for i in range(min(params['update_epoch'],params['max_epochs'])): # loop over epochs
                for j in range(len(index)): #range(params['epoch_size']//params['batch_size']): # loop over batches

                    '''''
                    if j == 0: 
                        start_ind = 0
                        end_ind = params['batch_size']
                    else:
                        start_ind = j*params['batch_size'] - 3*params['L']
                        end_ind = (j+1)*params['batch_size']  - 3*params['L']
                    '''''
                   
                    start_ind = index[j][0]
                    end_ind = index[j][1]
                    # print(start_ind, end_ind)

                    batch_idxs = np.arange(start_ind, end_ind )
                    train_dict = create_feed_dictionary(training_data, params, idxs=batch_idxs)
                    sess.run(train_op, feed_dict=train_dict)
                
                if params['print_progress'] and (i % params['print_frequency'] == 0):
                    train_loss = print_progress(sess, epoch_count, loss, losses, train_dict)
                    training_losses.append(train_loss)
                epoch_count += 1
                
                if params['sequential_thresholding'] and (i % params['threshold_frequency'] == 0) and (i > 0):
                    params['coefficient_mask'] = np.abs(sess.run(autoencoder_network['wsindy_coefficients'])) > params['coefficient_threshold']
                    print('THRESHOLDING: %d active coefficients' % np.sum(params['coefficient_mask']))
                    wsindy_model_terms.append(np.sum(params['coefficient_mask']))
                    
                if i % params['save_frequency'] == 0: # Save current model state 
                    params['training_losses'] = training_losses
                    params['epoch_count'] = epoch_count
                    wsindy_coefficients = []
                    for i in range(len(autoencoder_network['wsindy_coefficients'])):
                        wsindy_coefficients.append(sess.run(autoencoder_network['wsindy_coefficients'][i], feed_dict={}))
                    params['model_params'] = []
                    params['model_params'].append(wsindy_coefficients)
                    params['model_params'].append(sess.run(autoencoder_network['encoder_weights'], feed_dict={}))
                    params['model_params'].append(sess.run(autoencoder_network['encoder_biases'], feed_dict={}))
                    params['model_params'].append(sess.run(autoencoder_network['decoder_weights'], feed_dict={}))
                    params['model_params'].append(sess.run(autoencoder_network['decoder_biases'], feed_dict={}))
                    pickle.dump(params, open(params['fig_path'] + params['save_name'] + '_params.pkl', 'wb'))
                    saver.save(sess, params['fig_path'] + params['save_name'])
            
            # Store updated trainable parameters before evaluation on random subset
            wsindy_coefficients = []
            for i in range(len(autoencoder_network['wsindy_coefficients'])):
                wsindy_coefficients.append(sess.run(autoencoder_network['wsindy_coefficients'][i], feed_dict={}))
            params['model_params'] = []
            params['model_params'].append(wsindy_coefficients)
            params['model_params'].append(sess.run(autoencoder_network['encoder_weights'], feed_dict={}))
            params['model_params'].append(sess.run(autoencoder_network['encoder_biases'], feed_dict={}))
            params['model_params'].append(sess.run(autoencoder_network['decoder_weights'], feed_dict={}))
            params['model_params'].append(sess.run(autoencoder_network['decoder_biases'], feed_dict={}))
            
            
            # Evaluate current model on a random subset of parameters using a specified error indicator
            print(f'* Evaluating')
            err_array_tmp, max_err, idx, param_tmp, wsindy_idx_tmp = err_map_subset(params, test_data=test_data,
                                                                                   err_type=params['err_type'])
            testing_losses.append(max_err)
            err_array.append(err_array_tmp)
            max_err_idx_param.append(copy.copy(max_err_idx_param[-1]))
            wsindy_idx.append(wsindy_idx_tmp)
            
            
            # Update tolerance for error indicator
            tol_old = params['tol']
            params['tol'], max_rel_err = update_tol(params, training_data, err_type=params['err_type'])
            print(f"  Max rel. err.: {max_rel_err:.1f}%, Update tolerance for error indicator from {tol_old:.5f} to {params['tol']:.5f}")
            
            
            # Save current model state 
            # should save the state before new parameters are added, consistent with the graph
            params['training_losses'] = training_losses
            params['testing_losses'] = testing_losses
            params['err_array'] = err_array
            params['max_err_idx_param'] = max_err_idx_param
            params['wsindy_idx'] = wsindy_idx
            params['epoch_count'] = epoch_count
            pickle.dump(params, open(params['fig_path'] + params['save_name'] + '_params.pkl', 'wb'))
            final_losses = sess.run((losses['decoder'], losses['wsindy_x'], losses['wsindy_z'],
                                     losses['wsindy_regularization']), feed_dict=train_dict)
            saver.save(sess, params['fig_path'] + params['save_name'])
            
            
            # Update training dataset and parameter set
            for i in training_data['param']:
                if np.linalg.norm(i-test_data['param'][idx]) < 1e-8:
                    print(f"  PARAMETERS EXIST, NOT adding it! idx: {idx} param: {test_data['param'][idx]}")
                    param_flag = False
                    break
            if param_flag:
                print(f'* Update Training set: add case {param_tmp}')
                training_data['data'].append(test_data['data'][idx])
                training_data['param'].append(test_data['param'][idx])
                params['num_wsindy'] += 1
                params['param'] = training_data['param']
                params['train_idx'].append(idx)
                max_err_idx_param[-1].append((idx, param_tmp))
                    
            # Update random subset size
            subset_ratio = params['subsize']/params['num_test']*100 # new subset size
            if max_err <= params['tol']:
                w += 1
                if params['subsize']*2 <= params['num_test']:
                    params['subsize'] *= 2 # double the random subset size for evaluation
                else:
                    params['subsize'] = params['num_test']
                subset_ratio = params['subsize']/params['num_test']*100 # new subset size
                print(f"  Max error indicator <= Tol! Current subset ratio {subset_ratio:.1f}%")

            # check termination criterion
            if 'wsindy_max' in params.keys() and params['wsindy_max'] != None: # prescribed number of local DIs
                if params['num_wsindy'] == params['wsindy_max']+1:
                    print(f"  Max # WSINDys {params['wsindy_max']:d} is reached! Training done!")
                    train_flag = False 
            elif subset_ratio >= params['subsize_max']: # prescribed error toerlance
                print(f"  Current subset ratio {subset_ratio:.1f}% >= Target subset ratio {params['subsize_max']:.1f}%!")
                train_flag = False 
                
    results_dict = {}
    results_dict['num_epochs'] = epoch_count
    results_dict['wsindy_coefficients'] = wsindy_coefficients
    results_dict['model_params'] = params['model_params']
    results_dict['loss_decoder'] = final_losses[0]
    results_dict['loss_decoder_wsindy'] = final_losses[1]
    results_dict['loss_wsindy'] = final_losses[2]
    results_dict['loss_wsindy_regularization'] = final_losses[3]
    results_dict['testing_losses'] = np.array(testing_losses)
    results_dict['training_losses'] = np.array(training_losses)
    results_dict['wsindy_model_terms'] = np.array(wsindy_model_terms)
    results_dict['err_array'] = err_array
    results_dict['max_err_idx_param'] = max_err_idx_param
    results_dict['wsindy_idx'] = wsindy_idx
    return results_dict
    

def softplus(x):
    # return np.log(1 + np.exp(x))
    return np.log(1 + np.exp(-np.abs(x))) + np.maximum(x,0)

def NN(x, weights, biases, activation):
    """
    This network serves as either an encoder or a decoder, 
    where the output layer has a linear activation.
    """
    num_layers = len(weights)
    for i in range(num_layers-1):
        x = np.matmul(x, weights[i]) + biases[i]
        if activation == 'tanh':
            x = np.tanh(x)
        elif activation == 'sigmoid':
            x = 1.0 / (1.0 + np.exp(-x)) # sigmoid
        elif activation == 'relu':
            x = np.maximum(0,x)
        elif activation == 'linear_tanh':
            x = linear_tanh(x)
        elif activation == 'linear_sigmoid':
            x = linear_sigmoid(x)
        elif activation == 'softplus':
            x = softplus(x)
            
    # output layer (linear activation)
    x = np.matmul(x, weights[-1]) + biases[-1]
    return x #output

def linear_tanh(x):
    condition1 = x < -1
    condition2 = np.logical_and(x >= -1, x <= 1)
    condition3 = x > 1
    return np.where(condition1, -1, np.where(condition2, x, 1))


def linear_sigmoid(x):
    cond1 = x < -5
    cond2 = np.logical_and(x >= -5, x < -4)
    cond3 = np.logical_and(x >= -4, x < -3)
    cond4 = np.logical_and(x >= -3, x < -2)
    cond5 = np.logical_and(x >= -2, x < -1)
    cond6 = np.logical_and(x >= -1, x < 0)
    cond7 = np.logical_and(x >= 0, x < 1)
    cond8 = np.logical_and(x >= 1, x < 2)
    cond9 = np.logical_and(x >= 2, x < 3)
    cond10 = np.logical_and(x >= 3, x < 4)
    cond11 = np.logical_and(x >= 4, x < 5)
    cond12 = x >= 5

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))


    def line_sigmoid(x, x1, x2):
        y1 = sigmoid(x1)
        y2 = sigmoid(x2)
        slope = (y1 - y2)/(x1 - x2)
        intercept = y1 - slope*x1
        return slope*x + intercept
    
    f = cond1*0 + cond2*line_sigmoid(x, -5, -4) + cond3*line_sigmoid(x, -4, -3) + cond4*line_sigmoid(x, -3, -2) + cond5*line_sigmoid(x, -2, -1) + cond6*line_sigmoid(x, -1, 0) + cond7*line_sigmoid(x, 0, 1) + cond8*line_sigmoid(x, 1, 2) + cond9*line_sigmoid(x, 2, 3) + cond10*line_sigmoid(x, 3, 4) + cond11*line_sigmoid(x, 4, 5) + cond12*1
    return f

#def linear_sigmoid(x):
    #condition1 = x < -3
    #condition2 = np.logical_and(x >= -3, x <= 3)
    #condition3 = x > 3
    #return np.where(condition1, 0, np.where(condition2, 1/6*x + 1/2, 1)) 




def eval_model(test_data, params, test_param, idx=None, knn=4, calc_dz=False, calc_du=False):
    """
    This function evaluates the gLaSDI model on a given testing parameter case.
    
    inputs:
        test_data: dict, data of testing parameter case (could be updated to just provide the 
                    initial condition of the testing parameter case)
        params: dict, parameters of the gLaSDI model
        test_param: parameters of the testing case
        idx: int or None, the index of the DI used for evaluation; used only when knn=1; if knn=1 and it is None, 
             the DI closest to the testing parameter will be used.
        knn: int, the number of nearest local DIs used for convex interpolation of the DI coefficient matrix of the testing case
        calc_dz: bool, whether or not to calculate dz/dt
        calc_du: bool, whether or not to calculate du/dt
    
    outputs: 
        u_decoder: autoencoder reconstruction of full-order model solution u
        du_decoder: autoencoder prediction of du/dt
        u_sim: gLaSDI prediction of u by DI and decoder
        du_sim: gLaSDI prediction of du/dt by DI and decoder
        z_encoder: encoder-predicted latent-space dynamics
        dz_encoder: encoder-predicted gradient of latent-space dynamics
        z_sim: DI-predicted latent-space dynamics
        dz_sim: DI-predicted gradient of latent-space dynamics
        idx: the index of the DI closest to the testing case based on the Euclidean distance
        timer_rom: computational time of each step
    """
    timer = []
    
    # Step 1: Encoder-predicted latent-space dynamics and autoencoder reconstruction,
    # which is excluded from the measurement of ROM computational time
    timer.append(time()) 
    include_sine = False
    include_cosine = False
    if 'include_sine' in params.keys():
        include_sine = params['include_sine']
    if 'include_cosine' in params.keys():
        include_cosine = params['include_cosine']
        
    z_encoder = NN(test_data['x'], params['model_params'][1], params['model_params'][2], params['activation']) # encoder
    u_decoder = NN(z_encoder, params['model_params'][3], params['model_params'][4], params['activation']) # decoder
    
    
    # Step 2: find the nearest neighbor (optional)
    timer.append(time()) 
    if idx == None:
        train_param = np.stack(params['param'])
        idx = np.argmin(np.linalg.norm(train_param-test_param, axis=1))
    
    
    # Step 3: calculate DI coefficient matrix
    timer.append(time())        
    if knn == 1:
        wsindy_coeff = params['model_params'][0][idx]
        
    else: # KNN convex interpolation of coefficient matrix
        dist = np.linalg.norm(train_param-test_param, axis=1)
        knn_idx = np.argsort(dist)[:knn]
        phi = np.zeros_like(knn_idx)
        if dist[knn_idx[0]] == 0: # check if the min distance is zero
            phi[0] = 1
        else:
            phi = 1 / np.linalg.norm(train_param[knn_idx]-test_param, axis=1)**2
        psi = phi / phi.sum()

        wsindy_coeff = np.zeros(params['model_params'][0][0].shape)
        for i,kidx in enumerate(knn_idx):
            wsindy_coeff += psi[i] * params['model_params'][0][kidx]
            

    # Step 4: DI-predicted lastent-space dynamics and physical dynamics
    timer.append(time())
    if ('x0' in test_data.keys()):
        z0 = NN(test_data['x0'], params['model_params'][1], params['model_params'][2], params['activation'])
    else:
        z0 = z_encoder[0,:]
    z_sim = wsindy_utils.wsindy_simulate(z0, test_data['t'].squeeze(), 
                           wsindy_coeff, params['poly_order'], 
                           include_sine,include_cosine)
    u_sim = NN(z_sim, params['model_params'][3], params['model_params'][4], params['activation'])

    plt.plot(z_sim)
    plt.plot(z_encoder)
    timer.append(time())
    timer1 = np.array(timer)
    timer2 = timer1[1:]
    timer_rom = timer2 - timer1[:-1]
    
    # calculate dz/dt, du/dt  
    if calc_dz:
        # dz_encoder = derivative(z_encoder,params['pde']['tstop'])
        # dz_sim = derivative(z_sim,params['pde']['tstop'])
        dz_encoder = derivative2(z_encoder,params['pde']['dt'])
        dz_sim = derivative2(z_sim,params['pde']['dt'])
    else:
        dz_encoder = 0
        dz_sim = 0
        
    if calc_du:
        # du_decoder = derivative(u_decoder,params['pde']['tstop'])
        # du_sim = derivative(u_sim,params['pde']['tstop'])
        du_decoder = derivative2(u_decoder,params['pde']['dt'])
        du_sim = derivative2(u_sim,params['pde']['dt'])
    else:
        du_decoder = 0
        du_sim = 0
    return u_decoder, du_decoder, u_sim, du_sim, z_encoder, dz_encoder, z_sim, dz_sim, idx, timer_rom


def err_map_subset(params, test_data=None, err_type=1):
    """
    This function computes errors in random subsets of the parameter space 
    using a speciffied error indicator. The subset size and the threshold for 
    error indicator are adjusted during training.
    inputs:
        params: dict, parameters of the gLaSDI model
        test_data: dict, testing data
        err_type: int, types of error indicator. 
                 1: max relative error (if test data is available)
                 2: residual norm for 1D Burgers eqn
                 3: residual norm for 2D Burgers eqn
                 4: residual norm for time dependent heat conduction (MFEM example 16)
                 5: residual norm for radial advection (MFEM example 9)
    outputs:
        err_array: ndarray, errors in the parameter space
        err_max: float, max error
        err_idx: int, index of the max error
        test_data['param'][err_idx]: ndarray, parameters of the case with max error
        wsindy_idx: ndarray, indices of local WSINDys used for evaluation
    """
    amp = params['test_param'][:,0]
    width = params['test_param'][:,1]
    err_array = np.zeros([amp.size, width.size])
    err_array2 = np.zeros([amp.size, width.size])
    wsindy_idx = np.zeros([amp.size, width.size])
    
    # select a random subset for evaluation
    rng = np.random.default_rng()
    # a = np.setdiff1d(np.arange(params['num_test']),params['train_idx']) # exclude existing training cases
    # rng.shuffle(a)
    # subset = a[:params['subsize']]
    
    if 'local_dist' in params.keys():
        ld = params['local_dist']
    else:
        ld = 0.05
    
    if params['num_test'] > 1:
        d1 = np.linalg.norm(params['param'][0]-params['param'][3])
        d2 = np.linalg.norm(params['param'][1]-params['param'][2])
        d = max(d1,d2) # max distance in the parameter space
        
        # find parameters that are within local region of existing training parameters
        params_exclude = []
        all_params = np.stack(test_data['param'])
        for p in params['param']:
            params_exclude.append(np.where(np.linalg.norm(all_params-p, axis=1) <= d*ld)[0].tolist())
        params_exclude = [x for xs in params_exclude for x in xs]
        params_exclude = set(params_exclude)
        params_exclude_unique = list(params_exclude)
        a = np.setdiff1d(np.arange(params['num_test']),params_exclude_unique) # exclude existing training cases

        if len(a) > params['subsize']:
            rng.shuffle(a)
            subset = a[:params['subsize']]
        else:
            subset = a
        print(f"  subset size: {len(subset)}, params_exclude size: {len(params_exclude_unique)}")
    else:
        subset = np.arange(params['num_test'])
        
    count = 0
    start_time = time()
    for i,a in enumerate(amp):
        for j,w in enumerate(width):
            if count in subset:
                _,_,u_sim,_,_,_,_,_,idx,_ = eval_model(test_data['data'][count], params, 
                                                       test_data['param'][count], knn=params['convex_knn'])
                wsindy_idx[i,j] = idx+1
                params['pde']['param'] = [a, w]
                err_array[i,j] = err_indicator(u_sim, params, 
                                               data=test_data['data'][count]['x'], 
                                               err_type=err_type) # residual norm of all time steps
            else:
                wsindy_idx[i,j] = -1
                err_array[i,j] = -1
            count += 1
    end_time = time()
    err_max = err_array.max()
    err_idx = np.argmax(err_array)
    print(f"  Time: {end_time-start_time:.2f} s, Case: {test_data['param'][err_idx]}, Tol: {params['tol']:.5f}, Max Error: {err_max:.6f}")
    return err_array, err_max, err_idx, test_data['param'][err_idx], wsindy_idx


def update_tol(params, training_data, err_type=1):
    """
    This function computes the error indicator and max relative errors of existing 
    training cases in order to update the tolerance threshold of the error indicator.
    inputs:
        params: dict, parameters of the gLaSDI model
        training_data: dict, training data
        err_type: int, types of error indicator. 
                 1: max relative error (if test data is available)
                 2: residual norm for 1D Burgers eqn
                 3: residual norm for 2D Burgers eqn
                 4: residual norm for time dependent heat conduction (MFEM example 16)
                 5: residual norm for radial advection (MFEM example 9)
    outputs:
        tol_new: float, updated tolerance for the error indicator
    """
    num_wsindy = params['num_wsindy']
    err1 = np.zeros(num_wsindy) # residual norm
    err2 = np.zeros(num_wsindy) # max relative error
    for i in range(num_wsindy):
        _,_,u_sim,_,_,_,_,_,idx,_ = eval_model(training_data['data'][i], params, 
                                               training_data['param'][i], knn=params['convex_knn'])
        params['pde']['param'] = [training_data['param'][i][0], training_data['param'][i][1]]
        err1[i] = err_indicator(u_sim, params, data=training_data['data'][i]['x'], err_type=err_type) # residual norm
        err2[i] = err_indicator(u_sim, params, data=training_data['data'][i]['x'], err_type=1) # max relative error

    # update tolerance of error indicator
    if params['adaptive'] == 'mean':
        tol_new = (err1 / err2).mean() * params['tol2']
    elif params['adaptive'] == 'last':
        tol_new = (err1[-1] / err2[-1]).mean() * params['tol2']
    else:
        x = err2.reshape(-1,1)
        y = err1.reshape(-1,1)
        reg = LinearRegression().fit(x, y)
        if params['adaptive'] == 'reg_mean':
            tol_new = max(0, reg.coef_[0][0] * params['tol2'] + reg.intercept_[0])
            print(reg.coef_[0][0], reg.intercept_[0])
            
        elif params['adaptive'] == 'reg_max':
            y_diff = y - reg.predict(x)
            tol_new = max(0, reg.coef_[0][0] * params['tol2'] + reg.intercept_[0] + y_diff.max())
            
        elif params['adaptive'] == 'reg_min':
            y_diff = y - reg.predict(x)
            tol_new = max(0, reg.coef_[0][0] * params['tol2'] + reg.intercept_[0] + y_diff.min())

    return tol_new, err2.max()



def print_progress(sess, i, loss, losses, train_dict):
    """
    Print loss function values to keep track of the training progress

    inputs:
        sess: the tensorflow session
        i: the training iteration
        loss: tensorflow object representing the total loss function used in training
        losses: tuple of the individual losses that make up the total loss
        train_dict: feed dictionary of training data

    outputs:
        Tuple of losses calculated on the training set
    """
    training_loss_vals = sess.run((loss,) + tuple(losses.values()), feed_dict=train_dict)
    print("Epoch %d" % i)
    print("  train loss: {0:.4e}, decoder: {1:.4e}, wsindy-x: {2:.4e}, wsindy-z: {3:.4e}, wsindy-reg: {4:.4f}".format(training_loss_vals[0][0], training_loss_vals[1][0], training_loss_vals[2][0], training_loss_vals[3][0], training_loss_vals[4][0]))
    return training_loss_vals



def create_feed_dictionary(data, params, idxs=None):
    """
    Create the feed dictionary for passing into tensorflow

    inputs:
        data: Dictionary object containing the data to be passed in. Must contain input data x,
        along the first (and possibly second) order time derivatives dx (ddx).
        params: Dictionary object containing model and training parameters. The relevant
        parameters are sequential_thresholding (which indicates whether or not
        coefficient thresholding is performed), coefficient_mask (optional if sequential
        thresholding is performed; 0/1 mask that selects the relevant coefficients in the WSINDy
        model), and learning rate (float that determines the learning rate).
        idxs: Optional array of indices that selects which examples from the dataset are passed
        into tensorflow. If None, all examples are used.

    outputs:
        feed_dict: Dictionary object containing the relevant data to pass to tensorflow
    """
    num_wsindy = params['num_wsindy']
    if idxs is None:
        idxs = np.arange(data['data'][0]['x'].shape[0])
    feed_dict = {}
    
    data_x = []
    #masked_data_dx = []
    data_dx = []
    data_t = []
    for i in range(num_wsindy):
        data_x.append(data['data'][i]['x'][idxs])
    
    feed_dict['x:0'] = np.stack(data_x, axis=1)   # [batch,num_wsindy,input_dim]
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = params['coefficient_mask']
        
    feed_dict['learning_rate:0'] = params['learning_rate']
    return feed_dict


def create_feed_dictionary2(data, params, idxs=None):
    """
    Create the feed dictionary for passing into tensorflow
    For testing set that has only 1 case!

    inputs:
        data: Dictionary object containing the data to be passed in. Must contain input data x,
        along the first (and possibly second) order time derivatives dx (ddx).
        params: Dictionary object containing model and training parameters. The relevant
        parameters are sequential_thresholding (which indicates whether or not
        coefficient thresholding is performed), coefficient_mask (optional if sequential
        thresholding is performed; 0/1 mask that selects the relevant coefficients in the WSINDy
        model), and learning rate (float that determines the learning rate).
        idxs: Optional array of indices that selects which examples from the dataset are passed
        into tensorflow. If None, all examples are used.

    outputs:
        feed_dict: Dictionary object containing the relevant data to pass to tensorflow
    """
    num_wsindy = params['num_wsindy']
    feed_dict = {}

    data_x = []    
    data_dx = []
    if 'data' in data.keys():
        if idxs == None:
            idxs = data['data'][0]['x'].shape[0]
        for i in range(num_wsindy):
            data_x.append(data['data'][0]['x'][:idxs])   # initial condition
            data_dx.append(data['data'][0]['dx'][:idxs]) # initial condition

    else:
        if idxs == None:
            idxs = data['x'].shape[0]
        for i in range(num_wsindy):
            data_x.append(data['x'][:idxs])   # initial condition
            data_dx.append(data['dx'][:idxs]) # initial condition
            
    feed_dict['x:0'] = np.stack(data_x, axis=1)   # [batch,num_wsindy,input_dim]
    feed_dict['dx:0'] = np.stack(data_dx, axis=1) # [batch,num_wsindy,input_dim]
    
    if params['sequential_thresholding']:
        feed_dict['coefficient_mask:0'] = params['coefficient_mask']
        
    feed_dict['learning_rate:0'] = params['learning_rate']
    return feed_dict


# heat map of max relative errors
def max_err_heatmap(max_err, wsindy_idx, params, p1_test, p2_test, data_path, idx_list=[], idx_param=[], 
                    xlabel='param1', ylabel='param2', label='Max. Relative Error (%)', dtype='int', scale=1):
    sns.set(font_scale=1.3)
    if dtype == 'int':
        max_err = max_err.astype(int)
        fmt1 = 'd'
    else:
        fmt1 = '.1f'
    rect = []
    for i in range(len(idx_param)):
        print(f"idx: {idx_param[i][0]}, param: {idx_param[i][1]}")
        idd = idx_param[i][0]
        rect.append(patches.Rectangle((idx_list[idd,1], idx_list[idd,0]), 1, 1, linewidth=2, edgecolor='k', facecolor='none'))
    rect2 = deepcopy(rect)
    
    if max_err.size < 100:
        fig = plt.figure(figsize=(5,5))
    else:
        fig = plt.figure(figsize=(9,9))

    fontsize = 14
    if max_err.max() >= 10:
        fontsize = 12
        max_err = max_err.astype(int)
        fmt1 = 'd'
    ax = fig.add_subplot(111)
    cbar_ax = fig.add_axes([0.99, 0.19, 0.02, 0.7])
    sns.heatmap(max_err*scale, ax=ax, square=True, 
                xticklabels=p2_test, yticklabels=p1_test, 
                annot=True, annot_kws={'size':fontsize}, fmt=fmt1, 
                cbar_ax=cbar_ax, cbar=True, cmap='vlag', robust=True, vmin=max_err.min(), vmax=max_err.max()*scale)
        
    for i in rect2:
        ax.add_patch(i)
        
    # format text labels
    fmt = '{:0.2f}'
    xticklabels = []
    for item in ax.get_xticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        xticklabels += [item]
    yticklabels = []
    for item in ax.get_yticklabels():
        item.set_text(fmt.format(float(item.get_text())))
        yticklabels += [item]
    ax.set_xticklabels(xticklabels)
    ax.set_yticklabels(yticklabels)
    ax.set_xlabel(xlabel, fontsize=24)
    ax.set_ylabel(ylabel, fontsize=24)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30) 
    
    plt.tight_layout()
    if label == 'Residual Norm':
        plt.savefig(data_path + f'heatmap_resNorm.png', bbox_inches='tight')
    else:
        plt.savefig(data_path + f'heatmap_maxRelErr_glasdi.png', bbox_inches='tight')
    plt.show()