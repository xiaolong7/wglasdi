import tensorflow.compat.v1 as tf
#tf.compat.v1.enable_eager_execution
tf.compat.v1.disable_v2_behavior()
from tensorflow.python.ops.parallel_for.gradients import jacobian, batch_jacobian
from time import time
import pickle
import numpy as np
import wsindy_utils



def full_network(params):
    """
    Define the full network architecture.
    Arguments:
        params - Dictionary object containing the parameters that specify the training.
        See README file for a description of the parameters.
    Returns:
        network - Dictionary containing the tensorflow objects that make up the network.
    """
    input_dim = params['input_dim']
    latent_dim = params['latent_dim']
    activation = params['activation']
    poly_order = params['poly_order']
    num_wsindy = params['num_wsindy'] # number of local WSINDys
    L = params['L']
    model_params = []
    if params['coeff_exist']:
        model_params = pickle.load(open(params['fig_path'] + params['save_name'] + '_params.pkl', 'rb'))['model_params']
    if 'include_sine' in params.keys():
        include_sine = params['include_sine']
    else:
        include_sine = False
    if 'include_cosine' in params.keys():
        include_cosine = params['include_cosine']
    else:
        include_cosine = False
    library_dim = params['library_dim']

    network = {}
    x = tf.placeholder(tf.float64, shape=[None, num_wsindy, input_dim], name='x')
   
    N = params['batch_size'] 
    dt = params['pde']['dt']
    t = np.linspace(0, (N-1)*dt, N)

    #compute masked_dx 
    masked_dx = []
    for i in range(num_wsindy):
        temp = wsindy_utils.apply_phi_p(t = t, inputs = x[:, i, :], L = L)
        masked_dx.append(temp)
    masked_dx = tf.stack(masked_dx, axis = 1)

    
    #compute masked_x
    masked_x = []
    for i in range(num_wsindy):
        temp = x[ L//2 : N - L //2, i, :]
        masked_x.append(temp)
    masked_x = tf.stack(masked_x, axis = 1)
    
    
    # Autoencoder
    if activation == 'linear':
        masked_z, masked_x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = linear_autoencoder(masked_x, input_dim, latent_dim, model_params)
    else:
        masked_z, masked_x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases = nonlinear_autoencoder(masked_x, input_dim, latent_dim,
                                                                                                              params['widths'],
                                                                                                              model_params, activation=activation)
    masked_z = tf.cast(masked_z, tf.float64)
    masked_x_decode = tf.cast(masked_x_decode, tf.float64)   

    
    #compute z
    z = []
    for i in range(num_wsindy):
        temp = NN(x[:, i, :], encoder_weights, encoder_biases, activation=activation )
        #temp = z[ L//2: N + 5, i, :]
        z.append(temp)
    z = tf.stack(z, axis = 1)

    #compute x_decode
    x_decode = []
    for i in range(num_wsindy):
        temp = NN(z[:, i, :], decoder_weights, decoder_biases, activation=activation)
        x_decode.append(temp)
    x_decode = tf.stack(x_decode, axis = 1)
    
    
    # compute masked_dz (masked_dx/dt * dz/dx)
    if params['diff'] == 'symb': # symbolic differentiation
        masked_dz = z_derivative(masked_x, masked_dx, encoder_weights, encoder_biases, activation=activation) # [batch,num_wsindy,latent_dim]
    elif params['diff'] == 'auto': # automatic differentiation
        dzdx_batch = batch_jacobian(z, x)
        dzdx = []
        for i in range(num_wsindy):
            dzdx.append(dzdx_batch[:,i,:,i,:]) # [batch,output_dim,input_dim]
        dzdx = tf.stack(dzdx, axis=1) # [batch,num_wsindy,output_dim]
        masked_dz = tf.matmul(dzdx, masked_dx[:,:,:,None])[:,:,:,0] # [batch,num_wsindy,latent_dim]
    

    # WSINDy
    Theta = []
    wsindy_coefficients = []
    masked_wsindy_predict = [] # [batch,num_wsindy,latent_dim]
  
    if params['sequential_thresholding']: # 1 mask for all local WSINDys
        coefficient_mask = tf.placeholder(tf.float64, shape=[library_dim,latent_dim], name='coefficient_mask')
        network['coefficient_mask'] = coefficient_mask
            

    for i in range(num_wsindy):
        Theta.append(wsindy_library_tf(z = z[:,i,:], latent_dim=latent_dim, poly_order=poly_order,L = L, t = t, include_sine=include_sine, include_cosine=include_cosine))
        
        if params['coeff_exist']:
            if i < len(model_params[0]):
                wsindy_coefficients.append(tf.get_variable(f'wsindy_coefficients{i}', initializer=model_params[0][i]))
            else:
                print(f"  Existing WSINDys: {len(model_params[0])}, Create new WSINDy: {i+1}")
        
                # initialize the new local WSINDy with the coefficients of the nearest WSINDy
                all_param = np.stack(params['param'])
                idx = np.argmin(np.linalg.norm(all_param[:-1]-all_param[-1], axis=1))

                wsindy_coefficients.append(tf.get_variable(f'wsindy_coefficients{i}', initializer=model_params[0][idx]))  
        else:
            if params['coefficient_initialization'] == 'xavier':
                wsindy_coefficients.append(tf.get_variable(f'wsindy_coefficients{i}', shape=[library_dim,latent_dim], 
                                                          initializer=tf.truncated_normal_initializer()))
            elif params['coefficient_initialization'] == 'specified':
                wsindy_coefficients.append(tf.get_variable(f'wsindy_coefficients{i}', 
                                                          initializer=params['init_coefficients']))
            elif params['coefficient_initialization'] == 'constant':
                wsindy_coefficients.append(tf.get_variable(f'wsindy_coefficients{i}', shape=[library_dim,latent_dim], 
                                                          initializer=tf.constant_initializer(1.0)))
            elif params['coefficient_initialization'] == 'normal':
                wsindy_coefficients.append(tf.get_variable(f'wsindy_coefficients{i}', shape=[library_dim,latent_dim],
                                                          initializer=tf.initializers.random_normal()))
                
        wsindy_coefficients[i] = tf.cast(wsindy_coefficients[i], tf.float64) 
   
        if params['sequential_thresholding']:
            masked_wsindy_predict.append(tf.matmul(Theta[i], coefficient_mask * wsindy_coefficients[i]))
        else:
            masked_wsindy_predict.append(tf.matmul(Theta[i], wsindy_coefficients[i])) # [batch,input_dim]
                
    masked_wsindy_predict = tf.stack(masked_wsindy_predict, axis=1) # [batch,num_wsindy,latent_dim]

    
    # compute masked dx/dt (masked_dz/dt * dx/dz)
    if params['loss_weight_wsindy_x'] > 0:
        if params['diff'] == 'symb': # symbolic differentiation
            masked_dx_decode = z_derivative(masked_z, masked_wsindy_predict, decoder_weights, decoder_biases, activation=activation) # [batch,num_wsindy,input_dim]
        elif params['diff'] == 'auto': # automatic differentiation
            dxdz_batch = batch_jacobian(masked_x_decode, masked_z)
            dxdz = []
            for i in range(num_wsindy):
                dxdz.append(dxdz_batch[:,i,:,i,:]) # [batch,output_dim,input_dim]
            dxdz = tf.stack(dxdz, axis=1) # [batch,num_wsindy,output_dim]
            masked_dx_decode = tf.linalg.matmul(dxdz, masked_wsindy_predict[:,:,:,None])[:,:,:,0] # [batch,num_wsindy,input_dim]
    else:
        masked_dx_decode = tf.cast(tf.zeros(tf.shape(masked_dx)), tf.float64)

         
    network['x'] = x   # [batch,num_wsindy,input_dim]
    network['masked_dx'] = masked_dx # [batch,num_wsindy,input_dim]
    network['z'] = z   # [batch,num_wsindy,latent_dim]
    network['masked_dz'] = masked_dz # [batch,num_wsindy,latent_dim]
    network['x_decode'] = x_decode   # [batch,num_wsindy,input_dim]
    network['masked_dx_decode'] = masked_dx_decode # [batch,num_wsindy,input_dim]
    network['encoder_weights'] = encoder_weights
    network['encoder_biases'] = encoder_biases
    network['decoder_weights'] = decoder_weights
    network['decoder_biases'] = decoder_biases
    network['Theta'] = Theta # list
    network['wsindy_coefficients'] = wsindy_coefficients # list
    network['masked_dz_predict'] = masked_wsindy_predict # [batch,num_wsindy,latent_dim]
    return network

    
def define_loss(network, params):
    """
    Create the loss function.
    Arguments:
        network - Dictionary object containing the elements of the network architecture.
        This will be the output of the full_network() function.
    """
    x = network['x']                # [batch,num_wsindy,input_dim]
    x_decode = network['x_decode']  # [batch,num_wsindy,input_dim]
    num_wsindy = params['num_wsindy'] # number of local WSINDys
    wsindy_coefficients = network['wsindy_coefficients']
    masked_dz = network['masked_dz']                 # [batch,num_wsindy,latent_dim]
    masked_dz_predict = network['masked_dz_predict'] # [batch,num_wsindy,latent_dim]
    masked_dx = network['masked_dx']                 # [batch,num_wsindy,input_dim]
    masked_dx_decode = network['masked_dx_decode']   # [batch,num_wsindy,input_dim]

    losses = {}
    losses['decoder'] = tf.cast(tf.zeros((1)), tf.float64)
    losses['wsindy_x'] = tf.cast(tf.zeros((1)), tf.float64)
    losses['wsindy_z'] = tf.cast(tf.zeros((1)), tf.float64)
    losses['wsindy_regularization'] = tf.cast(tf.zeros((1)), tf.float64)
    
    for i in range(num_wsindy):
        losses['decoder'] += tf.reduce_mean((x[:,i,:] - x_decode[:,i,:])**2)   #L_recon
        losses['wsindy_x'] += tf.reduce_mean((masked_dx[:,i,:] - masked_dx_decode[:,i,:])**2) #L_u_dot
        losses['wsindy_z'] += tf.reduce_mean((masked_dz[:,i,:] - masked_dz_predict[:,i,:])**2)#L_z_dot
        losses['wsindy_regularization'] += tf.reduce_mean(tf.abs(wsindy_coefficients[i]))
        
    loss = params['loss_weight_decoder'] * losses['decoder'] \
           + params['loss_weight_wsindy_z'] * losses['wsindy_z'] \
           + params['loss_weight_wsindy_x'] * losses['wsindy_x'] \
           + params['loss_weight_wsindy_regularization'] * losses['wsindy_regularization']

    loss_refinement = params['loss_weight_decoder'] * losses['decoder']  \
                      + params['loss_weight_wsindy_z'] * losses['wsindy_z'] \
                      + params['loss_weight_wsindy_x'] * losses['wsindy_x']
    return loss, losses, loss_refinement


def linear_autoencoder(x, input_dim, latent_dim, model_params):
    """
    Construct a linear autoencoder.
    Arguments:
        x - 2D tensorflow array, input to the network (shape is [batch,num_wsindy,input_dim])
        input_dim - Integer, number of state variables in the input to the first layer
        latent_dim - Integer, number of latent variables in the embedding layer
        model_params - List, exsiting model parameters
        
    Returns:
        z - Tensorflow array, output of the embedding layer (shape is [batch,num_wsindy,latent_dim])
        x_decode - Tensorflow array, output of the output layer (shape is [batch,num_wsindy,output_dim])
        encoder_weights - List of tensorflow arrays containing the encoder weights
        encoder_biases - List of tensorflow arrays containing the encoder biases
        decoder_weights - List of tensorflow arrays containing the decoder weights
        decoder_biases - List of tensorflow arrays containing the decoder biases
    """
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, [], None, 'encoder', model_params)
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, [], None, 'decoder', model_params)
    return z, x_decode, encoder_weights, encoder_biases,decoder_weights,decoder_biases


def silu(x):
    """
    This function calculates the output of the SiLU activation 
    given an input x.
    """
    return x * tf.sigmoid(x)

def linear_tanh(x):
    condition1 = tf.cast(x < -1, tf.float64)
    condition2 = tf.cast(tf.logical_and(x >= -1, x <= 1), tf.float64)
    condition3 = tf.cast(x > 1, tf.float64)
    return condition1 * (-1) + condition2 * x + condition3 * 1

#def linear_sigmoid(x):
    #condition1 = tf.cast(x < -3, tf.float64)
    #condition2 = tf.cast(tf.logical_and(x >= -3, x <= 3), tf.float64)
    #condition3 = tf.cast(x > 3, tf.float64)
    #return condition1 * (0) + condition2 * (1/6*x + 1/2)  + condition3 * 1

def linear_sigmoid(x):
    cond1 = tf.cast(x < -5, tf.float64)
    cond2 = tf.cast(tf.logical_and(x >= -5, x < -4), tf.float64)
    cond3 = tf.cast(tf.logical_and(x >= -4, x < -3), tf.float64)
    cond4 = tf.cast(tf.logical_and(x >= -3, x < -2), tf.float64)
    cond5 = tf.cast(tf.logical_and(x >= -2, x < -1), tf.float64)
    cond6 = tf.cast(tf.logical_and(x >= -1, x < 0), tf.float64)
    cond7 = tf.cast(tf.logical_and(x >= 0, x < 1), tf.float64)
    cond8 = tf.cast(tf.logical_and(x >= 1, x < 2), tf.float64)
    cond9 = tf.cast(tf.logical_and(x >= 2, x < 3), tf.float64)
    cond10 = tf.cast(tf.logical_and(x >= 3, x < 4), tf.float64)
    cond11 = tf.cast(tf.logical_and(x >= 4, x < 5), tf.float64)
    cond12 = tf.cast(x >= 5, tf.float64)

    def line_sigmoid(x, x1, x2):
        x1 = tf.cast(x1, tf.float64)
        x2 = tf.cast(x2, tf.float64)
        y1 = tf.sigmoid(x1)
        y2 = tf.sigmoid(x2)
        slope = (y1 - y2)/(x1 - x2)
        intercept = y1 - slope*x1
        return slope*x + intercept
    
    f = cond1*0 + cond2*line_sigmoid(x, -5, -4) + cond3*line_sigmoid(x, -4, -3) + cond4*line_sigmoid(x, -3, -2) + cond5*line_sigmoid(x, -2, -1) + cond6*line_sigmoid(x, -1, 0) + cond7*line_sigmoid(x, 0, 1) + cond8*line_sigmoid(x, 1, 2) + cond9*line_sigmoid(x, 2, 3) + cond10*line_sigmoid(x, 3, 4) + cond11*line_sigmoid(x, 4, 5) + cond12*1
    return f

def derivative_sigmoid(x):

    def slope_sigmoid(x1, x2):
        x1 = tf.cast(x1, tf.float64)
        x2 = tf.cast(x2, tf.float64)
        y1 = tf.sigmoid(x1)
        y2 = tf.sigmoid(x2)
        slope = (y1 - y2)/(x1 - x2)
        return slope
    
    cond1 = tf.cast(x < -5, tf.float64)
    cond2 = tf.cast(tf.logical_and(x >= -5, x < -4), tf.float64)
    cond3 = tf.cast(tf.logical_and(x >= -4, x < -3), tf.float64)
    cond4 = tf.cast(tf.logical_and(x >= -3, x < -2), tf.float64)
    cond5 = tf.cast(tf.logical_and(x >= -2, x < -1), tf.float64)
    cond6 = tf.cast(tf.logical_and(x >= -1, x < 0), tf.float64)
    cond7 = tf.cast(tf.logical_and(x >= 0, x < 1), tf.float64)
    cond8 = tf.cast(tf.logical_and(x >= 1, x < 2), tf.float64)
    cond9 = tf.cast(tf.logical_and(x >= 2, x < 3), tf.float64)
    cond10 = tf.cast(tf.logical_and(x >= 3, x < 4), tf.float64)
    cond11 = tf.cast(tf.logical_and(x >= 4, x < 5), tf.float64)
    cond12 = tf.cast(x >= 5, tf.float64)
    f = cond1*0 + cond2*slope_sigmoid(-5, -4) + cond3*slope_sigmoid( -4, -3) + cond4*slope_sigmoid( -3, -2) + cond5*slope_sigmoid( -2, -1) + cond6*slope_sigmoid( -1, 0) + cond7*slope_sigmoid(0, 1) + cond8*slope_sigmoid(1, 2) + cond9*slope_sigmoid(2, 3) + cond10*slope_sigmoid(3, 4) + cond11*slope_sigmoid(4, 5) + cond12*0
    return f

  
def nonlinear_autoencoder(x, input_dim, latent_dim, widths, model_params, activation='elu'):
    """
    Construct a nonlinear autoencoder.
    Arguments:
        x - 2D tensorflow array, input to the network (shape is [batch,num_wsindy,input_dim])
        input_dim - Integer, number of state variables in the input to the first layer
        latent_dim - Integer, number of latent variables in the embedding layer
        widths - List of integers representing how many units are in each network layer
        model_params - List, exsiting model parameters
        activation - Tensorflow function to be used as the activation function at each layer
        
    Returns:
        z - Tensorflow array, output of the embedding layer (shape is [batch,num_wsindy,latent_dim])
        x_decode - Tensorflow array, output of the output layer (shape is [batch,num_wsindy,output_dim])
        encoder_weights - List of tensorflow arrays containing the encoder weights
        encoder_biases - List of tensorflow arrays containing the encoder biases
        decoder_weights - List of tensorflow arrays containing the decoder weights
        decoder_biases - List of tensorflow arrays containing the decoder biases
    """
    if activation == 'relu':
        activation_function = tf.nn.relu
    elif activation == 'elu':
        activation_function = tf.nn.elu
    elif activation == 'sigmoid':
        activation_function = tf.sigmoid
    elif activation == 'tanh':
        activation_function = tf.tanh
    elif activation == 'silu':
        activation_function = silu
    elif activation == 'linear_tanh':
        activation_function = linear_tanh
    elif activation == 'linear_sigmoid':
        activation_function = linear_sigmoid
    elif activation == 'softplus':
        activation_function = tf.math.softplus
    else:
        raise ValueError('invalid activation function')
    z,encoder_weights,encoder_biases = build_network_layers(x, input_dim, latent_dim, widths, activation_function, 'encoder', model_params)
    x_decode,decoder_weights,decoder_biases = build_network_layers(z, latent_dim, input_dim, widths[::-1], activation_function, 'decoder', model_params)
    return z, x_decode, encoder_weights, encoder_biases, decoder_weights, decoder_biases


def build_network_layers(input, input_dim, output_dim, widths, activation, name, model_params):
    """
    Construct one portion of the network (either encoder or decoder).
    Arguments:
        input - 2D tensorflow array, input to the network (shape is [batch,num_wsindy,input_dim])
        input_dim - Integer, number of state variables in the input to the first layer
        output_dim - Integer, number of state variables to output from the final layer
        widths - List of integers representing how many units are in each network layer
        activation - Tensorflow function to be used as the activation function at each layer
        name - String, prefix to be used in naming the tensorflow variables
        model_params - List, exsiting model parameters
    Returns:
        input - Tensorflow array, output of the network layers (shape is [batch,num_wsindy,output_dim])
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
    """
    weights = []
    biases = []
    last_width = input_dim # output width of last (previous) layer
    
    if len(model_params) > 0:
        # build hidden layers
        for i,n_units in enumerate(widths):
            if name == 'encoder':
                W = tf.get_variable(name+'_W'+str(i), initializer=model_params[1][i])
                b = tf.get_variable(name+'_b'+str(i), initializer=model_params[2][i])
            elif name == 'decoder':
                W = tf.get_variable(name+'_W'+str(i), initializer=model_params[3][i])
                b = tf.get_variable(name+'_b'+str(i), initializer=model_params[4][i])
            last_width = n_units
            weights.append(W)
            biases.append(b)

        # build last layer
        if name == 'encoder':
            W = tf.get_variable(name+'_W'+str(len(widths)), initializer=model_params[1][-1])
            b = tf.get_variable(name+'_b'+str(len(widths)), initializer=model_params[2][-1])
        elif name == 'decoder':
            W = tf.get_variable(name+'_W'+str(len(widths)), initializer=model_params[3][-1])
            b = tf.get_variable(name+'_b'+str(len(widths)), initializer=model_params[4][-1])
        weights.append(W)
        biases.append(b)
    
    else:
        # build hidden layers
        for i,n_units in enumerate(widths):
            W = tf.get_variable(name+'_W'+str(i), shape=[last_width,n_units])
            b = tf.get_variable(name+'_b'+str(i), shape=[n_units], initializer=tf.constant_initializer(0.0))
            last_width = n_units
            weights.append(W)
            biases.append(b)
        
        # build last layer
        W = tf.get_variable(name+'_W'+str(len(widths)), shape=[last_width,output_dim])
        b = tf.get_variable(name+'_b'+str(len(widths)), shape=[output_dim],initializer=tf.constant_initializer(0.0))
        weights.append(W)
        biases.append(b)

    
    weights = [tf.cast(tensor, tf.float64) for tensor in weights]
    biases = [tf.cast(tensor, tf.float64) for tensor in biases]

    # forward pass
    output = []
    for j in range(input.shape[1]):
        input_j = input[:,j,:]
        for i in range(len(weights)-1):
            input_j = tf.matmul(input_j, weights[i]) + biases[i]
            #input_j = tf.matmul(input_j, tf.cast(weights[i], tf.float64)) + tf.cast(biases[i], tf.float64)
            if activation is not None:
                input_j = activation(input_j)
        output.append(tf.matmul(input_j, weights[-1]) + biases[-1]) # last layer, [batch,output_dim]
        #output.append(tf.matmul(input_j, tf.cast(weights[-1], tf.float64)) + tf.cast(biases[-1], tf.float64)) # last layer, [batch,output_dim]
    output = tf.stack(output, axis=1) # [batch,num_wsindy,output_dim]
    return output, weights, biases


def wsindy_library_tf(z, latent_dim, poly_order, L, t, include_sine=False, include_cosine=False):
    """
    Build the WSINDy library.
    Arguments:
        z - 2D tensorflow array of the snapshots on which to build the library. Shape is number of
        time points by the number of state variables.
        latent_dim - Integer, number of state variable in z.
        poly_order - Integer, polynomial order to which to build the library. Max value is 5.
        include_sine - Boolean, whether or not to include sine terms in the library. Default False.
    Returns:
        2D tensorflow array containing the constructed library. Shape is number of time points by
        number of library functions. The number of library functions is determined by the number
        of state variables of the input, the polynomial order, and whether or not sines are included.
    """
    
    library = [tf.ones(tf.shape(z)[0], tf.float64)]
    
    if poly_order > 0:
        for i in range(latent_dim):
            library.append(z[:,i])
    
    if poly_order > 1:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                library.append(tf.multiply(z[:,i], z[:,j]))
    
    if poly_order > 2:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    library.append(z[:,i]*z[:,j]*z[:,k])

    if poly_order > 3:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p])

    if poly_order > 4:
        for i in range(latent_dim):
            for j in range(i,latent_dim):
                for k in range(j,latent_dim):
                    for p in range(k,latent_dim):
                        for q in range(p,latent_dim):
                            library.append(z[:,i]*z[:,j]*z[:,k]*z[:,p]*z[:,q])

    if include_sine:
        for i in range(latent_dim):
            library.append(tf.sin(z[:,i]))
            
    if include_cosine:
        for i in range(latent_dim):
            library.append(tf.cos(z[:,i]))

    Theta = tf.stack(library, axis=1)
    Theta = wsindy_utils.apply_phi( inputs=Theta, L = L, t = t)
    return Theta


def z_derivative(input, dx, weights, biases, activation='elu'):
    """
    Compute the first order time derivatives by propagating through the network.
    Arguments:
        input - 2D tensorflow array, input to the network. Dimensions are number of time points
        by number of state variables.
        dx - First order time derivatives of the input to the network.
        weights - List of tensorflow arrays containing the network weights
        biases - List of tensorflow arrays containing the network biases
        activation - String specifying which activation function to use. Options are
        'elu' (exponential linear unit), 'relu' (rectified linear unit), 'sigmoid',
        or linear.
    Returns:
        dz - Tensorflow array, first order time derivatives of the network output.
    """
    num_wsindy = input.shape[1]
    dz = []

    weights = [tf.cast(tensor, tf.float64) for tensor in weights]
    biases = [tf.cast(tensor, tf.float64) for tensor in biases]
    input = tf.cast(input, tf.float64)
    if activation == 'elu':
        for j in range(num_wsindy):
            input_j = input[:,j,:]
            dz_j = dx[:,j,:]
            for i in range(len(weights)-1):
                input_j = tf.matmul(input_j, weights[i]) + biases[i]
                dz_j = tf.multiply(tf.minimum(tf.exp(input_j),1.0),
                                      tf.matmul(dz_j, weights[i]))
                input_j = tf.nn.elu(input_j)
            dz.append(tf.matmul(dz_j, weights[-1])) # [batch,output_dim]
        dz = tf.stack(dz, axis=1) # [batch,num_wsindy,output_dim]
        
    elif activation == 'relu':
        for j in range(num_wsindy):
            input_j = input[:,j,:]
            dz_j = dx[:,j,:]
            for i in range(len(weights)-1):
                input_j = tf.matmul(input_j, weights[i]) + biases[i]
                dz_j = tf.multiply(tf.cast(tf.to_float(input_j > 0), tf.float64), tf.matmul(dz_j, weights[i]))
                input_j = tf.nn.relu(input_j)
            dz.append(tf.matmul(dz_j, weights[-1])) # [batch,output_dim]
        dz = tf.stack(dz, axis=1) # [batch,num_wsindy,output_dim]
        
    elif activation == 'sigmoid':
        for j in range(num_wsindy):
            input_j = input[:,j,:]
            dz_j = dx[:,j,:]
            for i in range(len(weights)-1):
                input_j = tf.matmul(input_j, weights[i]) + biases[i]
                input_j = tf.sigmoid(input_j)
                dz_j = tf.multiply(tf.multiply(input_j, 1-input_j), tf.matmul(dz_j, weights[i]))
            dz.append(tf.matmul(dz_j, weights[-1])) # [batch,output_dim]
        dz = tf.stack(dz, axis=1) # [batch,num_wsindy,output_dim]
     
    elif activation == 'softplus':
        for j in range(num_wsindy):
            input_j = input[:,j,:]
            dz_j = dx[:,j,:]
            for i in range(len(weights)-1):
                input_j = tf.matmul(input_j, weights[i]) + biases[i]
                dz_j = tf.multiply(tf.sigmoid(input_j), tf.matmul(dz_j, weights[i]))
                input_j = tf.math.softplus(input_j)
            dz.append(tf.matmul(dz_j, weights[-1])) # [batch,output_dim]
        dz = tf.stack(dz, axis=1) # [batch,num_wsindy,output_dim]
        
    elif activation == 'linear_tanh':
        for j in range(num_wsindy):
            input_j = input[:,j,:]
            dz_j = dx[:,j,:]
            for i in range(len(weights)-1):
                input_j = tf.matmul(input_j, weights[i]) + biases[i]
                dz_j = tf.multiply(tf.cast(tf.to_float(tf.logical_and(input_j > -1, input_j <= 1)), tf.float64), tf.matmul(dz_j, weights[i]))
                input_j = linear_tanh(input_j)
            dz.append(tf.matmul(dz_j, weights[-1])) # [batch,output_dim]
        dz = tf.stack(dz, axis=1) # [batch,num_wsindy,output_dim]
        
    elif activation == 'linear_sigmoid':
        for j in range(num_wsindy):
            input_j = input[:,j,:]
            dz_j = dx[:,j,:]
            for i in range(len(weights)-1):
                input_j = tf.matmul(input_j, weights[i]) + biases[i]
                #dz_j = tf.multiply(tf.cast(1/6*(tf.to_float(tf.logical_and(input_j > -3, input_j <= 3))), tf.float64), tf.matmul(dz_j, weights[i]))
                dz_j = tf.multiply(tf.cast(derivative_sigmoid(input_j), tf.float64), tf.matmul(dz_j, weights[i]))
                input_j = linear_sigmoid(input_j)
            dz.append(tf.matmul(dz_j, weights[-1])) # [batch,output_dim]
        dz = tf.stack(dz, axis=1) # [batch,num_wsindy,output_dim]
    else:
        for j in range(num_wsindy):
            input_j = input[:,j,:]
            dz_j = dx[:,j,:]
            for i in range(len(weights)-1):
                input_j = tf.matmul(input_j, weights[i]) + biases[i]
            dz.append(tf.matmul(dz_j, weights[-1])) # [batch,output_dim]
        dz = tf.stack(dz, axis=1) # [batch,num_wsindy,output_dim]
    return dz


def NN(x, weights, biases, activation):
    """
    This network serves as either an encoder or a decoder, 
    where the output layer has a linear activation.
    """
    num_layers = len(weights)
    for i in range(num_layers-1):
        x = tf.matmul(x, weights[i]) + biases[i]
        if activation == 'tanh':
            x = tf.tanh(x)
        elif activation == 'sigmoid':
            x = tf.sigmoid(x)
        elif activation == 'relu':
            x = tf.nn.relu(x)
        elif activation == 'linear_tanh':
            x = linear_tanh(x)
        elif activation == 'linear_sigmoid':
            x = linear_sigmoid(x)
        elif activation == 'softplus':
            x = tf.math.softplus(x)
            
    # output layer (linear activation)
    x = tf.matmul(x, weights[-1]) + biases[-1]
    return x
