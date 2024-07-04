import tensorflow as tf
from tensorflow import keras
import numpy as np
from scipy.special import binom
tf.keras.backend.set_floatx('float64')
from scipy.integrate import odeint


def apply_phi(t, inputs, L = 30, param = [0, np.inf, 0]):
    V, Vp = Uniform_grid(t, L, param)
    G = tf.matmul(V, inputs)
    return G

def apply_phi_p(t, inputs, L = 30, param = [0, np.inf, 0]):
    V, Vp = Uniform_grid(t, L, param)
    b = tf.matmul(Vp, inputs)
    return b


def apply_phi_tf(inputs, delta_t = 0.001, L = 30):
    """
    Apply the test function convolution layer onto the input data. 

    inputs:
        L: Test function support (where it's nonzero)
        s: Test function overlap parameter, ie. how much overlap 2 consecutive test functions have. 
        s is ranging from 0 to 1. 
        delta_t: time step
        inputs: input layer (Tensor) shape = (N,)
    outputs:
        outputs: output layer (Tensor)
    """
    inputs = tf.cast(inputs, tf.float64)
    p = 16.0

    #define test_function
    t = tf.cast(tf.linspace(0.0, delta_t*(L - 1), L), tf.float64)
    a = 0.0
    b = delta_t*(L - 1)
    phi = tf.cast(tf.multiply(tf.cast(tf.pow(t - a, p), tf.float64), tf.cast(tf.pow(b - t, p), tf.float64)), tf.float64)
    #norm = tf.reduce_max(tf.abs(phi))
    #phi = phi/norm
    phi = tf.reshape(phi, (1, L, 1, 1))

    # Expand dimensions of inputs and kernel
    inputs = tf.expand_dims(inputs, axis=-1)
    inputs = tf.expand_dims(inputs, axis=0)

    # Create a Conv1D layer and assign the test function kernel
    conv_layer = keras.layers.Conv2D(filters=1, kernel_size=(1, L), padding='valid', use_bias=False)
    conv_layer.build(input_shape=(None, None, None, 1))
    conv_layer.kernel.assign(phi)

    # Apply the Conv2D layer to the inputs
    outputs = conv_layer(inputs)

    # Remove the extra dimensions
    outputs = tf.squeeze(outputs, axis=[0, -1])

    return outputs #outputs

def apply_gaussian_p(inputs, L):
    inputs = tf.cast(inputs, tf.float64)

    # Create a custom Gaussian moving average kernel
    kernel_size = L
    sigma = 1.0
    x = tf.range(-(kernel_size // 2), kernel_size // 2 + 1, dtype=tf.float64)
    gaussian_kernel = tf.exp(-(x ** 2) / (2 * sigma ** 2))
    #norm = tf.reduce_sum(gaussian_kernel)
    gaussian_kernel = tf.multiply(gaussian_kernel, x)/(sigma**2)
    gaussian_kernel = tf.reshape(gaussian_kernel, (1, kernel_size, 1, 1))


    # Expand dimensions of inputs and kernel
    inputs = tf.expand_dims(inputs, axis=-1)
    inputs = tf.expand_dims(inputs, axis=0)

    # Create a Conv2D layer and assign the Gaussian kernel
    conv_layer = keras.layers.Conv2D(filters=1, kernel_size=(1, kernel_size), padding='valid', use_bias=False)
    conv_layer.build(input_shape=(None, None, None, 1))
    conv_layer.kernel.assign(gaussian_kernel)

    # Apply the Conv2D layer to the inputs
    outputs = conv_layer(inputs)

    # Remove the extra dimensions
    outputs = tf.squeeze(outputs, axis=[0, -1])

    return outputs

def apply_gaussian(inputs, L):
    inputs = tf.cast(inputs, tf.float64)

    # Create a custom Gaussian moving average kernel
    kernel_size = L
    sigma = 1.0
    x = tf.range(-(kernel_size // 2), kernel_size // 2 + 1, dtype=tf.float64)
    gaussian_kernel = tf.exp(-(x ** 2) / (2 * sigma ** 2))
    #gaussian_kernel = gaussian_kernel / tf.reduce_sum(gaussian_kernel)
    gaussian_kernel = tf.reshape(gaussian_kernel, (1, kernel_size, 1, 1))

    # Expand dimensions of inputs and kernel
    inputs = tf.expand_dims(inputs, axis=-1)
    inputs = tf.expand_dims(inputs, axis=0)

    # Create a Conv2D layer and assign the Gaussian kernel
    conv_layer = keras.layers.Conv2D(filters=1, kernel_size=(1, kernel_size), padding='valid', use_bias=False)
    conv_layer.build(input_shape=(None, None, None, 1))
    conv_layer.kernel.assign(gaussian_kernel)

    # Apply the Conv2D layer to the inputs
    outputs = conv_layer(inputs)

    # Remove the extra dimensions
    outputs = tf.squeeze(outputs, axis=[0, -1])

    return outputs


def apply_phi_p_tf(inputs, delta_t = 0.001, L = 31):
    """
    Apply the derivative of test function convolution layer onto the input data. 

    inputs:
        L: Test function support (where it's nonzero)
        s: Test function overlap parameter, ie. how much overlap 2 consecutive test functions have. 
        s is ranging from 0 to 1. 
        delta_t: time step
        inputs: input tensor
    outputs:
        outputs: output tensor
    """
    inputs = tf.cast(inputs, tf.float64)
    p = 16.0
    
    #define test_function
    t = tf.cast(tf.linspace(0.0, delta_t*(L - 1), L), tf.float64)
    a = 0.0
    b = delta_t*(L - 1)
    phi = tf.cast(tf.multiply(tf.cast(tf.pow(t - a, p), tf.float64), tf.cast(tf.pow(b - t, p), tf.float64)), tf.float64)
    #norm = tf.reduce_max(tf.abs(phi))
    #norm = 1/(p**p*p**p)*((2*p)/(b - a))**(2*p)
    #phi = tf.cast(tf.multiply(tf.cast(tf.pow(t - a, p - 1), tf.float64), tf.cast(tf.pow(b - t, p - 1), tf.float64)*p*(a + b - 2*t), tf.float64), tf.float64)
    phi = tf.cast(
    tf.multiply(
        tf.cast(tf.pow(t - a, p - 1), tf.float64),
        tf.cast(tf.pow(b - t, p - 1), tf.float64) * p * (a + b - 2 * t)
    ),
    tf.float64)
    phi = -phi #/norm
    phi = tf.reshape(phi, (1, L, 1, 1))

    # Expand dimensions of inputs and kernel
    inputs = tf.expand_dims(inputs, axis=-1)
    inputs = tf.expand_dims(inputs, axis=0)


    # Create a Conv2D layer and assign the test function kernel
    conv_layer = keras.layers.Conv2D(filters=1, kernel_size=(1, L), strides=1, padding='valid', use_bias=False)
    conv_layer.build(input_shape=(None, None, None, 1))
    #expanded_kernel = tf.expand_dims(phi, axis=-1)
    #expanded_kernel = tf.expand_dims(expanded_kernel, axis=0)
    #expanded_kernel = tf.reshape(expanded_kernel, conv_layer.kernel.shape)
    conv_layer.kernel.assign(phi)

    #apply the 

    #N = tf.shape(inputs)[0]
    #inputs = tf.reshape(inputs, (1, N, 1))
    #print(inputs)
    #outputs = conv_layer(inputs)

    #input_size = tf.shape(inputs)[1]
    #output_size = (input_size - 1) // 1 + 1
    #desired_output_size = output_size
    #padding_needed = desired_output_size - tf.shape(outputs)[1]
    #padding_left = padding_needed // 2
    #padding_right = padding_needed - padding_left
    #outputs_padded = tf.pad(outputs, [[0, 0], [padding_left, padding_right], [0, 0]])

   # Apply the Conv2D layer to the inputs
    outputs = conv_layer(inputs)

    # Remove the extra dimensions
    outputs = tf.squeeze(outputs, axis=[0, -1])

    return outputs

def wsindy_simulate(x0, t, Xi, poly_order, include_sine, include_cosine=False):
    m = t.size
    n = x0.size
    f = lambda x,t : np.dot(sindy_library(np.array(x).reshape((1,n)), poly_order, include_sine, include_cosine), Xi).reshape((n,))

    x = odeint(f, x0, t)
    return x

def library_size(n, poly_order, use_sine=False, use_cosine=False, include_constant=True):
    l = 0
    for k in range(poly_order+1):
        l += int(binom(n+k-1,k))
    if use_sine:
        l += n
    if use_cosine:
        l += n
    if not include_constant:
        l -= 1
    return l

def sindy_library(X, poly_order, include_sine=False, include_cosine=False):
    m,n = X.shape
    l = library_size(n, poly_order, include_sine, include_cosine, True)
    library = np.ones((m,l))
    index = 1

    if poly_order > 0: 
        for i in range(n):
            library[:,index] = X[:,i]
            index += 1
        
    if poly_order > 1:
        for i in range(n):
            for j in range(i,n):
                library[:,index] = X[:,i]*X[:,j]
                index += 1

    if poly_order > 2:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    library[:,index] = X[:,i]*X[:,j]*X[:,k]
                    index += 1

    if poly_order > 3:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]
                        index += 1
                    
    if poly_order > 4:
        for i in range(n):
            for j in range(i,n):
                for k in range(j,n):
                    for q in range(k,n):
                        for r in range(q,n):
                            library[:,index] = X[:,i]*X[:,j]*X[:,k]*X[:,q]*X[:,r]
                            index += 1

    if include_sine:
        for i in range(n):
            library[:,index] = np.sin(X[:,i])
            index += 1
            
    if include_cosine:
        for i in range(n):
            library[:,index] = np.cos(X[:,i])
            index += 1
            
    return library

def basis_fcn(p, q):
    def g(t, t1, tk): return (p > 0)*(q > 0)*(t - t1)**max(p, 0)*(tk - t)**max(q, 0) + (p == 0)*(q == 0)*(1 - 2 *
                                                                                                        np.abs(t - (t1+tk)/2)/(tk - t1)) + (p > 0)*(q < 0)*np.sin(p*np.pi/(tk - t1)*(t - t1)) + (p == -1)*(q == -1)

    def gp(t, t1, tk): return (t-t1)**(max(p-1, 0))*(tk-t)**(max(q-1, 0))*((-p-q)*t+p*tk+q*t1)*(q > 0)*(p > 0) + -2*np.sign(t -
                                                                                                                            (t1+tk)/2)/(tk-t1)*(q == 0)*(p == 0) + p*np.pi/(tk-t1)*np.cos(p*np.pi/(tk-t1)*(t-t1))*(q < 0)*(p > 0) + 0*(p == -1)*(q == -1)

    if p > 0 and q > 0:
        def normalize(t, t1, tk): return (
            t - t1)**max(p, 0)*(tk - t)**max(q, 0)

        def g(t, t1, tk): return ((p > 0)*(q > 0)*(t - t1)**max(p, 0)*(tk - t)**max(q, 0) + (p == 0)*(q == 0)*(1 - 2*np.abs(t - (t1+tk)/2) /
                                                                                                            (tk - t1)) + (p > 0)*(q < 0)*np.sin(p*np.pi/(tk - t1)*(t - t1)) + (p == -1)*(q == -1))/(np.abs(normalize((q*t1+p*tk)/(p+q), t1, tk)))

        def gp(t, t1, tk): return ((t-t1)**(max(p-1, 0))*(tk-t)**(max(q-1, 0))*((-p-q)*t+p*tk+q*t1)*(q > 0)*(p > 0) + -2*np.sign(t-(t1+tk)/2)/(tk-t1)*(q == 0)
                                * (p == 0) + p*np.pi/(tk-t1)*np.cos(p*np.pi/(tk-t1)*(t-t1))*(q < 0)*(p > 0) + 0*(p == -1)*(q == -1))/(np.abs(normalize((q*t1+p*tk)/(p+q), t1, tk)))
    return g, gp

def tf_mat_row(g, gp, t, t1, tk, param):
    N = len(t)
    #N = tf.shape(t)[0]
    #sess = tf.compat.v1.Session()
    #N = N.eval(session=sess)

    if param == None:
        pow = 1
        gap = 1
        nrm = np.inf
        ord = 0
    else:
        pow = param[0]
        nrm = param[1]
        ord = param[2]
        gap = 1

    if t1 > tk:
        tk_temp = tk
        tk = t1
        t1 = tk_temp

    V_row = np.zeros((1, N))
    Vp_row = np.copy(V_row)

    t_grid = t[t1:tk+1:gap]
    dts = np.diff(t_grid)
    w = 1/2*(np.append(dts, [0]) + np.append([0], dts))

    V_row[:, t1:tk+1:gap] = (g(t_grid, t[t1], t[tk])*w).T
    Vp_row[:, t1:tk+1:gap] = (-gp(t_grid, t[t1], t[tk])*w).T
    Vp_row[:, t1] = Vp_row[:, t1] - g(t[t1], t[t1], t[tk])
    Vp_row[:, tk] = Vp_row[:, tk] + g(t[tk], t[t1], t[tk])

    if pow != 0:
        if ord == 0:
            scale_fac = np.linalg.norm(
                np.ndarray.flatten(V_row[:, t1:tk+1:gap]), nrm)
        elif ord == 1:
            scale_fac = np.linalg.norm(
                np.ndarray.flatten(Vp_row[:, t1:tk+1:gap]), nrm)
        else:
            scale_fac = np.mean(dts)
        Vp_row = Vp_row/scale_fac
        V_row = V_row/scale_fac
    return V_row, Vp_row


def Uniform_grid(t, L, param):
    M = len(t)
    #M = t.shape[0]
    #sess = tf.compat.v1.Session()
    #M = M.eval(session=sess)

    #p = int(np.floor(1/8*((L**2*rho**2 - 1) + np.sqrt((L**2*rho**2 - 1)**2 - 8*L**2*rho**2))))
    p = 16

    #overlap = int(np.floor(L*(1 - np.sqrt(1 - s**(1/p)))))
    #print("support and overlap", L, overlap)
    overlap = L - 1

    # create grid
    grid = []
    a = 0
    b = L
    grid.append([a, b])
    while b - overlap + L < M :
        a = b - overlap
        b = a + L
        grid.append([a, b])

    grid = np.asarray(grid)
    N = len(grid)

    V = np.zeros((N, M))
    Vp = np.zeros((N, M))

    for k in range(N):
        g, gp = basis_fcn(p, p)
        a = grid[k][0]
        b = grid[k][1]
        V_row, Vp_row = tf_mat_row(g, gp, t, a, b, param)
        V[k, :] = V_row
        Vp[k, :] = Vp_row

    return V, Vp


def Uniform_grid_1(dt, M, L, param):
    #p = int(np.floor(1/8*((L**2*rho**2 - 1) + np.sqrt((L**2*rho**2 - 1)**2 - 8*L**2*rho**2))))
    p = 16

    #overlap = int(np.floor(L*(1 - np.sqrt(1 - s**(1/p)))))
    ##print("support and overlap", L, overlap)
    overlap = L - 1

    #a test function on every time point
    # create grid
    grid = []
    a = 0
    b = L
    grid.append([a, b])
    while a - overlap + L < M - L/2:
        a = b - overlap
        b = a + L
        grid.append([a, b])

    right_pad = b - (M-1)

    b = overlap
    a = b - L
    grid.insert(0, [a, b])
    while b + overlap - L >= L//2:
        b = b - L + overlap 
        a = b - L
        grid.insert(0, [a, b])

    left_pad = np.abs(a)

    N = len(grid)
    grid = np.asarray(grid)

    offset = -grid[0, 0]

    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            grid[i, j] = grid[i, j] + offset

    
    t = np.arange(0, dt*(M + left_pad + right_pad), dt)
    print(t)
    padded_M = len(t)
    
    V = np.zeros((N, padded_M))
    Vp = np.zeros((N, padded_M))

    for k in range(N):
        g, gp = basis_fcn(p, p)
        a = grid[k][0]
        b = grid[k][1]
        V_row, Vp_row = tf_mat_row(g, gp, t, a, b, param)
        V[k, :] = V_row
        Vp[k, :] = Vp_row

    V = V[:, left_pad: padded_M - right_pad]
    Vp = Vp[:, left_pad: padded_M - right_pad]

    return V, Vp
