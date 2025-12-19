import numpy as np

# Function to generate N data points from a normal distribution
def generate_data_SN(N, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if p:
        discrete = np.array([False for _ in range(p)])
        return np.random.normal(loc=0, scale=1, size=(N, p)), discrete
    else:
        return np.random.normal(loc=0, scale=1, size=N), False

def generate_data_TP(N, prob=0.5, value_1=0, value_2=1, p=None, seed=None):
    # Generate the distribution
    if seed is not None:
        np.random.seed(seed)
    if p:
        discrete = np.array([True for _ in range(p)])
        return np.random.choice([value_1, value_2], size=(N, p), p=[prob, 1-prob]), discrete
    else:
        return np.random.choice([value_1, value_2], size=N, p=[prob, 1-prob]), True

def generate_data_U(N, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if p:
        discrete = np.array([False for _ in range(p)])
        return np.random.uniform(low=0, high=1, size=(N, p)), discrete
    else:
        return np.random.uniform(low=0, high=1, size=N), False

# Function to generate N data points from a discrete uniform distribution
def generate_data_DR(N, value_1=1, value_2=11, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if p:
        discrete = np.array([True for _ in range(p)])
        return np.random.choice(range(value_1, value_2), size=(N, p), replace=True), discrete
    else:
        return np.random.choice(range(value_1, value_2), size=N, replace=True), True

def generate_data_Gumbel(N, loc=0, scale=1, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if p:
        discrete = np.array([False for _ in range(p)])
        return np.random.gumbel(loc=loc, scale=scale, size=(N, p)), discrete
    else:
        return np.random.gumbel(loc=loc, scale=scale, size=N), False

def generate_data_student_t(N, df=3, loc=0, scale=1, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if p:
        discrete = np.array([False for _ in range(p)])
        return np.random.standard_t(df=df, size=(N, p)) * scale + loc, discrete
    else:
        return np.random.standard_t(df=df, size=N) * scale + loc, False

def generate_data_normal(N, loc=50, scale=20, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if p:
        discrete = np.array([False for _ in range(p)])
        return np.random.normal(loc=loc, scale=scale, size=(N, p)), discrete
    else:
        return np.random.normal(loc=loc, scale=scale, size=N), False

def generate_data_mixing(N, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x1, discrete1 = generate_data_SN(N, p=p//5)
    x2, discrete2 = generate_data_normal(N, p=p//5)
    x3, discrete3 = generate_data_DR(N, p=p//5)
    x4, discrete4 = generate_data_U(N, p=p//5)
    x5, discrete5 = generate_data_TP(N, p=p//5)
    x = np.concatenate([x1, x2, x3, x4, x5], axis=1)
    discrete = np.concatenate([discrete1, discrete2, discrete3, discrete4, discrete5])
    shuffled_indices = np.random.permutation(x.shape[1])
    return x[:, shuffled_indices], discrete[shuffled_indices] 

def generate_data_Poisson(N, lam=1.0, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if p:
        discrete = np.array([True for _ in range(p)])
        data = np.random.poisson(lam=lam, size=(N, p))
        return data, discrete
    else:
        data = np.random.poisson(lam=lam, size=N)
        return data, True

def generate_data_Geometric(N, p_success=0.1, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if p:
        discrete = np.array([True for _ in range(p)])
        data = np.random.geometric(p=p_success, size=(N, p))
        return data, discrete
    else:
        data = np.random.geometric(p=p_success, size=N)
        return data, True

def generate_data_Lognormal(N, mean=0.0, sigma=1.0, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)

    if p:
        discrete = np.array([False for _ in range(p)])
        data = np.random.lognormal(mean=mean, sigma=sigma, size=(N, p))
        return data, discrete
    else:
        data = np.random.lognormal(mean=mean, sigma=sigma, size=N)
        return data, False

def generate_data_Exponential(N, scale=1.0, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if p:
        discrete = np.array([False for _ in range(p)])
        data = np.random.exponential(scale=scale, size=(N, p))
        return data, discrete
    else:
        data = np.random.exponential(scale=scale, size=N)
        return data, False

def generate_data_Gamma(N, shape=0.5, scale=1.0, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    
    if p:
        discrete = np.array([False for _ in range(p)])
        data = np.random.gamma(shape=shape, scale=scale, size=(N, p))
        return data, discrete
    else:
        data = np.random.gamma(shape=shape, scale=scale, size=N)
        return data, False

def generate_data_mixing_v2(N, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    x1, discrete1 = generate_data_Geometric(N, p=p//5)
    x2, discrete2 = generate_data_TP(N, p=p//5)
    x3, discrete3 = generate_data_Gamma(N, p=p//5)
    x4, discrete4 = generate_data_Lognormal(N, p=p//5)
    x5, discrete5 = generate_data_Exponential(N, p=p//5)
    x = np.concatenate([x1, x2, x3, x4, x5], axis=1)
    discrete = np.concatenate([discrete1, discrete2, discrete3, discrete4, discrete5])
    shuffled_indices = np.random.permutation(x.shape[1])
    return x[:, shuffled_indices], discrete[shuffled_indices] 
          
def generate_relevant_data(N, p=None, seed=None):
    data = np.zeros((N, p))
    
    data[:, 0] = np.random.normal(loc=0, scale=1, size=N)
    
    for j in range(1, p):
        epsilon = np.random.normal(loc=0, scale=1, size=N)
        
        if j % 2 == 1:
            data[:, j] = 2 * data[:, j - 1] + epsilon
        else:
            data[:, j] = data[:, j - 1] / 2 + epsilon
    discrete = np.array([False for _ in range(p)])
    return data, discrete

def generate_noise_N(N, scale=0.1, p=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    if p:
        return np.random.normal(loc=0, scale=scale, size=(N, p))
    else:
        return np.random.normal(loc=0, scale=scale, size=N)

def generate_RGL(p=5):
    pairs = np.tile([1, 0], (p, 1))
    for i, pair in enumerate(pairs):
        np.random.seed(i+207)
        np.random.shuffle(pair)
    S = pairs.flatten()
    return S

def generate_RGL_2d(N, p=None):
    S_list = []
    for _ in range(N):
        pairs = np.tile([1, 0], (p, 1))
        for pair in pairs:
            np.random.shuffle(pair)
        S = pairs.flatten()
        S_list.append(S)
    return np.array(S_list)