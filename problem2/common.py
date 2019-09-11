import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt
import random

def get_data(is_one_signal=False):
    file = open('./data/shashlik_61_pulses.txt', 'r')
    data = file.readlines()
    data = np.array([list(map(float, experiment.split())) for experiment in data])
   
    X = data[:, 2:]
    y_baseline = data[:, 1]
    y = data[:, 0]
    
    
    X = np.array([experiment - np.max(experiment) for experiment in X])
    X = np.array([experiment/-np.min(experiment) for experiment in X])
    
    ## Let's shift each signal so that reference time matches for each signal
    if not is_one_signal:
        y = y.astype(int)
        mean_ref_time = int(y.mean())
        X = np.array([signal_cyclic_shift(signal, mean_ref_time - y[i]) for i, signal in enumerate(X, 0)])
        y = np.array([mean_ref_time]*len(X))

    return X, y

def get_argmin_distr():
    X_one_signal, _ = get_data(is_one_signal=True)
    return np.argmin(X_one_signal, axis=1)
    
def get_freq_data(X, freq=1, start_point=384):
    X_freq = np.concatenate([X[:, start_point::-freq][:, ::-1], X[:, start_point + freq::freq]], axis=1)
    return X_freq

def signal_cyclic_shift(signal, tau):
    signal_start = signal[:-tau]
    
    new_signal = np.concatenate([signal[-tau:], signal_start])
    
    return new_signal
    
def get_ref_time(first_impulse, second_impulse, first_ref_time, second_ref_time):
    if np.min(first_impulse) < np.min(second_impulse):
         return first_ref_time
    else:
        return second_ref_time

ARGMIN_DISTR = get_argmin_distr()
    
def generate_multi_signal(X_origin, y_origin, tau, alpha, to_plot=False):
    first_idx, second_idx = np.random.choice(X_origin.shape[0], 2, replace=False)
    first_impulse = X_origin[first_idx]
    second_impulse = X_origin[second_idx]
    
    first_ref_time = y_origin[first_idx]
    second_ref_time = y_origin[second_idx]
    
    second_impulse = signal_cyclic_shift(second_impulse, tau)
    second_ref_time += tau
        
    multi_impulse = first_impulse + second_impulse*alpha
    multi_impulse /= -np.min(multi_impulse)
    
    mean_argmin = int(np.mean(np.argmin(X_origin, axis=1)))
    new_pos = random.choice(ARGMIN_DISTR)
    
    first_impulse_shifted = signal_cyclic_shift(first_impulse, new_pos - np.argmin(first_impulse))
    second_impulse_shifted = signal_cyclic_shift(second_impulse, new_pos - np.argmin(second_impulse))
    multi_impulse_shifted = signal_cyclic_shift(multi_impulse, new_pos - np.argmin(multi_impulse))

    first_ref_time +=  new_pos - np.argmin(multi_impulse)
    second_ref_time +=  new_pos - np.argmin(multi_impulse)
    
    if to_plot:
        plt.plot(first_impulse)
        plt.plot(second_impulse)
        plt.plot(multi_impulse_shifted)
        plt.legend(['First signal', 'Second signal', 'Sum of signals'])
        plt.show()
        
    ref_time = get_ref_time(first_impulse, second_impulse*alpha, first_ref_time, second_ref_time)
    
    return {'first_impulse': first_impulse_shifted,\
            'second_impulse': second_impulse_shifted,\
            'multi_ref_time': ref_time,\
            'multi_impulse': multi_impulse_shifted}
            
def dict_to_arrays(scores_dict):
    x, y, z = [], [], []
    for tau, alpha_dict in scores_dict.items():
        for alpha, score in alpha_dict.items():
            x.append(tau)
            y.append(alpha)
            z.append(score)
            
    return [x, y, z]