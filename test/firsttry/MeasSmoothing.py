import numpy as np;
import matplotlib.pyplot as plt;
import pandas as pd;

def load_data(file_path):
    data = np.loadtxt(file_path, delimiter=',', skiprows=1)
    x = data[:, 0]
    y = data[:, 1]
    return x, y

def plot_data(x, y, title="Data Plot", xlabel="X-axis", ylabel="Y-axis"):
    plt.plot(x, y)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()

class Measurement:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def mean(self):
        return np.mean(self.y)
    
    def smooth(self, window_size):
        kernel = np.ones(window_size) / window_size
        return np.convolve(self.y, kernel, mode='valid')
    def plot(self):
        x = self.x
        smoothed_y = self.smooth(10)
        plot_data(x[:len(smoothed_y)], smoothed_y, title="Smoothed Data Plot", xlabel="X-axis", ylabel="Smoothed Y-axis")
        
x, y = load_data('wave_data.csv')
measurement = Measurement(x, y)
measurement.plot()