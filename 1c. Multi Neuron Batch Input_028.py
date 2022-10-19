# Kania Meliana Fityanti_21091397028
# Multi Neuron Batch Input

# Inisialisasi menggunakan numpy
import numpy as np

# Inisialisasi variabel
# Input layer feature 10
# Per Batch 6 Input
inputs = [[0.5, 1.5, 2.5, 3, 4.5, 5, 1, 4, 3.5, 7],
          [1, 2, 3.5, 3, 5, 6, 4.5, 1.5, 8, 2.5],
          [2.5, 3, 4.5, 5, 1, 4, 3.5, 6, 9, 5.5],
          [4, 4.5, 2, 1.5, 2.5, 9, 0.5, 7, 6, 3],
          [6, 2.5, 9, 2, 1, 1.5, 7, 6.5, 3.5, 4],
          [5, 3, 4.5, 2.5, 2, 9, 3.5, 4, 6, 1]]

# Panjang Weights sama dengan panjang Input, yaitu 10
# Jumlah Weights sesuai dengan jumlah Neuron, yaitu 5
weights = [[0.5, 0.6, 0.47, -0.4, 0.31, 0.26, 0.4, 0.2, 0.15, -0.2],
           [0.2, 0.51, 0.8, 0.3, 0.9, 0.7, 0.1, 0.6, 0.4, -0.3],
           [0.15, 0.27, -0.2, 0.2, 0.16, -0.7, 0.31, 0.13, 0.18, 0.4],
           [0.1, 0.3, 0.9, 0.21, 0.8, 0.35, 0.4, 0.37, 0.2, 0.7],
           [0.3, -0.2, 0.46, 0.5, -0.4, 0.15, 0.25, 0.2, 0.13, -0.5]]

# Jumlah bias sama dengan jumlah Neuron, yaitu 5
biases = [2, 3, 5, 0.5, 4]

# Perhitungan output
layer_outputs = np.dot(inputs,np.array(weights).T) + biases
print(layer_outputs)
