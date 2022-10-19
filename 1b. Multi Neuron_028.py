# Kania Meliana Fityanti_21091397028
# Multi Neuron

# Inisialisasi menggunakan numpy
import numpy as np

# Inisialisasi variabel
# Input layer feature 10
inputs = [1, 2, 3.5, 3, 5, 6, 4.5, 1.5, 4, 2.5]

# Inisialisasi bobot variabel
# Panjang Weights sama dengan panjang Input, yaitu 10
# Jumlah Weights sesuai dengan jumlah Neuron yaitu 5
weights = [[0.2, 0.5, 0.8, 0.3, 0.9, 0.7, 0.1, 0.6, 0.4, -0.3],
           [0.1, 0.3, 0.9, 0.21, 0.8, -0.3, 0.4, 0.37, 0.2, 0.7],
           [0.5, 0.6, 0.47, -0.4, 0.31, 0.26, 0.4, 0.2, 0.15, -0.2],
           [0.3, -0.2, 0.46, 0.5, -0.4, 0.15, 0.25, 0.2, 0.13, -0.5],
           [0.15, 0.27, -0.2, 0.2, 0.16, -0.7, 0.31, 0.13, 0.18, 0.4]]

# Jumlah bias sama dengan jumlah Neuron yaitu 5
biases = [1, 0.5, 2, 3,5 ]

# Perhitungan output
layer_outputs = np.dot(weights,inputs) + biases
print(layer_outputs)
