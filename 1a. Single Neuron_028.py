# Kania Meliana Fityanti_21091397028
# Single Neuron

# Inisialisasi menggunakan numpy
import numpy as np

# Inisialisasi variabel
# Input layer feature 10
inputs = [2, 3.5, 4, 2.5, 6, 3, 5, 7, 10, 4.5]

# Inisialisasi bobot variabel
# Panjang Weights sama dengan panjang Input, yaitu 10
# Jumlah Weights sesuai dengan jumlah Neuron, yaitu 1
weights = [0.3, 0.9, 0.4, 0.27, -0.5, 0.8, 0.2, 0.6, 0.7, 0.17]

# Jumlah bias sama dengan jumlah Neuron, yaitu 1
bias = 5

# Perhitungan output
output = np.dot(weights,inputs) + bias
print(output)
