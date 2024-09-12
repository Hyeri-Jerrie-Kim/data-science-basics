# Numpy Lab Exercise

#### Author : Hyeri Kim


import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define a custom function
def my_function_hyeri(n, x, y):
    """Calculate the product of (2x) and (3y) added to n."""
    return (2 * x) * (3 * y) + n

# Section 1: Using np.fromfunction

# Generate a 3D array using np.fromfunction
result_fromfunction = np.fromfunction(my_function_hyeri, (3, 2, 6), dtype=int)
print("Result using np.fromfunction:")
print(result_fromfunction)

# Section 2: Alternative Approach to np.fromfunction

# Create x, y, and n indices using np.arange and reshaping for broadcasting
x_indices = np.arange(2).reshape(1, 2, 1)  # Reshape for broadcasting
y_indices = np.arange(6).reshape(1, 1, 6)  # Reshape for broadcasting
n_indices = np.arange(3).reshape(3, 1, 1)  # Reshape for broadcasting

# Apply the function using broadcasting
result_alternative = my_function_hyeri(n_indices, x_indices, y_indices)
print("Result using alternative broadcasting approach:")
print(result_alternative)

# Data buffer example with type change
# Define an array with int32 (4 bytes) and convert it to int16 (2 bytes)
hyeri = np.array([[1, 2], [1000, 2000]], dtype=np.int32)

# Convert data to bytes format and display
data_bytes = hyeri.data.tobytes()
print(data_bytes)

# Change the array type to int16 and convert to bytes format again
hyeri = np.array([[1, 2], [1000, 2000]], dtype=np.int16)
data_bytes = hyeri.data.tobytes()
print(data_bytes)

# Reshape example
g = np.arange(24)  # Create a 1D array with values from 0 to 23
print(g)

# Reshape to multiple formats
g.shape = (6, 4)
print(g)
g.shape = (2, 3, 4)
print(g)
g2 = g.reshape(4, 6)
g2[1, 2] = 999  # Modify the reshaped array
print(g2)

# Reshape the array to 2 by 12 for g_Hyeri
g_hyeri = g.reshape(2, 12)
print(g_hyeri)

# Upcasting example
k1 = np.arange(0, 5, dtype=np.int32)  # Array with int32 type
print(k1)

k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)  # Add int8 array to int32 array
print(k2.dtype, k2)

k3 = k1 + 1.5  # Adding float to int array promotes to float
print(k3.dtype, k3)

# Conditional operators example
m = np.array([20, -5, 30, 40])
result = m < 35  # Check which elements are less than 35
print(result)
print(m[m < 35])  # Elements less than 35

# Sum across axis example
c = np.arange(24).reshape(2, 3, 4)  # Create a 3D array
c_sum_axis2 = c.sum(axis=2)  # Sum across columns
print(c_sum_axis2)

# Binary ufuncs example
a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])
copysign_result = np.copysign(b, a)  # Change the sign of 'b' to match 'a'
print(copysign_result)

copysign_result_reverse = np.copysign(a, b)  # Change the sign of 'a' to match 'b'
print(copysign_result_reverse)

# Differences with regular Python arrays example
a = np.array([1, 5, 3, 19, 13, 7, 3])
a_slice = a[2:6]
a_slice[1] = 1000  # Modify slice, which changes the original array
print(a)

# Multi-dimensional array example
b = np.arange(48).reshape(4, 12)  # Create a 2D array
b_values = b[1, 4:7]  # Extract values 16, 17, 18
print(b_values)

# Boolean indexing example
cols_on2_hyeri = np.array([True] + [False] * 10 + [True])
first_and_last_columns = b[:, cols_on2_hyeri]  # Extract first and last columns
print(first_and_last_columns)

# Iterating over elements to find zeros
c = np.arange(24).reshape(2, 3, 4)
def check_zero(array):
    """Check for zero elements in the array and print their status."""
    for index, value in enumerate(array.flat):
        print(f"element{index + 1:02d}: {value != 0}")

check_zero(c)  # Function to check for zero values

# Vertically stack arrays
q1 = np.full((3, 4), 1.0)
q2 = np.full((4, 4), 2.0)
q5_hyeri = np.vstack((q1, q2))
print(q5_hyeri)

# Concatenate arrays example
q3 = np.full((3, 4), 3.0)
q8_hyeri = np.concatenate((q1, q3), axis=0)
print(q8_hyeri)

# Transpose example
t_hyeri = np.zeros((2, 7), dtype=int)  # Creating an ndarray of zeros
print(t_hyeri)

t_hyeri_transposed = t_hyeri.transpose()
print(t_hyeri_transposed)

# Matrix multiplication example
a1 = np.arange(8).reshape(2, 4)
a2 = np.arange(8).reshape(4, 2)
a3 = np.dot(a1, a2)
print(a3)
print(a3.shape)

# Matrix inverse and pseudo-inverse example
hyeri = np.arange(16).reshape(4, 4)
hyeri_pinv = linalg.pinv(hyeri)
print(hyeri)
print(hyeri_pinv)

# Identity matrix example
identity_matrix = np.eye(5)
print(identity_matrix)

# Determinant example
random_arr = np.random.rand(3, 3)
determinant = linalg.det(random_arr)
print(determinant)

# Eigenvalues and eigenvectors example
e_hyeri = np.random.rand(4, 4)
eigenvalues, eigenvectors = linalg.eig(e_hyeri)
print(eigenvalues)
print(eigenvectors)

# Solving a system of linear scalar equations example
coeffs = np.array([[2, 4, 1], [3, 8, 2], [1, 2, 3]])
depvars = np.array([8, 16, -2])
solution = linalg.solve(coeffs, depvars)
print(solution)
print(np.allclose(coeffs.dot(solution), depvars))  # Check the solution

# Cosine graph example
x_coords = np.arange(0, 1024)
y_coords = np.arange(0, 768)
X, Y = np.meshgrid(x_coords, y_coords)
data1 = np.cos(X * Y / 40.5)
plt.imshow(data1, cmap=cm.hot, interpolation="bicubic")
plt.show()

# Text format example
sav_hyeri = np.random.rand(4, 4)
np.savetxt("Hyeri_sav.csv", sav_hyeri, delimiter=",")
load_hyeri = np.loadtxt("Hyeri_sav.csv", delimiter=",")
print(load_hyeri)