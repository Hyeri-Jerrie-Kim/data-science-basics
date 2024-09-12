# Numpy Lab Exercise

#### Author : Hyeri Kim


import numpy as np
import numpy.linalg as linalg
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Define a custom function to apply over array coordinates
def my_function_hyeri(n, x, y):
    """
    Calculate the product of (2x) and (3y) added to n.
    This function will be used with np.fromfunction to generate an array.
    """
    return (2 * x) * (3 * y) + n

## 1. Using np.fromfunction

# Generate a 3D array using np.fromfunction
# np.fromfunction calls 'my_function_hyeri' for each coordinate to populate the array
result_fromfunction = np.fromfunction(my_function_hyeri, (3, 2, 6), dtype=int)
print("Result using np.fromfunction:")
print(result_fromfunction)

## 2. Alternative Approach to np.fromfunction

# Create index arrays for broadcasting
# 'x_indices' represents the x-dimension (reshaped for broadcasting)
x_indices = np.arange(2).reshape(1, 2, 1)

# 'y_indices' represents the y-dimension (reshaped for broadcasting)
y_indices = np.arange(6).reshape(1, 1, 6)

# 'n_indices' represents the n-dimension (reshaped for broadcasting)
n_indices = np.arange(3).reshape(3, 1, 1)

# Apply the custom function using broadcasting to achieve the same result as np.fromfunction
result_alternative = my_function_hyeri(n_indices, x_indices, y_indices)
print("Result using alternative broadcasting approach:")
print(result_alternative)

## 3. Data Buffer Example

# Create an array with 4-byte integer (int32) data type
hyeri = np.array([[1, 2], [1000, 2000]], dtype=np.int32)

# Convert the array data to bytes and display
# 'tobytes' shows the internal binary representation of the array in memory
data_bytes = hyeri.data.tobytes()
print(data_bytes)

# Change the array to a 2-byte integer (int16) and convert to bytes again
hyeri = np.array([[1, 2], [1000, 2000]], dtype=np.int16)
data_bytes = hyeri.data.tobytes()
print(data_bytes)

## 4. Reshape and Upcasting Examples

# Create a 1D array with values from 0 to 23
g = np.arange(24)
print(g)

# Reshape the array into a 6x4 matrix (2D array)
g.shape = (6, 4)
print(g)

# Reshape the array into a 2x3x4 (3D array)
g.shape = (2, 3, 4)
print(g)

# Reshape into a different format (4x6 matrix) and modify an element
g2 = g.reshape(4, 6)
g2[1, 2] = 999  # Modify one element to illustrate how reshaping affects the array
print(g2)

# Reshape the original array to 2x12 for 'g_Hyeri'
g_hyeri = g.reshape(2, 12)
print(g_hyeri)

# Create an integer array 'k1' of type int32
k1 = np.arange(0, 5, dtype=np.int32)
print(k1)

# Demonstrate type promotion by adding arrays of different types (int32 + int8)
k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print(k2.dtype, k2)

# Add a float to an integer array to show upcasting to float
k3 = k1 + 1.5
print(k3.dtype, k3)

## 5. Conditional Operators and Summing Across Axes

# Use conditional operators to filter elements in an array
m = np.array([20, -5, 30, 40])
result = m < 35  # Boolean array indicating which elements are less than 35
print(result)
print(m[m < 35])  # Display the elements that meet the condition

# Sum the values across the last axis (columns) of a 3D array
c = np.arange(24).reshape(2, 3, 4)  # Create a 3D array with shape (2, 3, 4)
c_sum_axis2 = c.sum(axis=2)  # Sum over the columns for each sub-array
print(c_sum_axis2)

## 6. Binary Universal Functions (ufuncs)

# Demonstrate binary ufuncs using np.copysign
# Change the signs of elements in array 'b' to match the signs of array 'a'
a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])
copysign_result = np.copysign(b, a)
print(copysign_result)

# Reverse the copysign operation to change signs in 'a' to match 'b'
copysign_result_reverse = np.copysign(a, b)
print(copysign_result_reverse)

## 7. Differences with Regular Python Arrays

# Illustrate differences between NumPy arrays and regular Python lists
a = np.array([1, 5, 3, 19, 13, 7, 3])
a_slice = a[2:6]  # Create a slice from index 2 to 5 (inclusive)
a_slice[1] = 1000  # Modify the slice, which also changes the original array
print(a)

## 8. Multi-dimensional Arrays and Boolean Indexing

# Create a multi-dimensional array and extract specific values
b = np.arange(48).reshape(4, 12)  # Reshape a 1D array into a 4x12 matrix
b_values = b[1, 4:7]  # Extract values in row 1 from columns 4 to 6
print(b_values)

# Use Boolean indexing to extract specific columns
cols_on2_hyeri = np.array([True] + [False] * 10 + [True])
first_and_last_columns = b[:, cols_on2_hyeri]  # Select only the first and last columns
print(first_and_last_columns)

## 9. Iterating Over Elements to Find Zeros

# Function to iterate over elements of an array and check for zeros
def check_zero(array):
    """Check for zero elements in the array and print their status."""
    for index, value in enumerate(array.flat):
        print(f"element{index + 1:02d}: {value != 0}")

# Use the check_zero function to examine the 3D array 'c'
c = np.arange(24).reshape(2, 3, 4)
check_zero(c)

## 10. Vertically Stacking and Concatenating Arrays

# Vertically stack two arrays 'q1' and 'q2' using np.vstack
# Creates a single array by stacking 'q1' on top of 'q2'
q1 = np.full((3, 4), 1.0)  # Create a 3x4 matrix filled with 1.0
q2 = np.full((4, 4), 2.0)  # Create a 4x4 matrix filled with 2.0
q5_hyeri = np.vstack((q1, q2))
print(q5_hyeri)

# Concatenate two arrays 'q1' and 'q3' along the first axis (rows)
# Demonstrates how to combine arrays of the same number of columns
q3 = np.full((3, 4), 3.0)  # Create a 3x4 matrix filled with 3.0
q8_hyeri = np.concatenate((q1, q3), axis=0)  # Concatenate along rows
print(q8_hyeri)

## 11. Transpose of Arrays

# Create an ndarray of zeros with shape (2, 7)
t_hyeri = np.zeros((2, 7), dtype=int)  
print(t_hyeri)

# Transpose the matrix 't_hyeri' (swap rows and columns)
# Useful for changing data orientation in data analysis
t_hyeri_transposed = t_hyeri.transpose()
print(t_hyeri_transposed)

## 12. Matrix Multiplication and Inversion

# Perform matrix multiplication using np.dot
# Demonstrates linear algebra operations like dot product
a1 = np.arange(8).reshape(2, 4)  # Create a 2x4 matrix
a2 = np.arange(8).reshape(4, 2)  # Create a 4x2 matrix
a3 = np.dot(a1, a2)  # Matrix multiplication resulting in a 2x2 matrix
print(a3)
print(a3.shape)  # Print the shape to confirm dimensions

# Calculate the pseudo-inverse of a matrix using linalg.pinv
# Pseudo-inverse is used when a matrix is not invertible
hyeri = np.arange(16).reshape(4, 4)  # Create a 4x4 matrix
hyeri_pinv = linalg.pinv(hyeri)
print(hyeri)
print(hyeri_pinv)

## 13. Identity Matrix and Determinant Calculation

# Create a 5x5 identity matrix
# Identity matrices are used as the multiplicative identity in matrix algebra
identity_matrix = np.eye(5)
print(identity_matrix)

# Calculate the determinant of a random 3x3 matrix
# Determinants are used in linear algebra to solve systems of equations
random_arr = np.random.rand(3, 3)  # Generate a random 3x3 matrix
determinant = linalg.det(random_arr)
print(determinant)

## 14. Eigenvalues and Eigenvectors

# Compute the eigenvalues and eigenvectors of a 4x4 matrix
# Eigenvalues and eigenvectors are used in data analysis to understand the variance structure of data
e_hyeri = np.random.rand(4, 4)  # Create a random 4x4 matrix
eigenvalues, eigenvectors = linalg.eig(e_hyeri)
print(eigenvalues)  # Eigenvalues represent the magnitude of variance in the data
print(eigenvectors)  # Eigenvectors represent the direction of variance

## 15. Solving a System of Linear Equations

# Solve a system of linear equations using linalg.solve
# The system represents equations in the form Ax = B, where A is a matrix of coefficients and B is a vector of constants
coeffs = np.array([[2, 4, 1], [3, 8, 2], [1, 2, 3]])  # Coefficient matrix A
depvars = np.array([8, 16, -2])  # Constants vector B
solution = linalg.solve(coeffs, depvars)  # Solve for x
print(solution)

# Verify the solution by checking if A*x equals B
print(np.allclose(coeffs.dot(solution), depvars))

## 16. Visualization with Matplotlib

# Create a cosine plot using NumPy and Matplotlib
# Demonstrates how to visualize data using mathematical functions
x_coords = np.arange(0, 1024)  # X-axis coordinates
y_coords = np.arange(0, 768)   # Y-axis coordinates
X, Y = np.meshgrid(x_coords, y_coords)  # Create a meshgrid for plotting
data1 = np.cos(X * Y / 40.5)  # Generate a cosine wave pattern

# Plot the cosine data using Matplotlib's imshow for visualization
plt.imshow(data1, cmap=cm.hot, interpolation="bicubic")  
plt.show()

## 17. Saving and Loading Data with NumPy

# Save a randomly generated matrix to a CSV file
# Demonstrates how to store data for later use
sav_hyeri = np.random.rand(4, 4)  # Create a 4x4 matrix with random values
np.savetxt("Hyeri_sav.csv", sav_hyeri, delimiter=",")  # Save to CSV

# Load the saved matrix back from the CSV file
# Demonstrates how to read data from files
load_hyeri = np.loadtxt("Hyeri_sav.csv", delimiter=",")
print(load_hyeri)