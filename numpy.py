# Numpy Lab Exercise

#### Author : Hyeri Kim


import numpy as np

def my_function_Hyeri(n, x, y):
    return (2*x) * (3*y) + n

np.fromfunction(my_function_Hyeri, (3, 2, 6), dtype = int)

"""## Data buffer
Change the name of the array from ‘f’ to your firstname and the type from 4 bytes to 2 bytes and rerun both cells and notice the difference.

"""

hyeri = np.array([[1,2],[1000,2000]], dtype = np.int32)

if (hasattr(hyeri.data, "tobytes")):
    data_bytes = hyeri.data.tobytes() # python 3
else:
    data_bytes = memoryview(hyeri.data).tobytes() # python 2

data_bytes

hyeri = np.array([[1,2],[1000,2000]], dtype = np.int16)

if (hasattr(hyeri.data, "tobytes")):
    data_bytes = hyeri.data.tobytes() # python 3
else:
    data_bytes = memoryview(hyeri.data).tobytes() # python 2

data_bytes

"""## Re-shape
Add a cell at the end of the section to reshape g into a 2 by 12 ndaray and save it to a variable named g_firstname where firstname is your firstname.
"""

g = np.arange(24)
print(g)
print("Rank:", g.ndim)

g.shape = (6, 4)
print(g)
print("Rank:", g.ndim)

g.shape = (2, 3, 4)
print(g)
print("Rank:", g.ndim)

g2 = g.reshape(4,6)
print(g2)
print("Rank:", g2.ndim)

g2[1, 2] = 999
g2

g

g_Hyeri = g.reshape(2,12)
g_Hyeri

"""## Upcasting
Change the type of k1 to int32
"""

k1 = np.arange(0, 5, dtype=np.int32)
print(k1.dtype, k1)

k2 = k1 + np.array([5, 6, 7, 8, 9], dtype=np.int8)
print(k2.dtype, k2)

k3 = k1 + 1.5
print(k3.dtype, k3)

"""## Conditional operators
Change the value of comparison from 25 to 35.
"""

m = np.array([20, -5, 30, 40])
m < [15, 16, 35, 36]

m < 35  # equivalent to m < [35, 35, 35, 35]

m[m < 35]

"""## ndarray methods
Add a cell showing the result of summing over axis 2 for array c and notice the difference.
"""

a = np.array([[-2.5, 3.1, 7], [10, 11, 12]])
print(a)
print("mean =", a.mean())

for func in (a.min, a.max, a.sum, a.prod, a.std, a.var):
    print(func.__name__, "=", func())

c=np.arange(24).reshape(2,3,4)
c

c.sum(axis=0)  # sum across matrices

c.sum(axis=1)  # sum across rows

c.sum(axis=(0,2))  # sum across matrices and columns

c.sum(axis=2) # sum across columns

"""## Binary ufuncs
Add a cell to show the result of copysign of (b,a)
"""

a = np.array([1, -2, 3, 4])
b = np.array([2, 8, -1, 7])
np.add(a, b)  # equivalent to a + b

np.greater(a, b)  # equivalent to a > b

np.maximum(a, b)

np.copysign(a, b)  # change the sign of 'a array' to the one of 'b array'

np.copysign(b, a)

"""## Differences with regular python arrays
Add a cell after amending the third element of a_slice to the value 2000 to print the final result of a
"""

a = np.array([1, 5, 3, 19, 13, 7, 3])

a[2:5] = -1

a_slice = a[2:6]
a_slice[1] = 1000
a  # the original array was modified!

a[3] = 2000
a_slice  # similarly, modifying the original array modifies the slice!

a

another_slice = a[2:6].copy()
another_slice[1] = 3000
a  # the original array is untouched

a[3] = 4000
another_slice  # similary, modifying the original array does not affect the slice copy

"""## Multi-dimensional arrays
Add a cell to extract values 16,17,18
"""

b = np.arange(48).reshape(4, 12)
b

b[1, 2]  # row 1, col 2

b[1, :]  # row 1, all columns

b[:, 1]  # all rows, column 1

b[1, :]

b[1:2, :]

"""The first expression returns row 1 as a 1D array of shape `(12,)`, while the second returns that same row as a 2D array of shape `(1, 12)`."""

b[1, 4:7]

"""## Boolean indexing
Using Boolean indexing, add a cell to extract the first and last columns of array b.(name the boolean array  cols_on2_firstname where first name is your firstname)
"""

b = np.arange(48).reshape(4, 12)
b

rows_on = np.array([True, False, True, False])
b[rows_on, :]  # Rows 0 and 2, all columns. Equivalent to b[(0, 2), :]

cols_on = np.array([False, True, False] * 4)
b[:, cols_on]  # All rows, columns 1, 4, 7 and 10

cols_on2_Hyeri =  np.array([True] + [False] * 10 + [True])
b[:, cols_on2_Hyeri]

"""## Iterating
Add a cell to iterate over c and print the Boolean values for items equivalent to zeros.
"""

c = np.arange(24).reshape(2, 3, 4)  # A 3D array (composed of two 3x4 matrices)
c

for m in c:
    print("Item:")
    print(m)

for i in range(len(c)):  # Note that len(c) == c.shape[0]
    print("Item:")
    print(c[i])

for i in c.flat:
    print("Item:", i)

def check_zero(array):
    j = 1
    for i in array.flat:
        if i == 0:
            print(f"element{j:02d}: " + str(False))
        else:
            print(f"element{j:02d}: " + str(True))
        j += 1

check_zero(c)

"""## vstack
Add a cell to create a variable name it q5_firstname where firstname is your firstname and vertically stack q1 and q2 and print the output.
"""

q1 = np.full((3,4), 1.0)
q1

q2 = np.full((4,4), 2.0)
q2

q3 = np.full((3,4), 3.0)
q3

q4 = np.vstack((q1, q2, q3))
q4

q5_Hyeri = np.vstack((q1,q2))
q5_Hyeri

"""## concatenate
Add a cell to create a variable name it q8_firstname where firstname is your firstname , concatenate q1 and q3 and print the results.
"""

q5 = np.hstack((q1, q3))
q5

q7 = np.concatenate((q1, q2, q3), axis=0)  # Equivalent to vstack
q7

q8_Hyeri = np.concatenate((q1, q3), axis=0)
q8_Hyeri

"""## Transpose
Add a cell and create a variable named t_firstname where firstname is your name, let the variable hold any ndaray size 2 by 7 with zero values, print the result then transpose and print the result.
"""

t = np.arange(24).reshape(4,2,3)
t

t1 = t.transpose((1,2,0))
t1
t1.shape

t2 = t.transpose()  # equivalent to t.transpose((2, 1, 0))
t2
t2.shape

t3 = t.swapaxes(0,1)  # equivalent to t.transpose((1, 0, 2))
t3
t3.shape

t_Hyeri = np.zeros(14, dtype = int).reshape(2, 7)
t_Hyeri

t_Hyeri_trans = t_Hyeri.transpose()
t_Hyeri_trans

"""## Matrix multiplication
Add a cell to create 2 ndarys name the first a1 and the second a2. Both arrays should contain numbers in the range 0 to 7. Print a1 and a2. Create a new variable a3 which holds the product multiplication of a1 and a2 name it a3 and print the output of a3, then the shape of a3.
"""

a1 = np.arange(8).reshape(2,4)
print(a1)

a2 = np.arange(8).reshape(4,2)
print(a2)

a3 = a1.dot(a2)
print(a3)

a3.shape

"""## Matrix inverse and pseudo-inverse
Add a cell to create a new 4 by 4 ndaray with values between0 and 15, name the variable that holds the array your first name, print the array and the inverse of the array.
"""

import numpy.linalg as linalg

hyeri = np.arange(16).reshape(4,4)
hyeri

linalg.pinv(hyeri)  # The result of linalg.inv(hyeri) is invalid; since it is a singular matrix.

"""## Identity matrix
Add a cell to create a 5 by 5 identity array.
"""

np.eye(5)

"""## Determinant
Add a cell to create a 3 by 3 matrix with values generated randomly then printout the determinant of the matrix.
"""

random_arr = np.random.rand(3, 3)
random_arr

linalg.det(random_arr)

"""## Eigenvalues and eigenvectors
Add a cell to create a 4 by 4 matrix with values generated randomly, assign the matrix to a variable named e_firstname. Printout the Eigenvalue and eigenvectors of the matrix.
"""

e_Hyeri = np.random.rand(4, 4)
e_Hyeri

eigenvalues, eigenvectors = linalg.eig(e_Hyeri)

eigenvalues # λ

eigenvectors # v

"""## Solving a system of linear scalar equations
Add a cell to solve the following linear equations:

2x+4y+z =8

3x+8y+2z =16

X+2y+3z = -2

Check the results using the allcolse method.
"""

coeffs  = np.array([[2, 4, 1], [3, 8, 2], [1, 2, 3]])
depvars = np.array([8, 16, -2])
solution = linalg.solve(coeffs, depvars)
solution

np.allclose(coeffs.dot(solution), depvars)  # Check the solution

"""## Vectorization
Add cells to replicate the example but instead of sin use cos. Name the data data1 and produce a graph.
"""

x_coords = np.arange(0, 1024)
y_coords = np.arange(0, 768)
X, Y = np.meshgrid(x_coords, y_coords)

X

Y

data1 = np.cos(X*Y/40.5)

import matplotlib.pyplot as plt
import matplotlib.cm as cm
fig = plt.figure(1, figsize=(7, 6))
plt.imshow(data1, cmap=cm.hot, interpolation="bicubic")
plt.show()

"""## Text format
Add cells to create a 4 by 4 matrix with values generated randomly, assign the matrix to a variable named sav_firstname. Save the matrix to an csv file named your firstname_sav. Load the csv file into a new variable called load_firstname.
"""

sav_Hyeri = np.random.rand(4, 4)
sav_Hyeri

np.savetxt("Hyeri_sav.csv", sav_Hyeri, delimiter=",")

with open("Hyeri_sav.csv", "rt") as load_Hyeri:
    print(load_Hyeri.read())

load_Hyeri = np.loadtxt("Hyeri_sav.csv", delimiter=",")
load_Hyeri