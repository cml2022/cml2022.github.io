---
jupyter:
  jupytext:
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.13.7
  kernelspec:
    display_name: Python 3
    language: python
    name: python3
---

# Intro to Numpy

The following notebook uses a lot of materials (the whole theoretical part) from [the Numpy tutorial](https://nbviewer.jupyter.org/github/cgpotts/cs224u/blob/master/numpy_tutorial.ipynb) created by Will Monroe, Chris Potts, and Lucy Li which is a part of the [Natual Language Understanding class](http://web.stanford.edu/class/cs224u/) from Stanford University.

```python
import numpy as np
```

# Vectrors
## Initialization

```python
np.zeros(5)
```

```python
np.ones(5)
```

```python
# convert list to numpy array
np.array([1,2,3,4,5])
```

```python
# convert numpy array to list
np.ones(5).tolist()
```

```python
# one float => all floats
np.array([1.0,2,3,4,5])
```

```python
# same as above
np.array([1,2,3,4,5], dtype='float')
```

```python
# spaced values in interval
np.array([x for x in range(20) if x % 2 == 0])
```

### Just in case in you forgot the list comprehension in Python

```python
# it returns a range object
range(20)
```

```python
[x for x in range(20)]
```

```python
[x for x in range(20) if x % 2 == 0]
```

```python
np.array([x for x in range(20) if x % 2 == 0])
```

```python
# same as above
np.arange(0,20,2)
```

```python
# random floats in [0, 1)
np.random.random(10)
```

```python
# random integers
np.random.randint(5, 15, size=10)
```

## Vector indexing

```python
x = np.array([10,20,30,40,50])
```

```python
x[0]
```

```python
# slice
x[0:2]
```

```python
x[0:1000]
```

```python
# last value
x[-1]
```

```python
x[-0]
```

```python
# last value as array
x[[-1]]
```

```python
# last 3 values
x[-3:]
```

```python
# pick indices
x[[0,2,4]]
```

## Vector assignment

Be careful when assigning arrays to new variables! 

```python
#x2 = x # try this line instead
x2 = x.copy()
```

```python
x2
```

```python
x2[0] = 10

x2
```

```python
x2[[1,2]] = 10

x2
```

```python
x2[[3,4]] = [0, 1]

x2
```

```python
# check if the original vector changed
x
```

## Vectorized operations

```python
x.sum()
```

```python
x.mean()
```

```python
x.max()
```

```python
x.argmax()
```

```python
np.log(x)
```

```python
np.exp(x)
```

```python
x + x  # Try also with *, -, /, etc.
```

```python
x + 1
```

## Comparison with Python lists

Vectorizing your mathematical expressions can lead to __huge__ performance gains. The following example is meant to give you a sense for this. It compares applying `np.log` to each element of a list with 10 million values with the same operation done on a vector.

```python
# log every value as list, one by one
def listlog(vals):
    return [np.log(y) for y in vals]
```

```python
# get random vector
samp = np.random.random_sample(int(1e7))+1
samp
```

```python
%time _ = np.log(samp)
```

```python
%time _ = listlog(samp)
```

# Matrices

The matrix is the core object of machine learning implementations. 


## Matrix initialization

```python
np.array([[1,2,3], [4,5,6]])
```

```python
np.array([[1,2,3], [4,5,6]], dtype='float')
```

```python
np.zeros((3,5))
```

```python
np.ones((3,5))
```

```python
np.identity(3)
```

```python
np.diag([1,2,3])
```

## Matrix indexing

```python
X = np.array([[1,2,3], [4,5,6]])
X
```

```python
X[0]
```

```python
X[0,0]
```

```python
# get row
X[0, :]
```

```python
# get column
X[ : , 0]
```

```python
# get multiple columns
X[ : , [0,2]]
```

## Matrix assignment

```python
# X2 = X # try this line instead
X2 = X.copy()

X2
```

```python
X2[0,0] = 20

X2
```

```python
X2[0] = 3

X2
```

```python
X2[: , -1] = [5, 6]

X2
```

```python
# check if original matrix changed
X
```

## Matrix reshaping

```python
z = np.arange(1, 7)

z
```

```python
z.shape
```

```python
Z = z.reshape(2,3)

Z
```

```python
Z.shape
```

```python
Z.reshape(6)
```

```python
# same as above
Z.flatten()
```

```python
# transpose
Z.T
```

## Numeric operations

```python
A = np.array(range(1,7), dtype='float').reshape(2,3)

A
```

```python
B = np.array([1, 2, 3])
```

You can read more about broadcasting [here](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html).

```python
# not the same as A.dot(B)
A * B
```

```python
A + B
```

```python
A / B
```

```python
# matrix multiplication
A.dot(B)
```

```python
B.dot(A.T)
```

```python
A.dot(A.T)
```

```python
# outer product
# multiplying each element of first vector by each element of the second
np.outer(B, B)
```

# Exercises
## Task 1
Initialize a vector of zeros of shape (5).

You should get the following output:
```
array([0., 0., 0., 0., 0.])
```

```python
# add your code here
```

## Task 2
Initialize a vector which contains a zero and negative odd numbers to -8 (inclusively). 

You should get the following output:
```
array([ 0, -2, -4, -6, -8])
```

```python
# add your code here
```

## Task 3
Get the last 3 elements of the given vector.

You should get the following output:
```
array([30, 40, 50])
```

```python
x = np.array([10,20,30,40,50])

# add your code here
```

## Task 4
Find a mistake in the code.

You should get the following output:
```
array([10, 30, 50])
```

```python
x = np.array([10,20,30,40,50])

# add your code here
```

## Task 5
Modify the given vector so that you get the following output:
```
[  10    0   30 1000   50]
```

```python
x = np.array([10,20,30,40,50])

# add your code here
```

## Task 6
Get the maximum number in the given list.

You should get the following output:
```
50
```

```python
x = np.array([10,20,30,40,50])

# add your code here
```

## Task 7
Initialize a matrix which looks like the following example.

You should get the following output:
```
array([[5, 5, 5],
       [1, 2, 3]])
```

```python
# add your code here
```

## Task 8
Initialize the same matrix as in the previous example but you data should be float numbers.

You should get the following output:
```
array([[5., 5., 5.],
       [1., 2., 3.]])
```

```python
# add your code here
```

## Task 9
Initialize a matrix of zeros which has 5 rows and 10 columns.

You should get the following output:
```
array([[0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
       [0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]])
```

```python
# add your code here
```

## Task 10
Initialize an identity matrix with 5 rows.
You should get the following output:
```
array([[1., 0., 0., 0., 0.],
       [0., 1., 0., 0., 0.],
       [0., 0., 1., 0., 0.],
       [0., 0., 0., 1., 0.],
       [0., 0., 0., 0., 1.]])
```

```python
# add your code here
```

## Task 11
Get the first column of the matrix X.

You should get the following output:
```
array([1, 4, 7])
```

```python
X = np.array([[1,2,3], [4,5,6], [7,8,9]])

# add your code here
```

## Task 12
Modify the matrix so that it looks like an example.

You should get the following output:
```
[[ 1  2 10]
 [ 4  5 10]
 [ 7  8 10]]
```

```python
X = np.array([[1,2,3], [4,5,6], [7,8,9]])

# add your code here
```

## Task 13
Get the values from first and the last columns of the second raw of matrix X.

You should get the following output:
```
array([5, 8])
```

```python
X = np.array([[1,2,3, 4], [5,6, 7, 8], [9, 10, 11, 12]])

# add your code here
```

## Task 14
Get the shape of vector x.

You should get the following output:
```
(12,)
```

```python
x = np.arange(1, 13)

# add your code here
```

## Task 15
Create a matrix X from the vector x using reshaping.

You should get the following output:
```
array([[ 1,  2],
       [ 3,  4],
       [ 5,  6],
       [ 7,  8],
       [ 9, 10],
       [11, 12]])
```

```python
# add your code here
```

## Task 16
Flatten the matrix X from the previous task.
You should get the following output:
```
array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12])
```

```python
# add your code here
```

## Task 17
Get a transponse of matrix X from Task 15.

You should get the following output:
```
array([[ 1,  3,  5,  7,  9, 11],
       [ 2,  4,  6,  8, 10, 12]])
```

```python
# add your code here
```

## Task 18
Multiply each row of the matrix X by B.

You should get the following output:
```
array([[ 1,  4],
       [ 3,  8],
       [ 5, 12],
       [ 7, 16],
       [ 9, 20],
       [11, 24]])
```

```python
B = np.array([1, 2])

# add your code here
```

## Task 19

Get a dot product of matrix X and B.

You should get the following output:
```
array([[ 6,  9, 12],
       [12, 19, 26],
       [18, 29, 40],
       [24, 39, 54],
       [30, 49, 68],
       [36, 59, 82]])
```

```python
B = np.arange(6).reshape([2,3])

# add your code here
```

## Task 20
Get a log for each number in matrix X.

You should get the following output:
```
array([[0.        , 0.69314718],
       [1.09861229, 1.38629436],
       [1.60943791, 1.79175947],
       [1.94591015, 2.07944154],
       [2.19722458, 2.30258509],
       [2.39789527, 2.48490665]])
```

```python
# add your code here
```

### You're done! Congratulations
#### There is one bonus task which you could try to solve.


## Bonus
Calculate Z and A:

$$Z = XW+b$$
$$A = tanh(z)$$

You should get the following output for A:
```
array([[0.80265621, 0.91295317, 0.9033375 ],
       [0.5570282 , 0.83529557, 0.6073829 ],
       [0.51714154, 0.80740807, 0.63937959]])
```

```python
X = [[0.74858744, 0.73163723], [0.39287332, 0.25377441], [0.6353768,  0.06158788]]
W = [[0.35263797, 0.13150443, 0.95494651], [0.73678507, 0.6127208,  0.93252179]]
b = [0.30299484, 0.9982467, 0.09294072]

Z = None # replace None with your code
A = None # replace None with your code

A
```
