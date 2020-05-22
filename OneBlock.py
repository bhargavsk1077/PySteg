
# Singular-value decomposition
import numpy 
from numpy import array
from scipy.linalg import svd
import imageio
import os
import sys
import math

image_path = f"input/{sys.argv[1]}"
block_size = 4
print(f"\nblock size is {block_size}\n")
cols_protected = 1
print(f"\nnumber of protected columns is {cols_protected}\n")
message = "hello"

def binarize_message(message):
    binary_message = ''.join(format(ord(x), 'b') for x in message)
    binary_list = []
    for character in binary_message:
        if character == '0':
            binary_list.append(-1)
        else:
            binary_list.append(1)

    return binary_list

def format_image(image): 
    image = image[:,:,0]
    return image
    
def computeSVD(image_block):
        """compute the SVD of a single image block (will add input later)"""

        #print(image_block)
        U, S, VT = numpy.linalg.svd(image_block)

        # create blank m x n matrix
        Sigma = numpy.zeros((U.shape[1], VT.shape[0]))

         

        for i in range(0, S.shape[0]):
            for j in range(0, S.shape[0]):
                if i == j:
                    Sigma[i,j] = S[i]

        
        block = U.dot(Sigma.dot(VT))
        
        return [U, Sigma, VT]

def make_orthogonal(u,m,cols_protected,block_size):
    eqs = []
    X = []
    for i in range(0,m):
        total = 0
        eq = []
        for j in range(0,block_size-m):
            total += u[j,i]*u[j,m]
        for j in range(block_size-m,block_size):
            eq.append(u[j,i])

        #eq.append(-1*total)
        X.append(-1*total)
        eqs.append(eq)
    
    sol = solve_eqs(eqs,X)
    z = 0
    for y in range(block_size-m,block_size):
        u[y,m] = sol[z]
        z += 1
    
    return u

def solve_eqs(eqs,X):
    A = numpy.array(eqs)
    #inv_A = numpy.linalg.inv(A)
    B = numpy.array(X)
    X = numpy.linalg.inv(A).dot(B)
    return X

msg = binarize_message(message)
print(f"\nmessage in binary {msg}\n")

bpb =  int(((block_size-cols_protected-1)*(block_size-cols_protected))/2)
print(f"\nthe number of bits in which message can be embedded {bpb}\n")

print("\nmessage bits that can be embedded ")
print(msg[0:bpb])

ebd_msg = msg[0:bpb]

img = imageio.imread(image_path)
image = img.astype(numpy.int32)

img_array = format_image(image)
first_block = img_array[0:block_size,0:block_size]
A = first_block

print("\nthe first block in the image")
print(" ")
print(A)

#U, s, VT = svd(A)
U,s,VT = computeSVD(A)

V=numpy.matrix.transpose(VT)

print("\nthe Matrix U \n")
print(U)
print("\nthe Matrix S\n")
print(s)
print("\nthe matrix VT \n")
print(VT)
print("\nthe matrix V\n ")
print(V)

U1 = U
V1 = V
s1 = s
for k in range(0, block_size):
    if U1[0,k]<0:
        U1[0:block_size,k]*=-1
        V1[0:block_size,k]*=-1

print("\nU and V before embedding but after convewrting to normal form \n")
print(U1)
print(" ")
print(V1)
print(" ")

lim = block_size-cols_protected-1
x=0
y = lim
col_lim = []
while(y != 0):
    col_lim.append(y)
    y=y-1
z=0

for j in range(cols_protected,block_size-1):
    for i in range(1,col_lim[x]+1):
        U1[j,i]=ebd_msg[z]*math.fabs(U1[j,i])
        z+=1
    x+=1
    U1 = make_orthogonal(U1,j,cols_protected,block_size)
    norm = math.sqrt(numpy.dot(U1[0:block_size,j],U1[0:block_size,j]))
    for p in range(0, block_size):
        U1[p,j] /= norm 

print("\nembedded U")
print(f"\n{U1}\n")

avg_dist = (s[1,1]+s[block_size-1,block_size-1])/block_size
for k in range(2,block_size):
    s1[2,2]=s1[1,1]-(k*avg_dist)

V1T = numpy.matrix.transpose(V1)

A1 = numpy.round(U1.dot(s1.dot(V1T)))


for x in range(0,block_size):
    for y in range(0, block_size):
        if A1[x,y]>255:
            A1[x,y]=255
        elif A1[x,y]<0:
            A1[x,y]=0

print("\n embedded A")
print(f"\n{A1}\n")

