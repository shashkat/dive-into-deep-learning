import torch

# some useful functions for performing convolution on image using kernel
def corr2d(X, K): 
	"""Compute 2D cross-correlation."""
	h, w = K.shape
	Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
	for i in range(Y.shape[0]):
		for j in range(Y.shape[1]):
			Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
	return Y

def corr2d_multi_in(X, K):
    # Iterate through the 0th dimension (channel) of K first, then add them up
    return sum(corr2d(x, k) for x, k in zip(X, K))

def corr2d_multi_in_out(X, K):
    # Iterate through the 0th dimension of K, and each time, perform
    # cross-correlation operations with input X. All of the results are
    # stacked together
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)

def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    # Matrix multiplication in the fully connected layer
    Y = torch.matmul(K, X)
    return Y.reshape((c_o, h, w))

temp = 0
for i in range(10000):
	X = torch.normal(0, 1, (3, 3, 3))
	K = torch.normal(0, 1, (2, 3, 1, 1))
	Y1 = corr2d_multi_in_out_1x1(X, K)
	Y2 = corr2d_multi_in_out(X, K)
	temp += float(torch.abs(Y1 - Y2).sum())
	if temp != 0:
		print(f'temp is non-zero in iteration {i}')
		break
	# assert float(torch.abs(Y1 - Y2).sum()) < 1e-6

# CONCLUSION:
# Even with 10k repetitions, it never happens that there is an inequality between the two 
# tensors. This indicates that the tensors are exactly the same, and that is because the same 
# exact operations are being performed in the same exact order, and hence small discrepancies 
# due to floating point precision details don't happen here.







# old stuff
# temp3 = torch.normal(0, 1, (1,3)) # one pixel of image, all ci channels
# temp4 = torch.normal(0, 1, (3,1)) # kernel values for all ci one co
# temp5 = torch.matmul(temp3, temp4)
# temp6 = sum(corr2d(x, k) for x, k in zip(temp3.reshape(3,1,1), temp4.reshape(3,1,1)))
# temp5 == temp6

# temp3[0][0]*temp4[0][0] + temp3[0][1]*temp4[1][0] + temp3[0][2]*temp4[2][0]

# p1 = corr2d(temp3.reshape(3,1,1)[0], temp4.reshape(3,1,1)[0])
# p2 = corr2d(temp3.reshape(3,1,1)[1], temp4.reshape(3,1,1)[1])
# p3 = corr2d(temp3.reshape(3,1,1)[2], temp4.reshape(3,1,1)[2])

# p1 == temp3[0][0]*temp4[0][0]
# p2 == temp3[0][1]*temp4[1][0]
# p3 == temp3[0][2]*temp4[2][0]

# p1 + p2 + p3
# sum(p for p in [p1, p2, p3])

# torch.zeros((2,2)).dtype

# temp7 = corr2d(torch.normal(0,1,(2,2)), torch.normal(0,1,(2,2)))
# X = torch.normal(0,1,(2,2))
# K = torch.normal(0,1,(2,2))
# temp8 = corr2d(X, K)
# temp8.dtype

# temp3 = torch.normal(0, 1, (1,1))
# temp4 = torch.normal(0, 1, (1,1))
# temp5 = torch.matmul(temp3, temp4)
# temp6 = (temp3*temp4).sum()
# temp5 == temp6




# temp2 = (i+2 for i in [1,2,3])
# for j in temp2:
# 	print(j)
# type(temp2)
# next(temp2)

# sum()

# Y1
# Y2
# Y1 == Y2
# float(torch.abs(Y1 - Y2).sum())
# type(Y1)



