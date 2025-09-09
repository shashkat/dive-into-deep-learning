# just playing around a bit with specifying the arguments signature of lambda functions in 
# different ways as found that in the initial implementation from scratch of an LSTM

from torch import nn
import torch

temp = torch.tensor([[1,2,3], [1,2,3], [1,2,3]], dtype = torch.float)

nn.Parameter(temp)


# now making a lambda func
temp_func = lambda *arg: nn.Parameter(torch.randn(*arg))

temp_func(5, 4)

temp1, temp2 = temp

temp1 = torch.tensor([[1,2,3], [4,5,6]])
temp2 = torch.tensor([7,8,9])

temp1 = torch.randn((2,4,3))
temp2 = torch.randn((2,3))
temp1 + temp2

temp2.dim()
temp1.dim()


temp1.shape
temp2.shape

temp3 = []
temp3.append(torch.randn(1,2))
type(temp3)


