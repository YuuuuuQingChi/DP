import torch.nn.functional as F
import torch

input = torch.tensor(
    [
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
        [0, 0, 0, 0, 1],
    ]
)
kernel = torch.tensor([[0, 1, 0], [0, 1, 0], [0, 1, 0]])
print(input.shape)
print(kernel.shape)
input = torch.reshape(input,(1,1,5,5))
kernel = torch.reshape(kernel,(1,1,3,3))
process_input = F.conv2d(input,kernel,stride=1,padding= 1)
print(process_input)
