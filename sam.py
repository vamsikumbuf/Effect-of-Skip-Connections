import torch
import torch.nn as nn
import argparse

torch.manual_seed(123)

parser = argparse.ArgumentParser()
parser.add_argument("--block_layers", type=int, default=1)
parser.add_argument("--blocks", type=int, default=5)

args = parser.parse_args()

class FeedForwardBlock(nn.Module):

    def __init__(self, input_dim ):
        super().__init__()
        self.linear_layer = nn.Linear(input_dim, input_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.relu(self.linear_layer(x))

class DNN(nn.Module):
    def __init__(self, input_dim, use_shortcut = False, block_layers = 1, blocks = 5):

        super().__init__()
        self.input_dim = input_dim
        self.use_shortcut = use_shortcut
        self.layers = nn.ModuleList([
            nn.Sequential(
                *[FeedForwardBlock(input_dim) for _ in range(block_layers)]
                )
            for _ in range(blocks)])
        self.final_layer = nn.Linear(input_dim, 1)

    def forward(self, x):

        for layer in self.layers:
            layer_output = layer(x)
            if self.use_shortcut:
                x = x + layer_output
            else:
                x = layer_output

        x = self.final_layer(x)

        return x

def get_gradients(model: nn.Module, x):

    output = model(x)
    target = torch.tensor([[1.]])
    
    loss = nn.MSELoss()
    loss = loss(output, target)

    loss.backward()
    dct = {}
    
    for name, param in model.named_parameters():
        
        if "weight" in name:
            # print(f"{name} has gradient mean of {param.grad.abs().mean().item()}")
            dct[name] = param.grad.abs().mean().item()

    return dct

sample_input = torch.tensor([[1., 0., -1.]])

model = DNN(3, block_layers=args.block_layers, blocks=args.blocks)

no_skip_grads = get_gradients(model, sample_input)

skip_model = DNN(3, use_shortcut=True, block_layers=args.block_layers, blocks=args.blocks)
skip_grads = get_gradients(skip_model, sample_input)

print("Following are the changes in grads when we make use of skip connections: ")
for key in list(skip_grads.keys()):
    print(f"{key} : {no_skip_grads[key]:.8f} => {skip_grads[key]:.8f}")
