import torch

cuttoffs = [155.153,164.27,170.256,173.256,178.195,182.223,184.147,185.223,186.255,187.283,189.211,191.234,193.262,195.222,196.25,197.225,197.326,198.27199]

model = torch.load('model.pt')

model = (model[0], model[1], model[2], model[3], model[4], cuttoffs)
breakpoint()
torch.save(model, 'model.pt')