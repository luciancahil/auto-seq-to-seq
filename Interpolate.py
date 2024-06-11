import sys
import torch
from utils import DEVICE, MAX_LENGTH, read_single_lang, filterWords, get_data_tensors
import deepchem as dc
import torch.nn.functional as F
device = DEVICE
import numpy as np
from Lang import EOS_token

model_path = 'model.pt'

NUM_POINTS = 5
featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    

MAX_MOLECULE_SIZE = MAX_LENGTH

def get_latents(model, x):
    x = model[0](x)
    output_latent, _ = model[1].encode(x[0])
    hidden_latent, _ = model[2].encode(x[1])
    return output_latent, hidden_latent

#MAIN

def interpolate(tensor_one, tensor_two, numpoints = 5):
    alphas = [1 / i for i in range(1, numpoints)]
    interpolated_latents = [torch.add(torch.sub(tensor_one, tensor_two, alpha = a), tensor_two) for a in alphas]
    return interpolated_latents

input, _, input_lang, output_lang = get_data_tensors('chem')
input = input[:6]
model = torch.load(model_path, map_location=device)
print("Starting interpolation")
# first one is the output, second is the hidden
latents = get_latents(model, input)
interpolated_latents = []
interpolated_hidden_latents = []

for i in range(len(latents[0]) - 1):
    for j in range(i + 1, len(latents[0])):
        # pairs of output_latents
        l_1 = latents[0][i,:,:].unsqueeze(dim = 0)
        l_2 = latents[0][j,:,:].unsqueeze(dim = 0)
        interpolated_latents += interpolate(l_1, l_2)
        l_1 = latents[1][0,i,:].unsqueeze(dim = 0)
        l_2 = latents[1][0,j,:].unsqueeze(dim = 0)
        interpolated_hidden_latents += interpolate(l_1, l_2)


input_latents = torch.cat(interpolated_latents, dim = 0)
hidden_latents = torch.cat(interpolated_hidden_latents, dim = 0).unsqueeze(dim = 0)

# the decoder
input_latents = model[1].decode(input_latents)
hidden_latents= model[2].decode(hidden_latents)

decoder_outputs,_,_ = model[3](input_latents, hidden_latents)
_, topi = decoder_outputs.topk(1)
decoded_ids = topi.squeeze()
decoded_chems = []

for decoded_ids in topi.squeeze():
    decoded_chars = []
    for idx in decoded_ids:
        if idx.item() == EOS_token:
            #decoded_chars.append('<EOS>')
            break
        decoded_chars.append(output_lang.index2char[idx.item()])
    
    decoded_chems.append("".join(decoded_chars))

for chem in decoded_chems:
    print(chem)

#data = [proces_smiles(seed) for seed in seeds if proces_smiles(seed) is not None]
#model.interpolate(data, NUM_POINTS)