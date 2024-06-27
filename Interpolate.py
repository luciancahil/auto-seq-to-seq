import sys
import torch
from utils import DEVICE, MAX_LENGTH, get_data_tensors, tensorFromSentence
import deepchem as dc
import torch.nn.functional as F
device = DEVICE
import numpy as np
from Lang import EOS_token
from rdkit import Chem, RDLogger 
from rdkit.Chem import rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

model_path = 'model.pt'

NUM_POINTS = 5
featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    

MAX_MOLECULE_SIZE = MAX_LENGTH

def get_latents(model, x):
    # encoder
    x = model[0](x)
    # first hal of the variators
    output_latent, _ = model[1].encode(x[0])
    hidden_latent, _ = model[2].encode(x[1])
    return output_latent, hidden_latent

def repair_smiles(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is not None:
        return smiles, rdMolDescriptors.CalcExactMolWt(mol)

    possible_last_chars = [')', '1', ']']
    
    for char in possible_last_chars:
        modified_smiles = smiles + char
        mol = Chem.MolFromSmiles(modified_smiles)

        if mol is not None:
            return modified_smiles, rdMolDescriptors.CalcExactMolWt(mol)
    
    return smiles, -1

def interpolate(tensor_one, tensor_two, numpoints = 5):
    alphas = [1 / i for i in range(1, numpoints)]
    interpolated_latents = [torch.add(torch.sub(tensor_one, tensor_two, alpha = a), tensor_two) for a in alphas]
    return interpolated_latents

def idx_to_smiles(indicies):
    smiles = []
    for decoded_ids in indicies:
        decoded_chars = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                #decoded_chars.append('<EOS>')
                break
            decoded_chars.append(lang.index2char[idx.item()])
        
        smiles.append("".join(decoded_chars))
    
    return smiles


def get_interpolated_latents(latents):
    interpolated_latents = []
    interpolated_hidden_latents = []
    for i in range(len(latents[0]) - 1):
        for j in range(i + 1, len(latents[0])):
            # pairs of output_latents
            l_1 = latents[0][i,:,:].unsqueeze(dim = 0)
            l_2 = latents[0][j,:,:].unsqueeze(dim = 0)
            interpolated_latent = interpolate(l_1, l_2)
            l_1 = latents[1][i,:,:].unsqueeze(dim = 0)
            l_2 = latents[1][j,:,:].unsqueeze(dim = 0)
            interpolated_hidden_latent = interpolate(l_1, l_2)
          

            # add the first element once no matter what
            interpolated_latents += interpolated_latent
            interpolated_hidden_latents += interpolated_hidden_latent
    
    return interpolated_latents, interpolated_hidden_latents

def decode_to_smiles(input_latent, hidden_latent):
    input_latent = model[1].decode(input_latent)
    hidden_latent= model[2].decode(hidden_latent)
    # go from batchsize x 1 x hiddenshape to 1 x batchsize x hidden_shape
    hidden_latent=hidden_latent.squeeze().unsqueeze(dim = 0)
    decoder_outputs,_,_ = model[3](input_latent, hidden_latent)
    _, topi = decoder_outputs.topk(1)
    return idx_to_smiles(topi)


def write_smiles_to_file(inputs, filename):
    valid_smiles = set()
    all_smiles  = []
    masses = []
    all = len(inputs)
    file = open(filename, mode='w')
    for smiles in inputs:
        smiles, weight = repair_smiles(smiles)
        if (weight != -1) and (smiles not in seed_smiles) and not(smiles[-3:] == "SOS" and smiles[0:3] == "SOS"):
            string = smiles + "," + str(weight)
            valid_smiles.add(string)
        
        all_smiles.append(smiles)

    file.write("Seeds:\n")
    for smile in seed_smiles:
        file.write(smile)
        file.write('\n')

    file.write("\n\nValid Smiles:\n")

    for smile in valid_smiles:
        file.write(smile)
        file.write('\n')

    file.write("\n\nAll Smiles:\n")
    for smile in all_smiles:
        file.write(smile)
        file.write('\n')

    file.write("There were {} valid smiles, out of {}".format(len(valid_smiles), all))
#MAIN


# objects:
# 0: The RNN encoder
# 1: The variator for encoder outputs
# 2: The variator for the encoder hidden state
# 3: The RNN Decoder
# 4: The Language object
# 5: The cutof points that tell wether a given value goes in a certain bin.
model = torch.load(model_path, map_location=device)
lang = model[5]

seeds = open("data/seeds.txt")
seeds = seeds.readlines()

# turn seeds into
input = [tensorFromSentence(lang, line.split(',')[0]).squeeze() for line in seeds]
input = torch.stack(input)

seed_smiles = idx_to_smiles(input)
model = torch.load(model_path, map_location=device)
print("Starting interpolation")

latents = get_latents(model, input)
# first one is the output, second is the hidden
interpolated_latents, interpolated_hidden_latents = get_interpolated_latents(latents)

        

        
input_latents_inter = torch.cat(interpolated_latents, dim = 0)
hidden_latents_inter = torch.cat(interpolated_hidden_latents, dim = 0)



input_latents_gen = torch.randn_like(input_latents_inter)
hidden_latents_gen = torch.randn_like(hidden_latents_inter)



# decode interpolated smiles
decoded_chems_inter = decode_to_smiles(input_latents_inter, hidden_latents_inter)
write_smiles_to_file(decoded_chems_inter, "Interpolated_Smiles.txt")


decoded_chems_gen = decode_to_smiles(input_latents_gen, hidden_latents_gen)
write_smiles_to_file(decoded_chems_gen, "Generated_Smiles.txt")
print('Done! Saved new SMILES to "Interpolated_Smiles.txt" and "Generated_Smiles.txt"')
