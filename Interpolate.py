import sys
import torch
from utils import DEVICE, MAX_LENGTH, read_single_lang, filterWords, get_data_tensors
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

def get_latents(model, x, y_s):
    # encoder
    x = model[0](x, y_s)
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
            decoded_chars.append(output_lang.index2char[idx.item()])
        
        smiles.append("".join(decoded_chars))
    
    return smiles


def get_interpolated_latents(latents, y_s):
    interpolated_latents = []
    interpolated_hidden_latents = []
    interpolated_ys = []
    for i in range(len(latents[0]) - 1):
        for j in range(i + 1, len(latents[0])):
            # pairs of output_latents
            l_1 = latents[0][i,:,:].unsqueeze(dim = 0)
            l_2 = latents[0][j,:,:].unsqueeze(dim = 0)
            interpolated_latent = interpolate(l_1, l_2)
            l_1 = latents[1][i,:,:].unsqueeze(dim = 0)
            l_2 = latents[1][j,:,:].unsqueeze(dim = 0)
            interpolated_hidden_latent = interpolate(l_1, l_2)

            y_1 = y_s[i]
            y_2 = y_s[j]

            # add the first element once no matter what
            interpolated_latents += interpolated_latent
            interpolated_hidden_latents += interpolated_hidden_latent
            interpolated_ys += [y_1] * (NUM_POINTS - 1)

            # if the y's are diferent, add another copy o both latents, as well as the second y
            if(y_1.item() != y_2.item()):
                interpolated_latents += interpolated_latent
                interpolated_hidden_latents += interpolated_hidden_latent
                interpolated_ys += [y_2] * (NUM_POINTS - 1)
    
    return interpolated_latents, interpolated_hidden_latents, interpolated_ys

def decode_to_smiles(input_latent, hidden_latent, ys_latent):
    input_latent = model[1].decode(input_latent, ys_latent)
    hidden_latent= model[2].decode(hidden_latent, ys_latent)
    # go from batchsize x 1 x hiddenshape to 1 x batchsize x hidden_shape
    hidden_latent=hidden_latent.squeeze().unsqueeze(dim = 0)
    decoder_outputs,_,_ = model[3](input_latent, hidden_latent)
    _, topi = decoder_outputs.topk(1)
    return idx_to_smiles(topi)


def write_smiles_to_file(inputs, filename):
    valid_smiles = set()
    all_smiles  = []
    masses = []
    num_valid = 0
    all = len(inputs)
    file = open(filename, mode='w')
    for smiles in inputs:
        smiles, weight = repair_smiles(smiles)
        if (weight != -1) and (smiles not in seed_smiles):
            string = smiles + "," + str(weight)
            valid_smiles.add(string)
            num_valid += 1
        
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

    file.write("There were {} valid smiles, out of {}".format(num_valid, all))
#MAIN
try:
    NUM_SEEDS = int(sys.argv[1])
except(Exception):
    NUM_SEEDS = 6

# if a positive number, we want that bin. If -1, we will target the largest bin.
target_bin = -1


input, _, input_lang, output_lang, y_s, num_bins, _ = get_data_tensors('chem')
input = input[:NUM_SEEDS]
y_s = y_s[:NUM_SEEDS]
seed_smiles = idx_to_smiles(input)
model = torch.load(model_path, map_location=device)
print("Starting interpolation")
# first one is the output, second is the hidden
latents = get_latents(model, input, y_s)
interpolated_latents, interpolated_hidden_latents, interpolated_ys = get_interpolated_latents(latents, y_s)

        

        
input_latents_inter = torch.cat(interpolated_latents, dim = 0)
hidden_latents_inter = torch.cat(interpolated_hidden_latents, dim = 0)
ys_latents_inter = torch.stack(interpolated_ys)


if(target_bin == -1):
    target_bin = model[1].embed_condition.num_embeddings - 1

input_latents_gen = torch.randn_like(input_latents_inter)
hidden_latents_gen = torch.randn_like(hidden_latents_inter)
ys_latent_gen = torch.tensor([target_bin] * len(ys_latents_inter))


# decode interpolated smiles
decoded_chems_inter = decode_to_smiles(input_latents_inter, hidden_latents_inter, ys_latents_inter)
write_smiles_to_file(decoded_chems_inter, "Interpolated Smiles.txt")


decoded_chems_gen = decode_to_smiles(input_latents_gen, hidden_latents_gen, ys_latent_gen)
write_smiles_to_file(decoded_chems_gen, "Generated Smiles.txt")
print('done!')
