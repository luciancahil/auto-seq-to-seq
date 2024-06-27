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
from GenerationUtil import *
RDLogger.DisableLog('rdApp.*')

model_path = 'model.pt'

NUM_POINTS = 5
featurizer = dc.feat.MolGraphConvFeaturizer(use_edges=True)

    

MAX_MOLECULE_SIZE = MAX_LENGTH
#MAIN


# objects in model
# 0: The RNN encoder
# 1: The variator for encoder outputs
# 2: The variator for the encoder hidden state
# 3: The RNN Decoder
# 4: The Linear Regression object
# 5: The Language object
model = torch.load(model_path, map_location=device)
lang = model[5]

seeds = open("data/seeds.txt")
seeds = seeds.readlines()

# turn seeds into
input = [tensorFromSentence(lang, line.split(',')[0]).squeeze() for line in seeds]
input = torch.stack(input)

seed_smiles = idx_to_smiles(input, lang)
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
decoded_chems_inter = decode_to_smiles(input_latents_inter, hidden_latents_inter, model, lang)
write_smiles_to_file(decoded_chems_inter, "Interpolated_Smiles.txt", seed_smiles)

# decode generated smiles
decoded_chems_gen = decode_to_smiles(input_latents_gen, hidden_latents_gen, model, lang)
write_smiles_to_file(decoded_chems_gen, "Generated_Smiles.txt", seed_smiles)
print('Done! Saved new SMILES to "Interpolated_Smiles.txt" and "Generated_Smiles.txt"')
