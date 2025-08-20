from rdkit.Chem import rdMolDescriptors
from rdkit import Chem, RDLogger 


file = open("newMolecules.txt", mode = "r")


for line in file:
    smiles = line.strip()
    mol = Chem.MolFromSmiles(smiles)
    print("{}, {}".format(smiles, rdMolDescriptors.CalcExactMolWt(mol)))