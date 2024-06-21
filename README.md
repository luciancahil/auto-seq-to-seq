# auto-seq-to-seq

A sequence to sequence model that focuses on generating new molecules.

## Data Formating
By default, the model will be trained according to data in "data/chem.txt".

To make your own data, eitehr edit that file, or add a new file in that folder, and change the "filename" field in main.py.

New data must have the follwing structure:

1. Each line is a new element.
1. Each line has one and only one comma.
1. The information before the comma is a SMILES string.
1. The information after the comma is some numerical value.

For example, the first line of chem.txt is:
````
CCOC(=O)[C@@H](N)CCC#N,156.185
````

In this case, the number at the end represents the molecular weight. Your data doesn't have to be of this type, but each value should be all of the same type. For instance, don't have half your numbers be logP and the other half molecular weight, but all of them being logP values would work fine.

## Training


Adjust hyperparameters in utils.py if desired. In particular look over "MAX_LENGTH", which control the largest possible string that can be encoded and decoded.

Once the hyperparameters are satisfactory. Run "python main.py"

## Interpolation.

Once you've run "main.py" the model should be saved as "model.pt". Once that file has been created, run "Interpolate.py". This will then cause the molecule to begin attemping to generate new molecules using the first few molecules in chem.txt as a starting seed.

If you wish to change the number of seeds used, add a number at the end. For example, if you want 10 seeds, run "python Interpolate.py 10"