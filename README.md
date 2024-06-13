# auto-seq-to-seq

A sequence to sequence model that focuses on generating new molecules.

## Training

Create a "data" folder, and add a list of SMILES Strings into a file called "chem.txt".

Adjust hyperparameters in utils.py if desired. In particular look over "MAX_LENGTH", which control the largest possible string that can be encoded and decoded.

Once the hyperparameters are satisfactory. Run "python main.py"

## Interpolation.

Once you've run "main.py" the model should be saved as "model.pt". Once that file has been created, run "Interpolate.py". This will then cause the molecule to begin attemping to generate new molecules using the first few molecules in chem.txt as a starting seed.

If you wish to change the number of seeds used, add a number at the end. For example, if you want 10 seeds, run "python Interpolate.py 10"