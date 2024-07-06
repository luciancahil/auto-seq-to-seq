# auto-seq-to-seq

A sequence to sequence model that focuses on generating new molecules.

## Environment

To setup the environment, run the environment.yml file with:

````
conda env create -f environment.yml
````

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

Once you've run "main.py" the model should be saved as "model.pt". Once that file has been created, run "Interpolate.py". This will then cause the molecule to begin attemping to generate new molecules using the molecules in the "data/seeds.txt" file.

If you wish to use different seeds, alter the seeds.txt file, formatted with the exact way as files in chem.txt.


## Biased generation

The values provided in chem.txt and seeds.txt are used to organize the molecules. This document will discuss molecular weights, but will work for other continuous values that molecuels can have.

This is to help bias the generation process. By including a Convolutional Neural Network that operates on the latent space, the network is trained to organize the latent space in a way that places molecules with similar moleculear weights "near" each other. Thus, when performing interpolation with seed molecules that all have molecular weights around a given value will make the network more likely to generate molecules with that weight.

To perform interpolation, edit "data/seeds.txt" to include seeds that have molecular weights close to your goal. Then, type the following into console from the main folder:

````
Python Interpolate.py 0
````