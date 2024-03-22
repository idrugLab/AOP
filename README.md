# AOP-Predict
Provides a script for predicting the antioxidant activity of molecules.

# Installation
1. Clone the repository and navigate to the root directory
    ```bash
    git clone https://github.com/idruglab/AOP-Predict.git
    cd AOP-Predict
    ```

1. Install Package
    > Recommended python version is 3.7.
    ```bash
    conda install -c openbabel openbabel
    pip install -r requirements.txt
    ```

# Usage
## Single molecule
1. Prepare the SMILES string of the molecule
1. Predict
    ```bash
    python predict.py -s "Oc1ccc(cc1)\C=C\c2cc(O)cc(O)c2" -o example/single_output.csv
    ```
## Several molecules
1. Prepare input file

    In the input file, the contents are as follows:

    ```
    SMILES
    C1=CC(=C(C=C1[C@H]2[C@@H](CC3=C(C=C(C=C3O2)O)O)O)O)O
    Oc1ccc(cc1)\C=C\c2cc(O)cc(O)c2
    ...
    ```
1. Predict
    ```bash
    python predict.py -i example/input.csv -o example/output.csv
    ```
## Specify which antioxidant assays are predicted
1. All
    ```bash
    python predict.py -i example/input.csv -l all -o example/output.csv
    ```
1. Custom
    ```bash
    python predict.py -i example/input.csv -l ABTS DPPH -o example/output.csv
    ```
> `-s` specifies the SMILES string of the molecule, `-i`specifies the path to the input file, `-o` specifies the path to the output file，`-l` specifies which antioxidant assays are predicted，default is `all`.

