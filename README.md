# AOP: Antioxidant Activity Predictor
This repository offers a script for the prediction of antioxidant activity in molecules.
## Installation
1. Clone the repository and navigate to the root directory:
    ```bash
    git clone https://github.com/idruglab/AOP.git
    cd AOP-Predict
    ```
2. Install the required packages:
    - Recommended Python version: `3.7.10`
    - Use Conda to install OpenBabel:
      ```bash
      conda install -c openbabel openbabel
      ```
    - Install other dependencies with `pip`:
      ```bash
      pip install -r requirements.txt
      ```
## Usage
### Predict for a Single Molecule
1. Obtain the SMILES string of the molecule.
2. Run the prediction command:
    ```bash
    python predict.py -s "Oc1ccc(cc1)\C=C\c2cc(O)cc(O)c2" -o example/single_output.csv
    ```
### Predict for Multiple Molecules
1. Create an input file with the following format:
    ```
    SMILES
    C1=CC(=C(C=C1[C@H]2[C@@H](CC3=C(C=C(C=C3O2)O)O)O)O)O
    Oc1ccc(cc1)\C=C\c2cc(O)cc(O)c2
    ...
    ```
2. Run the prediction command:
    ```bash
    python predict.py -i example/input.csv -o example/output.csv
    ```
### Select Antioxidant Assays for Prediction
- Predict for all assays:
    ```bash
    python predict.py -i example/input.csv -l all -o example/output.csv
    ```
- Predict for custom assays:
    ```bash
    python predict.py -i example/input.csv -l ABTS DPPH -o example/output.csv
    ```
### Command Line Options
- `-s`: Specifies the SMILES string of the molecule.
- `-i`: Specifies the path to the input file.
- `-o`: Specifies the path to the output file.
- `-l`: Specifies the antioxidant assays to predict. Default is `all`.
## Training
1. Prepare the datasets in the `data` directory.
2. Execute the training script:
    ```python
    python Class_hyperopt-MT.py
    ```