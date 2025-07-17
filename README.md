# Order Independent Optimal Classification Trees (OCT) Repository



## Files

- **OrderIndependentOCT.py**: Main script that contains the function which selects and runs OCT methods via Papermill.
- **setup.py**: Builds the Cython extension (`Pricing_Branching`).
- **Requirements.txt**: Lists all dependencies.
- **Datasets/**: CSVs organized as `<dataset>/fold=<n>_train.csv` and `fold=<n>_test.csv`.
- **Notebooks/**:
  - **OrderIndependentOCT\_ExampleNotebook.ipynb**: Runnable end‑to‑end example (method can be chosen using the selection parameter, amongst CompactOCT -1- , POCT -2-, BPOCT -3-, or left as default -0- which uses the hybrid method described in the paper, OrderIndependentOCT). Strongly recommended to be the first file to experiment with as it contains further installation and output interpretation directions. 
  - **OCT\_OCT\_OrderedModel\_Compact.ipynb**: Runnable ordered CompactOCT formulation, to compare against the order-independent version,  parameters adjusted at the top cell (Section 3.2).
  - **OCT\_IP\_Cuts.ipynb**: Runnable notebook for the experiments with tightening cuts proposed in the manuscript, parameters adjusted at the top cell (Section 3.6).
  - **OCT\_FairnessEpsilonCons.ipynb**: Runnable notebook containining the $\epsilon$-constraint framework for fairness experiments (Section 4).
  - **Supporting** (invoked by the main notebooks):\
    `OCT_OrderIndependentModel_Compact.ipynb`,\
    `OCT_IP.ipynb`,\
    `OCT_BnP.ipynb`,\
    `OCT_Fairness_IP.ipynb`.

## Requirements

Install dependencies via `Requirements.txt`, or directly by running the setup cells in the example notebook file.


## Installation

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Example Notebook Execution

### OrderIndependentOCT\_ExampleNotebook.ipynb

In the example notebook, go over existing datasets or add new ones (ensure they follow the `Datasets/<name>/fold=<n>_train.csv` structure):

```python
import pandas as pd
import numpy as np
from OrderIndependentOCT import OrderIndependentOCT

# Built-in datasets
data_list = [
    'adult','agaricus-lepiota','balance-scale','banknote_authentication',
    'car-evaluation','diabetes','haberman','kr-vs-kp','monks-1','monks-2',
    'monks-3','nursery','seismic-bumps','tae','tic-tac-toe','titanic',
    'wdbc','wine','NHPA'
]

fold_list = list(range(11))
depth_list = [2, 3, 4]

# Example run
fold    = 1
depth   = 2
dataset = 'kr-vs-kp'
OrderIndependentOCT(fold, dataset, depth)
```

 - Pass selection=0 for hybrid rule or 1/2/3 to force a specific OCT variant.
 - After running, inspect:
     * Datasets/<dataset>/fold=```<fold><output><depth>```.txt  ← summary metrics for the method chosen by the hybrid approach or specified by the user, will also be printed on the screen
     * Datasets/<dataset>/fold=<fold>_DecisionRules```_<tag>_<depth>```.txt  ← decision rules
     * RunNotebooks/OCT```<Variant>[…]```.ipynb  ← full notebook output

### OCT\_FairnessEpsilonCons.ipynb

At the top of the fairness‐constraint notebook, set parameters:

```python
parameters = {
    'fold': 1,
    'd': 2,
    'epsilon': 1,            # initial fairness bound
}

data_list = ['nursery']      # select datasets

sensitive_features = [5]     # e.g. gender column index, can be a list of features
```

