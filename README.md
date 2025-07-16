# Order Independent Optimal Classification Trees (OCT) Repository



## Files

- **OrderIndependentOCT.py**: Main script (CLI) that contains the function which selects and runs OCT methods via Papermill.
- **setup.py**: Builds the Cython extension (`Pricing_Branching`).
- **Requirements.txt**: Lists all dependencies.
- **Datasets/**: CSVs organized as `<dataset>/fold=<n>_train.csv` and `fold=<n>_test.csv`.
- **Notebooks/**:
  - **OrderIndependentOCT\_ExampleNotebook.ipynb**: Runnable end‑to‑end example (method can be chosen using the selection parameter, amongst CompactOCT -1- , POCT-2-, BPOCT-3-, or left as default -0- which uses the hybrid method described in the paper, OrderIndependentOCT) .
  - **OCT\_FairnessEpsilonCons.ipynb**: Runnable fairness‑constrained experiments.
  - **OCT\_IP\_Cuts.ipynb**: Runnable notebook for the experiments with cuts, parameters adjusted at the top.
  - **Supporting** (invoked by the main notebooks):\
    `OCT_OrderIndependentModel_Compact.ipynb`,\
    `OCT_OrderedModel_Compact.ipynb`,\
    `OCT_IP.ipynb`,\
    `OCT_BnP.ipynb`,\
    `OCT_Fairness_IP.ipynb`.

## Requirements

Install dependencies via `Requirements.txt`, or directly:

```bash
pip install Cython>=0.29 numpy pandas scipy scikit-learn gurobipy>=10.0.3 papermill more-itertools networkx matplotlib
```

## Installation

```bash
pip install -r requirements.txt
python setup.py build_ext --inplace
```

## Example Notebook Execution

### OrderIndependentOCT\_ExampleNotebook.ipynb

In the example notebook, go over preferred datasets or add new ones (ensure they follow the `Datasets/<name>/fold=<n>_train.csv` structure):

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
dataset = 'monks-1'
OrderIndependentOCT(fold, dataset, depth)
```

### OCT\_FairnessEpsilonCons.ipynb

At the top of the fairness‐constraint notebook, set parameters:

```python
parameters = {
    'fold': 1,
    'd': 2,
    'epsilon': 1,            # initial fairness bound
}

data_list = ['nursery']      # select datasets

sensitive_features = [5]     # e.g. gender column index, can be a list
```

