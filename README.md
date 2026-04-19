# Order Independent Optimal Classification Trees (OCT) Repository

<div align="center">
  <h2>📹 Walkthrough & Reproduction Videos</h2>
  <p><i>Expand any section below to view a step-by-step video demonstration of our environment setup and table-by-table reproduction.</i></p>
</div>

<details>
  <summary><b>1. Environment Installation & Setup</b></summary>
  <br>

https://github.com/user-attachments/assets/a96c7ddd-ded0-47f9-a522-12888990cd76

  <p><i>A walk-through on installing dependencies via Requirements.txt and building the Cython `Pricing_Branching` extensions.</i></p>
</details>

<details>
  <summary><b>2. Reproduction of Table 7 (Ordered vs Order Independent)</b></summary>
  <br>
  
https://github.com/user-attachments/assets/607db555-d833-4cee-a9b0-f0fcb28c5da4

  <p><i>Executing the compact notebook runs demonstrating the bounds and performance side-by-side.</i></p>
</details>

<details>
  <summary><b>3. Reproduction of Table 8 (BPOCT vs EnumOCT)</b></summary>
  <br>

https://github.com/user-attachments/assets/85af4098-236d-482e-bfe6-ac7084e4af13


  <p><i>Using the CLI to reproduce the BPOCT execution metrics dynamically.</i></p>
</details>

<details>
  <summary><b>4. Reproduction of Tables 9 & 10 (Method Comparisons)</b></summary>
  <br>
 

https://github.com/user-attachments/assets/cdfb51a4-5aa9-41da-a4fa-ef980316a76a


  <p><i>Executing the wrapper script to loop through OrderIndOCT, CompactOCT, EnumOCT, and BPOCT.</i></p>
</details>

<details>
  <summary><b>5. Reproduction of Table 15 (Effects of Cuts)</b></summary>
  <br>
  <video src="https://github.com/eranayoner/OrderIndependentOCT/raw/main/assets/table_15_video.mp4" controls="controls" style="max-width: 100%;"></video>
  <p><i>Running the `OCT_IP_Cuts.ipynb` notebook to generate the branch-and-bound cut metrics.</i></p>
</details>

<details>
  <summary><b>6. Reproduction of Table 16 (Fairness $\epsilon$-constraints)</b></summary>
  <br>
  <video src="https://github.com/eranayoner/OrderIndependentOCT/raw/main/assets/table_16_video.mp4" controls="controls" style="max-width: 100%;"></video>
  <p><i>Running the `OCT_FairnessEpsilonCons.ipynb` parameters cell mapping to the fairness metrics.</i></p>
</details>

<details>
  <summary><b>7. Higher depths trial with D=5 </b></summary>
  <br>
  <video src="https://github.com/eranayoner/OrderIndependentOCT/raw/main/assets/table_16_video.mp4" controls="controls" style="max-width: 100%;"></video>
</details>

## Files

- **OrderIndependentOCT.py**: Main script (CLI) that contains the function which selects and runs OCT methods via Papermill.
- **setup.py**: Builds the Cython extension (`Pricing_Branching`).
- **Requirements.txt**: Lists all dependencies.
- **Datasets/**: CSVs organized as `<dataset>/fold=<n>_train.csv` and `fold=<n>_test.csv`.
- **Notebooks/**:
  - **OrderIndependentOCT\_ExampleNotebook.ipynb**: Runnable end‑to‑end example (method can be chosen using the selection parameter, amongst CompactOCT -1- , POCT-2-, BPOCT-3-, or left as defualt -0- which uses the hybrid method described in the paper, OrderIndependentOCT) .
  - **OCT\_FairnessEpsilonCons.ipynb**: Runnable fairness‑constrained experiments.
  - **OCT\_IP\_Cuts.ipynb**: Runnable notebook for the experiements with cuts, parameters adjusted at the top.
  - **Supporting** (invoked by the main notebooks):\
    `OCT_OrderIndependentModel_Compact.ipynb`,\
    `OCT_OrderedModel_Compact.ipynb`,\
    `OCT_IP.ipynb`,\
    `OCT_BnP.ipynb`,\
    `OCT_Fairness_IP.ipynb`.

## Requirements

Install dependencies via `Requirements.txt`, or directly:

```bash
pip install Cython>=0.29 setuptools numpy pandas scipy scikit-learn gurobipy>=10.0.3 papermill ipykernel more-itertools networkx matplotlib
```

## Installation

```bash
pip install -r Requirements.txt
python -m ipykernel install --user --name=python3
python setup.py build_ext --inplace
```

## Example Notebook Execution

### OrderIndependentOCT\_ExampleNotebook.ipynb

In the example notebook, loop over datasets or add new ones (ensure they follow the `Datasets/<name>/fold=<n>_train.csv` structure):

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
data_list.sort()
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
    'dataset': 'NHPA',
    'd': 2,
    'epsilon': 1,            # initial fairness bound
    'sensitive_features': [5] # e.g. gender column index
}

another example:
data_list = ['nursery']      # select datasets
data_list
sensitive_features = [38,39,40,41,42]     # e.g. parent's occupation
```

### Table 7: Ordered vs Order Independent Formulation

To reproduce a fold comparison for Table 7, directly execute the standard parameters block built into the two formulation notebooks:

```python
# In OCT_OrderedModel_Compact.ipynb AND OCT_OrderIndependentModel_Compact.ipynb
# Parameters
fold = 1
dataset = 'monks-1'
D = 2 # depth
d = 2 
trace = 0
```
Run all cells in both notebooks to compare their execution metrics and objective bounds side-by-side.




## Reproducing Experiments

To reproduce experimental results from the manuscript (e.g., Table 8), use the `OrderIndependentOCT.py` command-line interface or the provided example notebook.

### CLI Usage
```bash
python OrderIndependentOCT.py <dataset> <fold> <depth> <selection>
```
- **dataset**: Name of the dataset folder in `Datasets/`.
- **fold**: Fold index (1-10).
- **depth**: Depth of the tree (e.g., 2, 3, 4).
- **selection**:
    - `0`: Hybrid method (**OrderIndependentOCT**) - selects the best method automatically.
    - `1`: Compact Formulation (**CompactOCT**).
    - `2`: Partial OCT (**POCT**).
    - `3`: Branch-and-Price OCT (**BPOCT**).

**Example: Reproducing a row for 'monks-1' at depth 2 using BPOCT:**
```bash
python OrderIndependentOCT.py monks-1 1 2 3
```

### Mapping Code to Methods

| Feature | Code / Notebook | Description |
| :--- | :--- | :--- |
| **Order Independent Hybrid** | `OrderIndependentOCT.py` | Orchestrates the hybrid method, choosing between Compact, POCT, and BPOCT based on problem size. |
| **Compact Formulation** | `OCT_OrderIndependentModel_Compact.ipynb` | Implements the flow-based compact formulation.|
| **Pattern OCT (POCT)** | `OCT_IP.ipynb` | Implements the formulation for Pattern-based Optimal Classification Trees. |
| **Branch-and-Price (BPOCT)** | `OCT_BnP.ipynb` | Implements the Branch-and-Price algorithm with Beam Search for solving pricing problems. |
| **Fairness Constraints** | `OCT_FairnessEpsilonCons.ipynb` | Implements fairness-constrained OCT experiments using $\epsilon$-constraint method. |

### Reproduction Mapping for Manuscript Tables

| Table | Content | Method / selection | Script / Notebook |
| :--- | :--- | :--- | :--- |
| **Table 7** | Ordered vs Order Independent Formulation | N/A | `OCT_OrderedModel_Compact.ipynb` <br> `OCT_OrderIndependentModel_Compact.ipynb` <br> Results can be recovered from tex files under the dataset ran. The res_ files will have Training Accuracy/ Test Accuracy/ Time/ Optimality gap written at each line|
| **Table 8** | BPOCT vs EnumOCT | `selection=3` for BPOCT <br> `selection=2` for EnumOCT | `OrderIndependentOCT.py` |
| **Table 9 & 10** | Method Comparisons | `selection=0` for Hybrid / OrderIndOCT <br> `selection=1` for CompactOCT <br> `selection=2` for EnumOCT <br> `selection=3` for BPOCT <br> *(Note: The "Root" baseline metric can be extracted from the `selection=3` console logs or the executed papermill run-logs permanently saved in `RunNotebooks/`. All parsed raw results are saved side-by-side as `.txt` files in their respective `Datasets/<dataset>` subfolders.)*| `OrderIndependentOCT.py` |
| **Table 15** | Effects of Cuts | N/A | `OCT_IP_Cuts.ipynb` |
| **Table 16** | Fairness results | N/A | `OCT_FairnessEpsilonCons.ipynb` |

To reproduce a specific row from Table 8, 9, or 10 you may use the CLI as described above with the corresponding `selection` code. For Table 7, 15, and 16, execute the designated notebooks directly after configuring the parameter block at the top of the file.
