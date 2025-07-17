# %%
# read dataset info:
# choose appropriate method:
## Apply method:

import ast
from pprint import pformat

# Function to classify based on the decision tree rules
def classify_method(row):
    if row['depth'] <= 2:
        if row['# of Features'] <= 249:
            return 2  # Class 2
        else:
            return 1  # Class 1
    else:
        if row['# of Datapoints'] <= 1499:
            if row['# of Features'] <= 14:
                return 2  # Class 2
            else:
                return 1  # Class 1
        else:
            if row['# of Features'] <= 49:
                return 2  # Class 2
            else:
                return 3  # Class 3

# Function to select time based on predicted class
# BP_OCT	P_OCT	CompactOCT
def choose(res):
    if res == 1:
        return 'CompactOCT','_OrderIndependentModel_Compact','_resCompact_','Compact'
    elif res == 2:
        return 'P_OCT','_IP','_res_IP_','IP'
    elif res== 3:
        return 'BP_OCT','_BnP','_res_BnP_', 'BnP'
    



# %%
import papermill as pm
import pandas as pd
import numpy as np




# Execute notebook with different parameters
def OrderIndependentOCT(fold,dataset, depth, trace=0,selection=0):

        # Define parameters
        parameters = {
        }
        parameters['fold'] = fold  # Update parameter values as specified
        parameters['dataset'] = dataset
        parameters['D'] = depth
        parameters['d'] = depth
        parameters['trace']=trace
        selection=selection# defualt is 0 and lets OrderIndependentOCT choose the method to use


        if selection==0:
            # get dataset info for classification of method to be chosen
            relpath="./Datasets/"+dataset +"/fold="+str(fold)+"_train.csv"
            data = pd.read_csv(relpath)[:]#pd.read_csv("./Datasets/fold=1_train.csv")
            N=len(data) # number of datapoints
            J=len(data.columns)-1 # number of features
            del data
            instance_info={'# of Features':J,'depth':depth, '# of Datapoints':N}
            method,extention,output,decision_rules_output=choose(classify_method(instance_info))
            print(method,'is used based on instance specifications...')
        else:
            method,extention,output, decision_rules_output=choose(selection)
            print(method, 'selected by user...')
        
        output_notebook_path = f'RunNotebooks/OCT'+extention+str(list(parameters.values()))+'.ipynb'  # Specify output path
        try:
            print('OCT'+extention+'.ipynb')
            pm.execute_notebook(
                input_path='OCT'+extention+'.ipynb',
                output_path=output_notebook_path,
                parameters=parameters
            )
            print('Done')
        except:
            print('An exception occured...')
        

        relpath="./Datasets/"+dataset +"/fold="+str(fold)
        res=np.loadtxt(relpath+output+str(depth)+'.txt',dtype=np.float32,delimiter=',')[:,1].tolist()
        train_acc=res[0]
        test_acc=res[1]
        time=res[2]
        opt_gap=res[3]


        print(f"Train Accuracy : {train_acc:.2%}")
        print(f"Test  Accuracy : {test_acc:.2%}")
        print(f"Elapsed Time   : {time:.2f} s")
        print(f"Optimality Gap : {opt_gap:.2%}")
        # read decision rules found
        filename = f"{relpath}_DecisionRules_{decision_rules_output}_{depth}.txt"




        # 1) Read & parse the file into a Python object (list of rules)
        with open(filename, 'r') as f:
            text = f.read()
        decision_rules = ast.literal_eval(text)   # now a list (not a string)

        # 2) Pretty‐print each rule
        print("Decision rules for each leaf are as follows:")
        for idx, rule in enumerate(decision_rules, start=1):
            rule_str = pformat(rule)
            print(f" Leaf {idx}: {rule_str}")


        


