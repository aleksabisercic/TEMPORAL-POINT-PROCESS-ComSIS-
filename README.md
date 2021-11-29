# MACHINE LEARNING APPROACH FOR LEARNING TEMPORAL POINT PROCESS

### **Setup**

```terminal
  pip install -r requirements.txt
```

### **To get dataset:**

- Please contact the authors for more information

## **Funcional documentation:**

```python
class Simulation
    def step_simulation(self, dataset, atributes=None, no_steps_max=500):
        ...
        return tn

    def simulate(self, no_simulation=1, dataset: str = 'autoput'):
        ...
        return simulation 
```

Purpose ```Simulation``` class and it's methods is to generate point processes (events) based on learned CIF that was learned durning training.


```python
def evaluate(train_df: pd.DataFrame, path, bin_size, model_name, dataset_type):
    ...
    return 
```

Purpose ```evaluate()```  is to take the simulated data in point processes and first put it into bins (windows) of varible lenght ( `bin_size` ) than to evaluate MAE with ground truth happenings for each bin and do that for each simulation.


### **To run simulation based on learned CIF:**

```terminal
python3 main.py --time-upper 400 --no-sim 10000 --dataset-type 'autoput'
```

### **Arguments**

  | Argument  | Description | Defult | 
  | ------------- | ------------- | ------------- |
  | --time-upper  | How many future events do we want to generate | 400
  | --no-sim  | How many simulations do we want to run | 10000 |
  | --dataset-type  | On what data do we run our simulation | 'autoput' |

## __Purposed Methodology outline:__

![image](https://user-images.githubusercontent.com/64646644/142767166-df20ec70-ad64-48f6-8928-9180a05a712a.png)