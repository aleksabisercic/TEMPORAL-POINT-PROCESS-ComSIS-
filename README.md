# A machine learning approach for learning temporal point process
> [Computer Science and Information Systems 2022 Volume 19, Issue 2, Pages: 1007-1022](https://doiserbia.nb.rs/Article.aspx?id=1820-02142200016P)

*Despite a vast application of temporal point processes in infectious disease diffusion forecasting, ecommerce, traffic prediction, preventive maintenance, etc, there is no significant development in improving the simulation and prediction of temporal point processes in real-world environments. With this problem at hand, we propose a novel methodology for learning temporal point processes based on one-dimensional numerical integration techniques. These techniques are used for linearising the negative maximum likelihood (neML) function and enabling backpropagation of the neML derivatives. Our approach is tested on two real-life datasets. Firstly, on high frequency point process data, (prediction of highway traffic) and secondly, on a very low frequency point processes dataset, (prediction of ski injuries in ski resorts). Four different point process baseline models were compared: second-order Polynomial inhomogeneous process, Hawkes process with exponential kernel, Gaussian process, and Poisson process. The results show the ability of the proposed methodology to generalize on different datasets and illustrate how different numerical integration techniques and mathematical models influence the quality of the obtained models. The presented methodology is not limited to these datasets and can be further used to optimize and predict other processes that are based on temporal point processes.*

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
