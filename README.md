
# Fairness Guarantees under Demographic Shift

This repository is the official reproduction of the implementation of [Fairness Guarantees under Demographic Shift](https://openreview.net/pdf?id=wbPObLm6ueA). 

## Requirements

Requires Python 3.x, Numpy 1.16+, and Cython 0.29+

To install further required packages and modules:

```setup
pip install -r requirements.txt
```

The pre-processed datasets, as well as the original datasets, are provided in the repository.

## Training and Evaluation

The experiments from the paper can be executed by running the provided batch file from the Python directory, as follows:

```setup
./iclr_ds_experiments.bat
```
     
Once the experiments are completed, the figures can be generated by running the following batch file.

```setup
./iclr_ds_figures.bat
```
Given that the files are structured as follows:

```setup
- results
    - results_original_experiments
        - iclr_adult_{mode}_ds_rl_{constraint}
            - iclr_adult_{mode}_ds_rl_{constraint}.h5
        - iclr_brazil_{mode}_ds_rl_{constraint}
            - iclr_brazil_{mode}_ds_rl_{constraint}.h5
        - etc.
            - etc.
    - results_mlp_experiments
        - etc.
            - etc.
    - results_diabetes_experiments   
        - etc.
```
Where `{mode}` can consist of either 'fixed' or 'antag', corresponding to a known and unknown distributional shift respectively, and `{constraint}` can correspond to 'di' and 'dp', meaning the fairness constraints Disparate Impact and Demographic Parity.

Once completed, the figures will be saved to `Fairness-Guarantees-under-Demographic-Shift/figures/*` by default.


## Experiment results

You can download results of our experiments [here](https://drive.google.com/drive/folders/1u41wPeqjdMjkaXf5T0nJtW0i446fycLV?usp=sharing) 


## Overall results

Our experiments support the following claims made in the original paper:

- [X] Claim 1: *High Confidence Fairness Guarantee*

    Reproduction of the original experiments as well as the conducted additional experiments, show that this claims holds. Namely, `Shifty` **never** returns and unfair model. This is shown by utilizing an unseen dataset and a different classifier. 
    
- [X] Claim 2: *Minor Loss of Accuracy*

    The original and additional experiments show strong support for this claim, as results indeed show only a **3%** loss in accuracy when comparing `Shifty` to the other baseline fairness algorithms. 
    
- [ ] Claim 3: *Finding a Solution*

    In this study not enough evidence was found to support this claim, namely that `Shifty` avoids returning *NO_SOLUTION_FOUND* when increasing the number of samples in the training data. 

...

## Contributing

> SeldonianML is released under the MIT license.
