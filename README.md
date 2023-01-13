
# Fairness Guarantees under Demographic Shift

This repository is the official implementation of [Fairness Guarantees under Demographic Shift](https://openreview.net/pdf?id=wbPObLm6ueA). 

>📋  Optional: include a graphic explaining your approach/main result, bibtex entry, link to demos, blog posts and tutorials

## Requirements

Requires Python 3.x, Numpy 1.16+, and Cython 0.29+

To install further required packages and modules:

```setup
pip install -r requirements.txt
```

Datasets are provided in the repository

## Training and Evaluation

The experiments from the paper can be executed by running the provided batch file from the Python directory, as follows:

```setup
./experiments/scripts/iclr_ds_experiments.bat
```
     
Once the experiments complete, the figures found in the paper can be generated using the following two commands, 

```setup
    python -m experiments.scripts.iclr_figures_adult
    python -m experiments.scripts.iclr_figures_adult --unknown_ds
    python -m experiments.scripts.iclr_figures_brazil
    python -m experiments.scripts.iclr_figures_brazil --unknown_ds
```
    
Once completed, the new figures will be saved to `Python/figures/*` by default.


## Pre-trained Models (Maybe change to Results?)

You can download pretrained models here:

- [Adult Census dataset - Known shift](https://drive.google.com/mymodel.pth) trained on ... using parameters ... 
- [Adult Census dataset - Unknown shift](https://drive.google.com/mymodel.pth) trained on ...
- [Brazilian Student Grades dataset - Known shift](https://drive.google.com/mymodel.pth) trained ...
- [Brazilian Student Grades dataset - Unknown shift](https://drive.google.com/mymodel.pth) trained ...

>📋  Give a link to where/how the pretrained models can be downloaded and how they were trained (if applicable).  Alternatively you can have an additional column in your results table with a link to the models.

## Results

Our model achieves the following performance on :

### [Image Classification on ImageNet](https://paperswithcode.com/sota/image-classification-on-imagenet)

| Model name         | Top 1 Accuracy  | Top 5 Accuracy |
| ------------------ |---------------- | -------------- |
| My awesome model   |     85%         |      95%       |

>📋  Include a table of results from your paper, and link back to the leaderboard for clarity and context. If your main result is a figure, include that figure and link to the command or notebook to reproduce it. 


## Contributing

> SeldonianML is released under the MIT license.
