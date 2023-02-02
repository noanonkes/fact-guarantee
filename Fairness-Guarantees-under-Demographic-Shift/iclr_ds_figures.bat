:: Generation of figures 
python -m scripts.iclr_figures_brazil
python -m scripts.iclr_figures_brazil   --unknown_ds
python -m scripts.iclr_figures_adult
python -m scripts.iclr_figures_adult    --unknown_ds

python -m scripts.iclr_figures_adult    --mlp
python -m scripts.iclr_figures_adult    --unknown_ds --mlp

python -m scripts.iclr_figures_diabetes --unknown_ds --dshift_var sex
python -m scripts.iclr_figures_diabetes --unknown_ds --dshift_var race