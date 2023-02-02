:: Original Experiments
python -m scripts.brazil_demographic_shift iclr_brazil_fixed_ds_rl_di  --n_jobs 8 --n_trials 10 --n_train 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 --definition DisparateImpact    --e -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --gpa_cutoff 3.0 --dshift_var race --cs_scale 1.5 --fixed_dist --robust_loss
python -m scripts.brazil_demographic_shift iclr_brazil_fixed_ds_rl_dp  --n_jobs 8 --n_trials 10 --n_train 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 --definition DemographicParity  --e  0.1 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --gpa_cutoff 3.0 --dshift_var race --cs_scale 1.5 --fixed_dist --robust_loss

python -m scripts.brazil_demographic_shift iclr_brazil_antag_ds_rl_di  --n_jobs 8 --n_trials 10 --n_train 10000 20000 30000 40000 50000 60000 --definition DisparateImpact    --e -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --gpa_cutoff 3.0 --dshift_var race --dshift_alpha 0.25 --cs_scale 1.5 --robust_loss
python -m scripts.brazil_demographic_shift iclr_brazil_antag_ds_rl_dp  --n_jobs 8 --n_trials 10 --n_train 10000 20000 30000 40000 50000 60000 --definition DemographicParity  --e  0.1 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --gpa_cutoff 3.0 --dshift_var race --dshift_alpha 0.25 --cs_scale 1.5 --robust_loss

python -m scripts.adult_demographic_shift iclr_adult_fixed_ds_rl_di  --n_jobs 8 --n_trials 10 --n_train 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 --definition DisparateImpact    --e  -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --cs_scale 1.5 --fixed_dist --robust_loss
python -m scripts.adult_demographic_shift iclr_adult_fixed_ds_rl_dp  --n_jobs 8 --n_trials 10 --n_train 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 --definition DemographicParity  --e  0.05 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --cs_scale 1.5 --fixed_dist --robust_loss

python -m scripts.adult_demographic_shift iclr_adult_antag_ds_rl_di  --n_jobs 8 --n_trials 10 --n_train 10000 20000 30000 40000 50000 60000 --definition DisparateImpact    --e -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --dshift_alpha 0.5 --cs_scale 1.5 --robust_loss
python -m scripts.adult_demographic_shift iclr_adult_antag_ds_rl_dp  --n_jobs 8 --n_trials 10 --n_train 10000 20000 30000 40000 50000 60000 --definition DemographicParity  --e  0.1 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --dshift_alpha 0.5 --cs_scale 1.5 --robust_loss

:: Additional Experiments - MLP classifier
python -m scripts.adult_demographic_shift iclr_adult_fixed_ds_rl_di  --model_type mlp --hidden_layers 16 8  --n_jobs 8 --n_trials 10 --n_train 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 --definition DisparateImpact    --e  -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --cs_scale 1.5 --fixed_dist --robust_loss
python -m scripts.adult_demographic_shift iclr_adult_fixed_ds_rl_dp  --model_type mlp --hidden_layers 16 8  --n_jobs 8 --n_trials 10 --n_train 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000 --definition DemographicParity  --e  0.05 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --cs_scale 1.5 --fixed_dist --robust_loss

python -m scripts.adult_demographic_shift iclr_adult_antag_ds_rl_di  --model_type mlp --hidden_layers 16 8  --n_jobs 8 --n_trials 10 --n_train 10000 20000 30000 40000 50000 60000 --definition DisparateImpact    --e -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --dshift_alpha 0.5 --cs_scale 1.5 --robust_loss
python -m scripts.adult_demographic_shift iclr_adult_antag_ds_rl_dp  --model_type mlp --hidden_layers 16 8  --n_jobs 8 --n_trials 10 --n_train 10000 20000 30000 40000 50000 60000 --definition DemographicParity  --e  0.1 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --dshift_alpha 0.5 --cs_scale 1.5 --robust_loss

:: Additional Experiments - different dataset
python -m scripts.diabetes_demographic_shift iclr_diabetes_antag_ds_rl_sex_di  --n_jobs 8 --n_trials 10 --n_train 10000 20000 30000 40000 50000 60000 --definition DisparateImpact    --e -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --dshift_alpha 0.5 --cs_scale 1.5 --robust_loss
python -m scripts.diabetes_demographic_shift iclr_diabetes_antag_ds_rl_sex_dp  --n_jobs 8 --n_trials 10 --n_train 10000 20000 30000 40000 50000 60000 --definition DemographicParity  --e  0.1 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var sex --dshift_alpha 0.5 --cs_scale 1.5 --robust_loss

python -m scripts.diabetes_demographic_shift iclr_diabetes_antag_ds_rl_race_di  --n_jobs 8 --n_trials 10 --n_train 10000 20000 30000 40000 50000 60000 --definition DisparateImpact    --e -0.8 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var race --dshift_alpha 0.25 --cs_scale 1.5 --robust_loss
python -m scripts.diabetes_demographic_shift iclr_diabetes_antag_ds_rl_race_dp  --n_jobs 8 --n_trials 10 --n_train 10000 20000 30000 40000 50000 60000 --definition DemographicParity  --e  0.1 --n_iters 20000 --standardize --r_train_v_test 0.4 --r_cand_v_safe 0.4 --d 0.05 --optimizer cmaes --dshift_var race --dshift_alpha 0.25 --cs_scale 1.5 --robust_loss


:: Generation of figures 
python -m scripts.iclr_figures_brazil
python -m scripts.iclr_figures_brazil   --unknown_ds
python -m scripts.iclr_figures_adult
python -m scripts.iclr_figures_adult    --unknown_ds

python -m scripts.iclr_figures_adult    --mlp
python -m scripts.iclr_figures_adult    --unknown_ds --mlp

python -m scripts.iclr_figures_diabetes --unknown_ds --dshift_var sex
python -m scripts.iclr_figures_diabetes --unknown_ds --dshift_var race