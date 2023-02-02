# Import Modules
import numpy as np
from time import time
from copy import deepcopy

#####################
# Import from files #
#####################

# Fairness Constraints
from baselines import fair_constraints as f_c

# Seldonian
from baselines.seldonian import SeldonianClassifier

# Diabetes dataset
import datasets.diabetes.diabetes as diabetes

# Helper functions
from helpers.argsweep import ArgumentSweeper
from helpers.constraint_manager import get_parser, get_classification_cm
from helpers.constraints import make_constraints
import helpers.launcher as launcher
import helpers.demographic_shift as ds


########################
#   Model Evaluators   #
########################

# Wrapper for training each model and evaluating it under antagonistic resampling

def _evaluate_model(dataset, trainf, mp):
    cm = get_classification_cm(mp['constraints'])

    # Resample the base dataset uniformly to obtain a training dataset
    n_train = dataset.resample_n_train
    n_candidate = np.floor(mp['r_cand_v_safe'] * n_train).astype(int)
    n_safety = n_train - n_candidate
    dataset0 = dataset.resample(
        n_candidate=n_candidate, n_safety=n_safety, n_test=0)
    dataset0._tparams = dataset._tparams
    dataset0._seed = dataset._seed
    t = time()
    predictf, is_nsf = trainf(dataset0, mp)
    t = time()-t
    dshift_opts = {k: mp[k] for k in ['demographic_variable',
                                      'demographic_variable_values', 'demographic_marginals', 'known_demographic_terms']}
    acc_orig, g_orig, acc_ant, g_ant = ds.evaluate_antagonistic_demographic_shift(
        predictf, mp['constraints'], dataset, dshift_opts)

    return {
        'original_nsf': is_nsf,
        'original_acc': acc_orig,
        'original_g': g_orig,
        'antagonist_acc': acc_ant,
        'antagonist_g': g_ant,
        'runtime': t
    }

# Methods to train models
def _get_hoeff_sc(dataset, mp, enforce_robustness=False):
    model_params = {
        'verbose': False,
        'shape_error': True,
        'model_type': mp['model_type'],
        'ci_mode': 'hoeffding',
        'cs_scale': mp['cs_scale'],
        'robust_loss': False,
        'hidden_layers': mp['hidden_layers']}
    if enforce_robustness:
        for k in ['demographic_variable', 'demographic_variable_values', 'demographic_marginals', 'known_demographic_terms', 'robust_loss']:
            model_params[k] = mp[k]

    # Train SC using hoeffding's inequality
    apply_fairness_constraints = 1
    apply_accuracy_constraint = 0
    sep_constraint = 0
    gamma = None
    e = -mp['e']*100

    if mp["model_type"] == "linear":
        split = dataset.training_splits()
        X, Y, S, R = split['X'], split['Y'], split['S'], split['R']
        x_control = {'S': S.astype(np.int64), 'R': R.astype(np.int64)}
        sensitive_attrs = ['S', 'R']
        sensitive_attrs_to_cov_thresh = {'S': 0.1,
                                        'R': 0.01}
        w = f_c.train_model(X, Y, x_control, f_c._logistic_loss, apply_fairness_constraints,
                            apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)

    if mp["model_type"] == "mlp":
        layer_sizes = [dataset.n_features] + mp["hidden_layers"] + [1] # one, because binary classification
        n_params = 0
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            n_params += (n_in * n_out)
        w = np.random.randn(n_params)

    model = SeldonianClassifier(
        mp['constraints'], mp['deltas'], **model_params)
    accept = model.fit(
        dataset, n_iters=mp['n_iters'], optimizer_name=mp['optimizer'], theta0=w)
    return model.predict, ~accept


def _get_ttest_sc(dataset, mp, enforce_robustness=False):
    model_params = {
        'verbose': False,
        'shape_error': True,
        'model_type': mp['model_type'],
        'ci_mode': 'ttest',
        'cs_scale': mp['cs_scale'],
        'robust_loss': False,
        'hidden_layers': mp['hidden_layers']}
    if enforce_robustness:
        for k in ['demographic_variable', 'demographic_variable_values', 'demographic_marginals', 'known_demographic_terms', 'robust_loss']:
            model_params[k] = mp[k]

    if mp["model_type"] == "linear":
        # Train SC using hoeffding's inequality
        apply_fairness_constraints = 1
        apply_accuracy_constraint = 0
        sep_constraint = 0
        gamma = None
        e = -mp['e']*100

        split = dataset.training_splits()
        X, Y, S, R = split['X'], split['Y'], split['S'], split['R']
        x_control = {'S': S.astype(np.int64), 'R': R.astype(np.int64)}
        sensitive_attrs = ['S', 'R']
        sensitive_attrs_to_cov_thresh = {'S': 0.1,
                                        'R': 0.01}
        w = f_c.train_model(X, Y, x_control, f_c._logistic_loss, apply_fairness_constraints,
                            apply_accuracy_constraint, sep_constraint, sensitive_attrs, sensitive_attrs_to_cov_thresh, gamma)
    
    elif mp['model_type'] == "mlp":
        layer_sizes = [dataset.n_features] + mp["hidden_layers"] + [1] # one, because binary classification
        n_params = 0
        for n_in, n_out in zip(layer_sizes[:-1], layer_sizes[1:]):
            n_params += (n_in * n_out)            
        w = np.random.randn(n_params)

    model = SeldonianClassifier(
        mp['constraints'], mp['deltas'], **model_params)
    accept = model.fit(
        dataset, n_iters=mp['n_iters'], optimizer_name=mp['optimizer'], theta0=w)
    return model.predict, ~accept



# Actual evaluation functions
def eval_hoeff_sc(dataset, mp):
    def trainf(dataset, mp): return _get_hoeff_sc(
        dataset, mp, enforce_robustness=False)
    return _evaluate_model(dataset, trainf, mp)

def eval_ttest_sc(dataset, mp):
    def trainf(dataset, mp): return _get_ttest_sc(
        dataset, mp, enforce_robustness=False)
    return _evaluate_model(dataset, trainf, mp)


def eval_ttest_sc_robust(dataset, mp):
    def trainf(dataset, mp): return _get_ttest_sc(
        dataset, mp, enforce_robustness=True)
    return _evaluate_model(dataset, trainf, mp)



######################
#   Dataset Loader   #
######################

def load_dataset_s(tparams, seed):
    dset_args = {
        'r_train': 1.0,
        'include_intercept': True,
        'include_R': tparams['include_R'],
        'include_S': tparams['include_S'],
        'use_pct': 1.0,
        'seed': seed,
        'standardize': tparams['standardize'],
        'R0': 'Black',
        'R1': 'White'
    }
    dataset = diabetes.load_s(**dset_args)
    dataset.resample_n_train = tparams['n_train']
    dataset._tparams = deepcopy(tparams)
    dataset._seed = seed
    return dataset

def load_dataset_r(tparams, seed):
    dset_args = {
        'r_train': 1.0,
        'include_intercept': True,
        'include_R': tparams['include_R'],
        'include_S': tparams['include_S'],
        'use_pct': 1.0,
        'seed': seed,
        'standardize': tparams['standardize'],
    }
    dataset = diabetes.load_r(**dset_args)
    dataset.resample_n_train = tparams['n_train']
    dataset._tparams = deepcopy(tparams)
    dataset._seed = seed
    return dataset


############
#   Main   #
############

if __name__ == '__main__':

    # Note: This script computes experiments for the cross product of all values given for the
    #       sweepable arguments.
    # Note: Sweepable arguments allow inputs of the form, <start>:<end>:<increment>, which are then
    #       expanded into ranges via np.arange(<start>, <end>, <increment>).
    with ArgumentSweeper() as parser:
        parser.add_argument('base_path', type=str)
        parser.add_argument('--include_R',  action='store_true',
                            help='Whether or not to include race as a predictive feature.')
        parser.add_argument('--include_S',  action='store_true',
                            help='Whether or not to include sex as a predictive feature.')
        parser.add_argument('--standardize',  action='store_true',
                            help='Whether or not to standardize input features.')
        parser.add_argument('--n_jobs',     type=int,
                            default=4,    help='Number of processes to use.')
        parser.add_argument('--n_trials',   type=int,
                            default=10,   help='Number of trials to run.')
        parser.add_argument('--n_iters',    type=int,   default=10,
                            help='Number of SMLA training iterations.')
        parser.add_argument('--optimizer',  type=str,
                            default='cmaes', help='Choice of optimizer to use.')
        parser.add_argument('--definition', type=str,   default='DisparateImpact',
                            help='Choice of safety definition to enforce.')
        parser.add_argument('--e',          type=float,
                            default=0.05, help='Value for epsilon.')
        parser.add_argument('--d',          type=float,
                            default=0.05, help='Value for delta.')
        parser.add_argument('--robust_loss',  action='store_true',
                            help='Causes the loss function to estimate post-demographic shift loss.')
        parser.add_sweepable_argument('--n_train',   type=int,  default=10000,   nargs='*',
                                      help='Number of samples to draw from the population for training.')
        parser.add_sweepable_argument('--r_train_v_test', type=float, default=0.4,
                                      nargs='*', help='Ratio of data used for training vs testing.')
        parser.add_argument('--r_cand_v_safe',  type=float, default=0.4,
                            help='Ratio of training data used for candidate selection vs safety checking. (SMLA only)')
        parser.add_sweepable_argument(
            '--model_type',     type=str, default='linear', nargs='*', help='Base model type to use for SMLAs.')
        parser.add_argument('--fixed_dist',  action='store_true',
                            help='Fixed the distribution post-deployment (only works when dshift_var=race.')
        parser.add_argument('--dshift_var', type=str,       default='race',
                            help='Choice of variable to evaluate demographic shift for.')
        parser.add_argument('--dshift_alpha', type=float,   default=0.1,
                            help='Width of intervals around true marginals representing valid demographic shifts.')
        parser.add_argument('--cs_scale', type=float, default=1.0,
                            help='Scaling factor for predicted confidence intervals during candidate selection.')
        parser.add_argument('--hidden_layers', type=int, nargs='*',help='Number of nodes for each hidden layer')
        args = parser.parse_args()
        args_dict = dict(args.__dict__)

        
        # Generate the constraints and deltas
        if args.dshift_var.lower()[0] == 's':
            population = diabetes.load_s(R0='Black', R1='White')
            constraints = make_constraints(
                args.definition, 'R', np.unique(population._R), args.e)
        if args.dshift_var.lower()[0] == 'r':
            population = diabetes.load_r()
            constraints = make_constraints(
                args.definition, 'S', np.unique(population._S), args.e)
        deltas = [args.d for _ in constraints]

        print()
        print(args.definition, ':')
        print('   Interpreting constraint string \'%s\'' % constraints[0])
        print('                               as \'%r\'.' %
              get_parser().parse(constraints[0]))

        smla_names = ['SC', 'QSC', 'SRC', 'QSRC']

        model_evaluators = {
            'SC': eval_hoeff_sc,
            'QSC': eval_ttest_sc,
            'QSRC': eval_ttest_sc_robust,
        }

        # Store task parameters:
        tparams = {k: args_dict[k] for k in [
            'n_jobs', 'base_path', 'r_train_v_test', 'include_R', 'include_S', 'standardize', 'n_train']}

        # Generate options for enforcing robustness constraints
        if args.dshift_var.lower() == 'sex':
            D = get_parser(mode='inner').parse('S')
            D_values = population._S
        elif args.dshift_var.lower() == 'race':
            D = get_parser(mode='inner').parse('R')
            D_values = population._R
        else:
            raise RuntimeError(
                'This experiment does not support demographic shift for the variable \'%s\'' % args.dshift_var)
        unique_D_values = np.unique(D_values)
        Pr_D = np.array([(D_values == d).mean() for d in unique_D_values])
        assert (args.dshift_alpha >= 0) and (args.dshift_alpha <=
                                             1.0), 'Demographic shift alpha value must be between 0 and 1.'
        smla_dshift_opts = {
            'demographic_variable': D,
            'demographic_variable_values': unique_D_values,
            'demographic_marginals': ds.make_intervals(Pr_D, args.dshift_alpha, epsilon=1e-3),
            'known_demographic_terms': ds.get_population_conditionals(population.all_sets(), constraints, D)
        }

        # Fill in parameter dictionaries for each model
        srl_mparam_names = ['n_iters', 'optimizer', 'model_type',
                            'definition', 'e', 'cs_scale', 'robust_loss', 'hidden_layers']
        bsln_mparam_names = ['definition', 'e']
        mparams = {}
        for name in model_evaluators.keys():
            if name in smla_names:
                mparams[name] = {k: args_dict[k] for k in srl_mparam_names}
            else:
                mparams[name] = {k: args_dict[k] for k in bsln_mparam_names}
            mparams[name]['constraints'] = constraints
            mparams[name]['deltas'] = deltas
            mparams[name]['dshift_alpha'] = args.dshift_alpha
            mparams[name]['dshift_var'] = args.dshift_var
            mparams[name]['r_cand_v_safe'] = args.r_cand_v_safe
            mparams[name].update(smla_dshift_opts)


        #  Expand the parameter sets into a set of configurations
        args_to_expand = parser._sweep_argnames + \
            ['loss', 'kernel', 'cov', 'fl_e', 'n_train']
        tparams, mparams = launcher.make_parameters(
            tparams, mparams, expand=args_to_expand)


        print(ds.make_intervals(Pr_D, args.dshift_alpha, epsilon=1e-3))
        print(ds.make_intervals(Pr_D, args.dshift_alpha, epsilon=1e-3))

        print()
        # Create a results file and directory
        save_path = launcher.prepare_paths(
            args.base_path, tparams, mparams, smla_names, root='results', filename=None)
        print()
        # Run the experiment
        if args.dshift_var.lower() == 'sex':
            launcher.run(args.n_trials, save_path, model_evaluators,
                        load_dataset_s, tparams, mparams, n_workers=args.n_jobs, seed=None)
        elif args.dshift_var.lower() == 'race':
            launcher.run(args.n_trials, save_path, model_evaluators,
                        load_dataset_r, tparams, mparams, n_workers=args.n_jobs, seed=None)
