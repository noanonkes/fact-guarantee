import numpy as np
from copy import deepcopy
from .constraint_manager import ConstraintManager, get_parser
from .utils import optimize_on_simplex

def evaluate_antagonistic_demographic_shift(predictf, constraints, population, opts):
    assert len(constraints) == 1, ('evaluate_antagonistic_demographic_shift(): This function is only designed to evaluate shift under a single constraint. %d Constraints provided.' % len(constraints))

    data = population.all_sets()
    data['Yp'] = predictf(data['X'])            

    # Compute original accuracy
    acc_orig = np.mean(data['Yp'] == data['Y'])
    
    # Compute original g(theta)
    cm = ConstraintManager(constraints)
    cm.set_data(data)
    g_orig = cm.evaluate()[0]
    
    # Compute accuracy and g(theta) after demographic shift
    g_shifted, acc_shifted = get_antagonistic_results_opt(data, population, constraints, opts)

    return acc_orig, g_orig, acc_shifted, g_shifted


def get_antagonistic_results_opt(data, population, constraints, opts):
    assert len(constraints) == 1, ('get_antagonistic_results_opt(): This function is only designed to evaluate shift under a single constraint. %d Constraints provided.' % len(constraints))
    new_opts = deepcopy(opts)
    if opts['demographic_marginals'].ndim == 2:
        def foo(probs, data=data, constraints=constraints, opts=opts):
            new_opts = deepcopy(opts)
            new_opts['demographic_marginals'] = probs
            cm = ConstraintManager(constraints, **new_opts)
            cm.set_data(data)
            return -cm.evaluate()[0]
        f = lambda _p: foo(_p)
        # result = op0timize_on_simplex(f, deepcopy(opts['demographic_marginals']))
        # eq_cons =   [{'type': 'eq', 'fun' : lambda _q: _q.sum() - 1} ]
        # ant_result = optimize.shgo(f, deepcopy(opts['demographic_marginals']), constraints=eq_cons, minimizer_kwargs={'method':'SLSQP', 'constraints':eq_cons}, options={'disp':False}, sampling_method='sobol')
        new_opts['demographic_marginals'] = optimize_on_simplex(f, deepcopy(opts['demographic_marginals']))
    cm = ConstraintManager(constraints+['E[Y=Yp]'], **new_opts)
    cm.set_data(data)
    return cm.evaluate()


def _compute_population_conditional(C, D_str, D_unique, cm):
    if C is None:
        # If C is None, that means we're conditioning on True
        Pr_D_g_C = []
        for d in D_unique:
            D = get_parser(mode='event').parse('%s=%r' % (D_str,d))
            D_values = cm._evaluate(D)
            Pr_D_g_C.append( D_values.sum() / len(D_values) )
        return {
            'D' : D_unique,
            'P(C|D)' : np.ones(len(D_unique)),
            'P(D|C)' : np.array(Pr_D_g_C)
        }
    else:
        C_values = cm._evaluate(C)
        Pr_C_g_D = []
        Pr_D_g_C = []
        for d in D_unique:
            D = get_parser(mode='event').parse('%s=%r' % (D_str,d))
            D_values = cm._evaluate(D)
            C_and_D_values = np.logical_and(C_values,D_values)
            Pr_C_g_D.append( C_and_D_values.sum() / D_values.sum() )
            Pr_D_g_C.append( C_and_D_values.sum() / C_values.sum() )
        return {
            'D' : D_unique,
            'P(C|D)' : np.array(Pr_C_g_D),
            'P(D|C)' : np.array(Pr_D_g_C)
        }


def get_population_conditionals(population_data, constraints, demographic_variable):
    conditions = [ ev.sample_set.condition.name for ev in ConstraintManager(constraints).expected_values.values() ]
    dummy_constraints = [ ('E[%s]'%s) for s in conditions ]
    dummy_constraints.append('E[%s]'%demographic_variable.name)
    dummy_cm = ConstraintManager(dummy_constraints)
    dummy_cm.set_data(population_data)

    D = get_parser
    D_str = demographic_variable.name
    D_values = np.unique(population_data[D_str])

    conditionals = {None : _compute_population_conditional(None, D_str, D_values, dummy_cm)}
    for C_str in conditions:
        C = get_parser(mode='event').parse(C_str)
        conditionals[C.name] = _compute_population_conditional(C, D_str, D_values, dummy_cm)
    return conditionals


def make_intervals(marginals, alpha, epsilon=1e-3):
    intervals = []
    for p in marginals:
        pmin = (1-alpha)*p + alpha*epsilon
        pmax = (1-alpha)*p + alpha*(1-epsilon)
        intervals.append([pmin,pmax])
    return np.array(intervals)