from helpers.utils import TimerCollection
import numpy as np
from functools import partial

from helpers.cmaes_optimizer import OPTIMIZERS
from helpers.constraint_manager import ConstraintManager

from sys import float_info

MAX_VALUE = float_info.max

# Set-up for the classifier
class SeldonianClassifierBase:
    def __init__(
        self,
        constraint_strs,
        shape_error,
        model_type,
        model_params,
        verbose=False,
        ci_mode="hoeffding",
        robustness_bounds={},
        term_values={},
        cs_scale=2.0,
        importance_samplers={},
        demographic_variable=None,
        demographic_variable_values=[],
        demographic_marginals=[],
        known_demographic_terms=None,
        seed=None,
        robust_loss=False,
        hidden_layers=[],
    ):
        self.shape_error = shape_error
        self.verbose = verbose
        self.model_type = model_type
        self.model_variables = {}
        self.model_params = model_params
        self.ci_mode = ci_mode
        self._robustness_bounds = robustness_bounds
        self._term_values = term_values
        self._cs_scale = cs_scale
        self._seed = seed
        self.hidden_layers = hidden_layers

        # Set up the constraint manager to handle the input constraints
        self._cm = ConstraintManager(
            constraint_strs,
            trivial_bounds={},
            keywords=self.keywords,
            importance_samplers=importance_samplers,
            demographic_variable=demographic_variable,
            demographic_variable_values=demographic_variable_values,
            demographic_marginals=demographic_marginals,
            known_demographic_terms=known_demographic_terms,
            debug=False,
            timername="Constraints",
        )

        self._robust_loss = False if demographic_variable is None else robust_loss

        if self._robust_loss:
            self._error_cm = ConstraintManager(
                ["E[Y!=Yp]"],
                trivial_bounds={},
                keywords=self.keywords,
                importance_samplers=importance_samplers,
                demographic_variable=demographic_variable,
                demographic_variable_values=demographic_variable_values,
                demographic_marginals=demographic_marginals,
                known_demographic_terms=known_demographic_terms,
                debug=False,
                timername="Loss",
            )

        # Used for profiling
        self._tc = TimerCollection("SeldonianClassifier")

    @property
    def n_constraints(self):
        return self._cm.n_constraints

    def load_split(self, split, **extra):
        all_data = {**split, **extra}
        return {k: all_data[k] for k in self._cm.referenced_variables}

    def load_error_split(self, split, **extra):
        all_data = {**split, **extra}
        return {k: all_data[k] for k in self._error_cm.referenced_variables}

    def copy_cm(self, constraint_strs):
        return ConstraintManager(
            constraint_strs,
            trivial_bounds=self._cm.trivial_bounds,
            keywords=self.keywords,
            importance_samplers=self._cm._importance_samplers,
            demographic_variable=self._cm._demographic_variable,
            demographic_variable_values=self._cm._demographic_variable_values,
            demographic_marginals=self._cm._demographic_marginals,
            known_demographic_terms=self._cm._known_demographic_terms,
        )

    # Split the data in Data_candidate and Data_fairness (I think?)
    def set_cm_data(self, predictf, split):
        try:
            Yp = predictf(split["X"])
        except TypeError as e:
            return np.array([np.inf])
        self._cm.set_data(self.load_split(split, Yp=Yp))
        if self._robust_loss:
            self._error_cm.set_data(self.load_error_split(split, Yp=Yp))

    # Error and loss function
    def _error(self, predictf, X, Y):
        return np.mean(Y != predictf(X))

    def _loss(self, X, Y, theta=None):
        predictf = self.get_predictf(theta=theta)
        return self._error(predictf, X, Y)

    # Accessor for different optimizers
    def get_optimizer(self, name, dataset, opt_params={}):


        ######
        # This optimizer seems to be the only one that is actually used in the experiments
        # Covariance Matrix Adaptation Evolution Strategy -> stochastic & derivative free (based on evolution concepts ~woa)
        if name == "cmaes":
            return OPTIMIZERS[name](
                self.n_features, sigma0=0.01, n_restarts=50, seed=self._seed
            )
        ######

        raise ValueError(
            "SeldonianClassifierBase.get_optimizer(): Unknown optimizer '%s'." % name
        )

    # Base models - the actual classifiers which input data X and theta

    def get_predictf(self, theta=None):
        return partial(self.predict, theta=theta)

    def predict(self, X, theta=None):

        theta = self.theta if (theta is None) else theta

        ######
        # This model_type seems to be the only one that is actually used in the original experiments
        # Linear -> returns -1 when <0, 0 if ==0, and 1 when >0
        if self.model_type == "linear":
            return np.sign(X.dot(theta))
        ######

        elif self.model_type == "mlp":
            layer_sizes = [self.dataset.n_features] + self.hidden_layers + [1] # one, because binary classification
            for n_in, n_out in zip(layer_sizes[:-2], layer_sizes[1:-1]):
                theta_part = theta[:n_in * n_out].reshape((n_in, n_out))
                X = np.tanh(X @ theta_part)

            return np.sign(X.dot(theta[-layer_sizes[-2]:])) # no tahn over last layer
        
        raise ValueError(
            "SeldonianClassifierBase.predict(): unknown model type: '%s'"
            % self.model_type
        )

    @property
    def n_features(self):
        if self.model_type == "linear":
            return self.dataset.n_features
        if self.model_type == "mlp":
            return self.dataset.n_features * (self.dataset.n_features + 1)

    # Constraint evaluation

    def _loss_using_cm(self):
        # If demographic shift contains intervals, this will return the worst case error rate
        return self._error_cm.upper_bound_constraints(
            None,
            mode="none",
            robustness_bounds=self._robustness_bounds,
            term_values=self._term_values,
        )[0]

    def safety_test(self):
        return self._cm.upper_bound_constraints(
            self.deltas,
            mode=self.ci_mode,
            interval_scaling=1.0,
            robustness_bounds=self._robustness_bounds,
            term_values=self._term_values,
        )

    def predict_safety_test(self, data_ratio):
        return self._cm.upper_bound_constraints(
            self.deltas,
            mode=self.ci_mode,
            interval_scaling=self._cs_scale,
            n_scale=data_ratio,
            robustness_bounds=self._robustness_bounds,
            term_values=self._term_values,
        )


    # Candidate selection

    def candidate_objective(self, theta, split, data_ratio):
        self._tc.tic("eval_c_obj")
        self._tc.tic("set_cm_data")
        self.set_cm_data(self.get_predictf(theta), split)
        self._tc.toc("set_cm_data")
        self._tc.tic("c_obj_predict_safety_test")
        sc_ubs = self.predict_safety_test(data_ratio)
        self._tc.toc("c_obj_predict_safety_test")
        if any(np.isnan(sc_ubs)):
            self._tc.toc("eval_c_obj")
            return MAX_VALUE
        elif (sc_ubs <= 0.0).all():
            self._tc.tic("c_obj_compute_loss")
            val = (
                self._loss_using_cm()
                if self._robust_loss
                else self._loss(split["X"], split["Y"], theta=theta)
            )
            self._tc.toc("c_obj_compute_loss")
            self._tc.toc("eval_c_obj")
            return val
        elif self.shape_error:
            self._tc.tic("c_obj_shape_error")
            sc_ub_max = np.maximum(np.minimum(sc_ubs, MAX_VALUE), 0.0).sum()
            self._tc.toc("c_obj_shape_error")
            self._tc.toc("eval_c_obj")
            return 1.0 + sc_ub_max
        self._tc.toc("eval_c_obj")
        return 1.0

    # Model training
    def fit(
        self,
        dataset,
        n_iters=1000,
        optimizer_name="linear-shatter",
        theta0=None,
        opt_params={},
    ):
        self.dataset = dataset
        opt = self.get_optimizer(
            optimizer_name, dataset, opt_params=opt_params)

        # Store any variables that will be required by the model
        # only in case of rbf so unused...
        # self._store_model_variables(dataset)

        # Compute number of samples of g(theta) for the candidate and safety sets
        # Note: This assumes that the number of samples used to estimate g(theta)
        #       doesn't depend on theta itself.
        data_ratio = dataset.n_safety / dataset.n_optimization

        # Get the optimizer and set the candidate selection function
        split = dataset.optimization_splits()

        c_objective = partial(
            self.candidate_objective, split=split, data_ratio=data_ratio
        )

        # Perform candidate selection using the optimizer
        self._tc.tic("minimize_c_obj")
        self.theta, _ = opt.minimize(c_objective, n_iters, theta0=theta0)
        self._tc.toc("minimize_c_obj")

        # Perform the safety check to determine if theta can be used
        predictf = self.get_predictf()
        self.set_cm_data(predictf, dataset.safety_splits())
        st_thresholds = self.safety_test()
        accept = False if any(np.isnan(st_thresholds)) else (
            st_thresholds <= 0.0).all()
        return accept
        # return True


    # Model evaluation

    def evaluate(self, dataset, predictf=None, override_is_seldonian=False):
        ds_ratio = dataset.n_safety / dataset.n_optimization
        meta = {}
        splits = {
            "candidate": dataset.optimization_splits(),
            "safety": dataset.safety_splits(),
            "train": dataset.training_splits(),
            "test": dataset.testing_splits(),
        }

        # We don't assume to know what model predictf uses, so assume that
        #   that any predictf passed in as an argument is Non-Seldonian
        if predictf is None:
            predictf = self.get_predictf()
            meta["is_seldonian"] = True
        else:
            meta["is_seldonian"] = override_is_seldonian

        # Record statistics for each split of the dataset
        for name, split in splits.items():
            try:
                Yp = predictf(split["X"])
            except TypeError as e:
                meta["loss_%s" % name] = np.nan
                for cnum in range(self.n_constraints):
                    meta["co_%d_mean" % cnum] = np.nan
            else:
                meta["loss_%s" % name] = self._error(
                    predictf, split["X"], split["Y"])
                data = self.load_split(split, Yp=Yp)
                self._cm.set_data(data)
                values = self._cm.evaluate()
                for cnum, value in enumerate(values):
                    meta["co_%d_mean" % cnum] = value

        # Record SMLA-specific values or add baseline defaults
        if meta["is_seldonian"]:
            self.set_cm_data(predictf, splits["safety"])
            stest = self.safety_test()
            self.set_cm_data(predictf, splits["candidate"])
            pred_stest = self.predict_safety_test(ds_ratio)
            meta["accept"] = False if any(
                np.isnan(stest)) else (stest <= 0.0).all()
            meta["predicted_accept"] = (
                False if any(np.isnan(pred_stest)) else (
                    pred_stest <= 0.0).all()
            )
            for i, (st, pst) in enumerate(zip(stest, pred_stest)):
                meta["co_%d_safety_thresh" % i] = st
                meta["co_%d_psafety_thresh" % i] = pst
        else:
            meta["accept"] = True
            meta["predicted_accept"] = True
            for i in range(self.n_constraints):
                meta["co_%d_safety_thresh" % i] = np.nan
                meta["co_%d_psafety_thresh" % i] = np.nan
        return meta


########################
#   Base Classifiers   #
########################
class SeldonianClassifier(SeldonianClassifierBase):
    keywords = {
        "FPR": "E[Yp=1|Y=-1]",
        "FNR": "E[Yp=-1|Y=1]",
        "TPR": "E[Yp=1|Y=1]",
        "TNR": "E[Yp=-1|Y=-1]",
        "PR": "E[Yp=1]",
        "NR": "E[Yp=-1]",
    }

    def __init__(
        self,
        constraint_strs,		# specified in experiment file
        deltas,					# specified in experiment file
        shape_error=False,		# in model_params
        verbose=False,			# in model_params
        model_type="linear",  # in model_params - always linear
        hidden_layers=[], 
        model_params={},		# kwarg? in experiment file - if robustness=enforced extra params are added
        # experiments use both 'ttest' (for Quasi-) and 'Hoeffding' (for Regular)
        ci_mode="hoeffding",
        robustness_bounds=[],
        term_values={},
        cs_scale=2.0,			# in model_params
        importance_samplers={},
        demographic_variable=None,			# for robustness
        demographic_variable_values=[],		# for robustness
        demographic_marginals=[],			# for robustness
        known_demographic_terms=None,		# for robustness
        seed=None,
        robust_loss=False,					# for robustness
    ):
        self.deltas = deltas
        super().__init__(
            constraint_strs,
            shape_error=shape_error,
            verbose=verbose,
            model_type=model_type,
            hidden_layers=hidden_layers,
            model_params=model_params,
            ci_mode=ci_mode,
            robustness_bounds=robustness_bounds,
            term_values=term_values,
            cs_scale=cs_scale,
            importance_samplers=importance_samplers,
            demographic_variable=demographic_variable,
            demographic_variable_values=demographic_variable_values,
            demographic_marginals=demographic_marginals,
            known_demographic_terms=known_demographic_terms,
            seed=seed,
            robust_loss=robust_loss,
        )