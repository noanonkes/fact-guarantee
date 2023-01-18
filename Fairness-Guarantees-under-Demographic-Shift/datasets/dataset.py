import numpy as np
from copy import deepcopy

class Dataset(object):
	def __init__(self, n_candidate, n_safety, n_test, seed=None, meta_information={}, **contents):
		# Record dataset split sizes
		self._n_safety    = n_safety
		self._n_candidate = n_candidate
		self._n_test      = n_test
		self._n_train     = n_candidate + n_safety
		self._n_samples   = self._n_train + n_test

		self._seed = seed
		self._meta_information = meta_information

		self._contents = deepcopy(contents)
		self._unique_values = {}
		for k, v in contents.items():
			setattr(self, '_%s' % k, v)
			# Adjusted
			if v.dtype == np.int32 or v.dtype == int:
				self._unique_values[k] = np.unique(v)

		# Compute indices for the splits
		self._inds = {
			'all'   : np.arange(0, self._n_samples),
			'train' : np.arange(0, self._n_train),
			'test'  : np.arange(self._n_train, self._n_samples),
			'opt'   : np.arange(0, self._n_candidate),
			'saf'   : np.arange(self._n_candidate, self._n_train)
		}
	
	@property
	def n_train(self):
		return len(self._inds['train'])
	@property
	def n_test(self):
		return len(self._inds['test'])
	@property
	def n_optimization(self):
		return len(self._inds['opt'])
	@property
	def n_safety(self):
		return len(self._inds['saf'])

	def _get_splits(self, index_key, keys=None):
		keys = self._contents.keys() if (keys is None) else keys
		inds = self._inds[index_key]
		return { k:self._contents[k][inds] for k in keys }
	def all_sets(self, keys=None):
		return self._get_splits('all', keys=keys)
	def training_splits(self, keys=None):
		return self._get_splits('train', keys=keys)
	def testing_splits(self, keys=None):
		return self._get_splits('test', keys=keys)
	def optimization_splits(self, keys=None):
		return self._get_splits('opt', keys=keys)
	def safety_splits(self, keys=None):
		return self._get_splits('saf', keys=keys)

class ClassificationDataset(Dataset):
	def __init__(self, all_labels, n_candidate, n_safety, n_test, seed=None, meta_information={}, **contents):
		assert 'X' in contents.keys(), 'ClassificationDataset.__init__(): Feature matrix \'X\' is not defined.'
		assert 'Y' in contents.keys(), 'ClassificationDataset.__init__(): Label vector \'Y\' is not defined.'
		super().__init__(n_candidate, n_safety, n_test, seed=seed, meta_information=meta_information, **contents)
		self._labels = np.unique(all_labels)
	@property
	def n_features(self):
		return self._X.shape[1]
	@property
	def n_labels(self):
		return len(self._labels)

	def resample(self, n_candidate=None, n_safety=None, n_test=None, probf=None):
		n_candidate = self._n_candidate if n_candidate is None else n_candidate
		n_safety = self._n_safety if n_safety is None else n_safety
		n_test = self._n_test if n_test is None else n_test
		n = len(self._X)
		rand = np.random.RandomState(self._seed)
		if probf is None:
			P = np.ones(n) / n
		else:
			P = np.array([ probf(i,x,y,t) for i,(x,y,t) in enumerate(zip(self._X, self._R, self._T)) ])

		I = rand.choice(n, n_candidate+n_safety+n_test, replace=True, p=P)
		contents = { k:v[I] for k,v in self._contents.items() }
		output = ClassificationDataset(self._labels, n_candidate, n_safety, n_test, seed=self._seed, meta_information=self._meta_information, **contents)
		output._unique_values = deepcopy(self._unique_values)
		return output
	
def standardized(X):
	if X.ndim == 1:
		X = X - X.mean(0)
		X = X / X.std(0,ddof=1)
	else:
		X = X - X.mean(0)[None,:]
		X = X / X.std(0,ddof=1)[None,:]
	return X

def with_intercept(X):
	return with_feature(X, np.ones(len(X)))

def with_feature(X, T):
	if X.ndim == 1:
		X = X[:,None]
	if T.ndim == 1:
		T = T[:,None]
	return np.hstack((X, T))