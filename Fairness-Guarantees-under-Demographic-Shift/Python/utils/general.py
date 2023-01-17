import sys
import code
import time
import joblib
import itertools
import os
import threading
import copy
import numpy as np
import datetime
from scipy.spatial          import ConvexHull
from matplotlib.collections import LineCollection
from matplotlib.colors      import to_rgba_array


def is_iterable(x):
	try:
		iter(x)
	except Exception:
		return False
	else:
		return True	

def make_seed(digits=8, random_state=np.random):
    return np.floor(random_state.rand()*10**digits).astype(int)

def subdir_incrementer(sd):
	for i in itertools.count():
		yield (sd+'_%d') % i


#####################################################
#   Helpers for dividing parameters among workers   #
#####################################################


def _stack_dicts(base, next, n, replace=False, max_depth=np.inf, na_val=None):
	if isinstance(next,dict) and isinstance(base,dict):
		if max_depth <= 0:
			return np.array([base,next]) if (n>0) else np.array([next])
		out  = {}
		keys = set(base.keys()).union(next.keys())
		for k in keys:
			_base = base[k] if (k in base.keys()) else None
			_next = next[k] if (k in next.keys()) else None
			out[k] = _stack_dicts(_base, _next, n, replace, max_depth-1, na_val=na_val)
	elif isinstance(next,dict) and (base is None):
		out = _stack_dicts({}, next, n, replace, max_depth, na_val=na_val)
	elif isinstance(base,dict) and (next is None):
		out = _stack_dicts(base, {}, n, replace, max_depth, na_val=na_val)
	else:
		if replace:
			out = next if (base is None) else base
		else:
			base_val = np.repeat(na_val,n) if (base is None) else base
			next_val = na_val              if (next is None) else next
			out = np.array(base_val.tolist() + [next_val])
	return out

def stack_all_dicts(*dicts, na_val=None):
	out = {}
	for i,d in enumerate(dicts):
		out = _stack_dicts(out, d, i, max_depth=np.inf, na_val=na_val)
	return out

def stack_all_dicts_shallow(*dicts, na_val=None):
	out = {}
	for i,d in enumerate(dicts):
		out = _stack_dicts(out, d, i, max_depth=1, na_val=na_val)
	return out


#################
#   Profiling   #
#################

class TimerCollection:
	def __init__(self, name=None):
		self.name = name
		self.reset()
	def tic(self, name):
		if not(name in self._times.keys()):
			self._times[name] = []
		self._tics[name] = time.time()
	def toc(self, name):
		t = time.time() - self._tics[name]
		self._times[name].append(t)
		del self._tics[name]
		return t
	def toctic(self, name):
		t1 = time.time()
		t0 = self._tics[name]
		self._times[name].append(t1-t0)
		self._tics[name] = t1
		return t1-t0
	def reset(self, name=None):
		if name is None:
			self._times = {}
			self._tics  = {}
		else:
			self._times[name] = []
			del self._tics[name]
	def get_avg_time(self, name=None):
		if name is None:
			return { name:np.mean(times) for name, times in self._times.items() if len(times) > 0 }
		return np.mean(self._times[name])
	def print_avg_times(self, pad=''):
		key_length = max([ len(k) for k in self._times.keys() ])
		name_str = '' if self.name is None else f'[{self.name}]'
		print(f'{pad}{name_str} Average Times:')
		for name, t in self.get_avg_time().items():
			print(f'{pad}   {name.rjust(key_length)}: {t}')
	def get_total_time(self, name=None):
		if name is None:
			return { name:np.sum(times) for name, times in self._times.items() if len(times) > 0 }
		return np.mean(self._times[name])
	def print_total_times(self, pad=''):
		key_length = max([ len(k) for k in self._times.keys() ])
		name_str = '' if self.name is None else f'[{self.name}]'
		print(f'{pad}{name_str} Total Times:')
		for name, t in self.get_total_time().items():
			print(f'{pad}   {name.rjust(key_length)}: {t}')
	def get_num_tics(self, name=None):
		if name is None:
			return { name:len(times) for name, times in self._times.items() }
		return len(self._times[name])
	def print_num_tics(self, pad=''):
		key_length = max([ len(k) for k in self._times.keys() ])
		name_str = '' if self.name is None else f'[{self.name}]'
		print(f'{pad}{name_str} Tic Counts:')
		for name, t in self.get_num_tics().items():
			print(f'{pad}   {name.rjust(key_length)}: {t}')
	def get_times(self, name=None):
		if name is None:
			return { name:times for name, times in self._times.items() }
		return self._times[name]


#################
#   Debugging   #
#################

def keyboard(quit=False, banner=''):
	''' Interrupt program flow and start an interactive session in the current frame.
		 * quit   : If True, exit the program upon terminating the session. '''
	try:
		raise None
	except:
		frame = sys.exc_info()[2].tb_frame.f_back
	namespace = frame.f_globals.copy()
	namespace.update(frame.f_locals)
	from sys import exit as quit
	namespace.update({'quit':quit})
	code.interact(banner=banner, local=namespace)
	if quit:
		sys.exit()