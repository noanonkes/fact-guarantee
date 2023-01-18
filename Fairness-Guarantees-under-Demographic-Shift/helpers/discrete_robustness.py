import numpy as np
from scipy.optimize import minimize, LinearConstraint
from time import time
import warnings
import sys, code

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

def cCGF(c, V, p):
	''' Computes the centered cumulant generating function. '''
	EV  = np.sum(p * V)
	# A numerically stable computation
	with warnings.catch_warnings():
		warnings.filterwarnings('error')
		offset = (c*V).max()
		return (offset-c*EV) + np.log(np.sum([ _p * np.exp(c*v-offset) for (_p,v) in zip(p,V) if _p > 0 ])) 
	# MGF = np.sum([ _p * np.exp(c*(v-EV)) for (_p,v) in zip(p,V) if _p > 0 ])
	# return np.log(MGF)


def get_largest_step_ev_bounds(p, G, c, V, evmin, evmax, max_step=np.inf, debug=False):
	''' Calculates the largest step that can be taken before a constraint is violated. '''
	if np.isclose(V.dot(G), 0):
		pass
	elif V.dot(G) > 0:
		max_step = min(max_step, (evmax-sum(p*V))/sum(G*V))
	elif V.dot(G) < 0:
		max_step = min(max_step, (evmin-sum(p*V))/sum(G*V))

	for (g,_p) in zip(G,p):
		if np.isclose(g,0):
			pass
		elif g > 0:
			max_step = min(max_step,  (1-_p)/g)
		else:
			max_step = min(max_step, -(_p-0)/g)
	return max_step

def project_grad_ev_bounds(G, p, V, evmin, evmax):
	''' Removes components of G that point in directions that violate constraints. '''
	

	J_mins = np.isclose(0,p)
	J_maxs = np.isclose(1,p)
	k = len(p)

	def f(g, G=G):
		r = G - g
		return r.dot(r), -2*r

	c_As = []
	c_Bs = []
	c_Cs = []
	E = (np.eye(k) * k - np.ones((k,k))) / np.sqrt(k**2-k)
	if any(J_mins):
		c_As.append(E[J_mins])
		c_Bs.append(np.zeros(sum(J_mins)))
		c_Cs.append(np.inf * np.ones(sum(J_mins)))
	if any(J_maxs):
		c_As.append(E[J_maxs])
		c_Bs.append(-np.inf * np.ones(sum(J_maxs)))
		c_Cs.append(np.zeros(sum(J_maxs)))

	EV = sum(p*V)
	d = V - V.mean()
	d = d / np.linalg.norm(d)
	if np.isclose(EV,evmin) or (EV < evmin):
		c_As.append(d[None,:])
		c_Bs.append(np.array([ 0 ]))
		c_Cs.append(np.array([ np.inf ]))
	elif np.isclose(EV,evmax) or (EV > evmax):
		c_As.append(d[None,:])
		c_Bs.append(np.array([ -np.inf ]))
		c_Cs.append(np.array([ 0 ]))

	if len(c_As) == 0:
		Gp = G
	else:
		c_As = np.concatenate(c_As)
		c_Bs = np.concatenate(c_Bs)
		c_Cs = np.concatenate(c_Cs)
		lc = LinearConstraint( c_As, c_Bs, c_Cs )
		Gp = minimize(f, args=(G,), x0=np.zeros_like(G), jac=True, constraints=[lc]).x

	Gp[J_mins] = np.maximum(Gp[J_mins],0)
	Gp[J_maxs] = np.minimum(Gp[J_maxs],0)
	Gp[~np.logical_or(J_mins,J_maxs)] = Gp[~np.logical_or(J_mins,J_maxs)] - Gp.sum() 
	return Gp

def maximize_over_P(c, e, V, p, projectionf, stepf, step_ratio=1.0, upper=True):	
	''' Takes a step to maxmize the upper bound or minimize the lower bound w.r.t. p.'''
	if np.isclose(e,0) and upper:
		def gradf(p):
			return projectionf(V-V.mean(), p)
	elif np.isclose(e,0):
		def gradf(p):
			return projectionf(-V+V.mean(), p)
	else:
		_c = c if upper else -c
		Z = _c*V
		E = np.maximum(np.exp(Z - Z.max()), 1e-20)
		def gradf(p,c=_c,Z=Z,E=E,V=V):
			G = (1/np.abs(c)) * E / p.dot(E)
			G = G - G.mean()
			return projectionf(G, p)
	ps = [p]
	i = 0
	G = gradf(p)
	while not(all(np.isclose(G,0))):
		step = step_ratio*stepf(p, G, c, V, max_step=np.inf)
		p = p + step*G
		ps.append(p)
		G = gradf(p)
	return p

def minimize_over_c(c, e, V, p, upper=True):
	''' Computes the c that defines the upper/lower bound. '''
	if np.isclose(e,0):
		return 0.0

	def invLF(c, e, V, p, upper=upper):
		if np.isclose(c, 0) and not(np.isclose(e,0)):
			return np.inf
		if upper:
			return (cCGF( c,V,p) + e) / c
		else:
			return (cCGF(-c,V,p) + e) / c


	c_start = c
	v_start = invLF(c_start, e, V, p) 	
	alpha = 1e-12
	while not(np.isclose(c_start,0)) and (invLF(c_start+alpha, e, V, p) >= v_start):
		c_start = c_start / 2
		v_start = invLF(c_start, e, V, p) 	

	
	# Perform a line search to find an interval containing the optimum
	c_0 = c_start
	v_0 = v_start
	c_1 = c_start
	v_1 = v_start
	i = 0
	while True:
		c_2 = c_start + alpha
		v_2 = invLF(c_2, e, V, p)
		if c_2 >= 1e100:
			return c_2
		if np.isnan(v_2) or v_2 > v_1: 
			break
		else:
			if i > 0:
				alpha = alpha*10
			c_0 = c_1
			v_0 = v_1
			c_1 = c_2
			v_1 = v_2
			i += 1

	# We now have v_1 < v_0 and v_1 < v_2: finish the search using bisection
	while True:
		if min(v_2-v_1, v_0-v_1) < 1e-15:
			return c_1
		c_05 = 0.5 * (c_0 + c_1) 
		v_05 = invLF(c_05, e, V, p)
		if v_0 < v_05:
			return c_0
		if v_05 <= v_1:
			v_2 = v_1
			c_2 = c_1
			v_1 = v_05
			c_1 = c_05
		else:
			c_15 = 0.5 * (c_1 + c_2) 
			v_15 = invLF(c_15, e, V, p)
			if v_15 <= v_1:
				v_0 = v_1
				c_0 = c_1
				v_1 = v_15
				c_1 = c_15
			else:
				v_2 = v_15
				c_2 = c_15
				v_0 = v_05
				c_0 = c_05

def optimize_bound(e, c, V, p, projectionf, stepf, upper=True):
	''' Computes the upper or lower bound subject to constraints on the components of p.'''
	v_last = np.inf
	if np.isclose(e,0):
		def func(e,c,V,p):
			return p.dot(V)
	elif upper:
		def func(e,c,V,p):
			return p.dot(V) + (cCGF(c,V,p)+e)/c
	else:
		def func(e,c,V,p):
			return p.dot(V) - (cCGF(-c,V,p)+e)/c
	# print('UPPER' if upper else 'LOWER')
	c_orig = c
	p_orig = p
	while True:
		p_old = p
		t = time()
		p = maximize_over_P(c, e, V, p_orig, projectionf, stepf, upper=upper)
		# print('optimized %r --> %r   (%f)' % (p_old, p, time()-t))
		
		c_old = c
		t = time()
		c = minimize_over_c(c_orig, e, V, p, upper=upper)
		# print('optimized %f --> %f   (%f)' % (c_old, c, time()-t))
		
		v = func(e,c,V,p)
		# print(v, c, c_orig, p, p_orig)
		if np.abs(v_last-v) < 1e-15 or v_last < v:
			# print()
			return v 
		v_last = v

def compute_robustness_bounds_ev(e, V, p, evmin, evmax, c0=1e-12, delta=0.05):
	''' Computes a lower bound and upper bound on E_Q[f(Z)] given samples of f(Z) ~ P.'''
	k = len(V)
	def projectionf(G, p, V=V, evmin=evmin, evmax=evmax):
		return project_grad_ev_bounds(G, p, V, evmin=evmin, evmax=evmax)
	def stepf(p, G, c, V, max_step=np.inf, evmin=evmin, evmax=evmax):
		return get_largest_step_ev_bounds(p, G, c, V, evmin=evmin, evmax=evmax, max_step=max_step)
	ub = optimize_bound(e,c0,V,p,projectionf,stepf,upper=True)
	lb = optimize_bound(e,c0,V,p,projectionf,stepf,upper=False)
	return ub, lb
