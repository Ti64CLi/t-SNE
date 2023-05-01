import numpy as np
from math import log, exp

# compute L2 distance between two vectors
def L2(x1, x2):
	return np.linalg.norm(x1 - x2)

# compute pairwise distance in all vectors in X
def xtod(X):
	n, _ = X.shape
	dist = np.zeros((n, n))

	for i in range(n):
		for j in range(n):
			d = L2(X[i], X[j])
			dist[i, j] = d
			dist[j, i] = d

	return dist

def d2p(D, perplexity, tol):
	row, column = D.shape
	assert row == column

	Htarget = log(perplexity)
	P = np.zeros((row, row))
	prow = np.zeros(row)

	for i in range(row):
		betamin = -float('inf')
		betamax = float('inf')
		beta = 1
		done = False
		maxtries = 50

		num = 0
		while not done:
			# compute entropy and kernel row with beta precision
			psum = 0
			for j in range(row):
				pj = exp(- D[i, j] * beta)
				if i == j:
					pj = 0

				prow[j] = pj
				psum += pj
			
			# normalize p and compute entropy
			Hhere = 0
			for j in range(row):
				pj = prow[j] / psum
				prow[j] = pj
				if pj > 1e-7:
					Hhere -= pj * log(pj)
			
			# adjust beta based on result
			if Hhere > Htarget:
				betamin = beta
				if betamax == float('inf'):
					beta *= 2
				else:
					beta = (beta + betamax) / 2
			else:
				betamax = beta
				if betamin == -float('inf'):
					beta /= 2
				else:
					beta = (beta + betamin) / 2
			
			num += 1
			if abs(Hhere - Htarget) < tol:
				done = true
			if num >= maxtries:
				done = true

		for j in range(row):
			P[i, j] = prow[j]
	
	Pout = np.zeros((row, row))
	for i in range(row):
		for j in range(row):
			Pout[i, j] = max((P[i, j] + P[j, i]) / (2 * row), 1e-100)
	
	return Pout

# take a set of high-dimensional points and 
# create matrix P from them using gaussian kernel
def initDataRaw(X, perplexity):
	dists = xtod(X)
	P = d2p(dists, perplexity, 1e-4)
	return P

# take a fattened distance matrix and create matrix P from them
# D is assumed to be squared
def initDataDists(D, perplexity):
	row, _ = D.shape
	P = d2p(D, perplexity, 1e-4)
	return P

