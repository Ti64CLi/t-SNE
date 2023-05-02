import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from math import log, exp, cos, sin
from random import random

# PI = 3.141592653589793238462643373289502

class TSNE:
	def __init__(self, **args):
		self.perplexity = args.get('perplexity', 30)
		self.dim = args.get('dim', 2)
		self.epsilon = args.get('epsilon', 5)
		self.N = args.get('N', 100)
		#self.rng = args.get('rng')
		self.iter = 0
		self._v_val = 0
		self._return_v = False
	
	# compute L2 distance between two vectors
	def L2(self, x1, x2):
		return np.linalg.norm(x1 - x2)

		# compute pairwise distance in all vectors in X
	def xtod(self, X):
		n, _ = X.shape
		dist = np.zeros((n, n))

		for i in range(n):
			for j in range(n):
				d = self.L2(X[i], X[j])
				dist[i, j] = d
				dist[j, i] = d

		return dist

	def d2p(self, D, perplexity, tol):
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
					done = True
				if num >= maxtries:
					done = True

			for j in range(row):
				P[i, j] = prow[j]
		
		Pout = np.zeros((row, row))
		for i in range(row):
			for j in range(row):
				Pout[i, j] = max((P[i, j] + P[j, i]) / (2 * row), 1e-100)
		
		return Pout

	def gaussRandom(self):
		if self._return_v:
			self._return_v = False
			return self._v_val
		
		u = 2 * random() - 1
		v = 2 * random() - 1
		r = u * u + v * v
		if r == 0 or r > 1:
			return self.gaussRandom()
		
		c = (-2 * log(r) / r) ** 0.5
		self._v_val = v * c
		self._return_v = True
		return u * c

	def randn(self, mu, std):
		return mu + self.gaussRandom() * std

	def randn2d(self, n, d):
		x = np.zeros((n, d))

		for i in range(n):
			for j in range(d):
				x[i, j] = self.randn(0, 1e-4)
		
		return x

	def costGrad(self, Y):
		N = self.N
		dim = self.dim
		P = self.P

		pmul = 4 if self.iter < 100 else 1

		Qu = np.zeros((N, N))
		qsum = 0
		for i in range(N):
			for j in range(i+1, N):
				dsum = 0
				for d in range(dim):
					dhere = Y[i][d] - Y[j][d]
					dsum += dhere * dhere
				
				qu = 1 / (1 + dsum)
				Qu[i, j] = qu
				Qu[j, i] = qu
				qsum += 2 * qu
		
		cost = 0
		grad = []
		for i in range(N):
			gsum = np.zeros(dim)
			for j in range(N):
				normedProb = max(Qu[i, j] / qsum, 1e-100)
				cost += -P[i, j] * log(normedProb)
				premult = 4 * (pmul * P[i, j] - normedProb) * Qu[i, j]
				for d in range(dim):
					gsum[d] += premult * (Y[i][d] - Y[j][d])
			
			grad.append(gsum)
		
		return (cost, np.array(grad))

	# take a set of high-dimensional points and 
	# create matrix P from them using gaussian kernel
	def initDataRaw(self, X):
		dists = self.xtod(X)
		return self.initDataDists(self.d2p(dists, self.perplexity, 1e-4))

	# take a fattened distance matrix and create matrix P from them
	# D is assumed to be squared
	def initDataDists(self, D):
		row, _ = D.shape
		self.P = self.d2p(D, self.perplexity, 1e-4)
		self.N = row
		self.initSolution()


	def initSolution(self):
		self.Y = self.randn2d(self.N, self.dim)
		self.gains = np.ones((self.N, self.dim))
		self.ystep = np.zeros((self.N, self.dim))
		self.iter = 0
	
	def getSolution(self):
		return self.Y

	def step(self):
		self.iter += 1
		N = self.N

		cost, grad = self.costGrad(self.Y)
		
		ymean = np.zeros(self.dim)
		for i in range(N):
			for d in range(self.dim):
				gid = grad[i][d]
				sid = self.ystep[i][d]
				gainid = self.gains[i][d]

				newgain = (gainid * 0.8) if gid * sid > 0 else (gainid + 0.2)
				newgain = max(newgain, 0.01)
				self.gains[i][d] = newgain

				momval = 0.5 if self.iter < 250 else 0.8
				newsid = momval * sid - self.epsilon * newgain * grad[i][d]
				self.ystep[i][d] = newsid

				self.Y[i][d] += newsid
				ymean[d] += self.Y[i][d]
		
		for i in range(N):
			for d in range(self.dim):
				self.Y[i][d] -= ymean[d] / N
		
		return cost

	## Dataset examples
	@staticmethod
	def linkData(n):
		colors = []
		points = []
		def rotate(x, y, z):
			u = x
			cos4 = cos(.4)
			sin4 = sin(.4)
			v = cos4 * y + sin4 * z
			w = -sin4 * y + cos4 * z
			return [u, v, w]
		
		for i in range(n):
			t = 2 * np.pi * i / n
			sint = sin(t)
			cost = cos(t)

			points.append(rotate(cost, sint, 0))
			colors.append("dodgerblue")
			points.append(rotate(1 + cost, 0, sint))
			colors.append("gold")
		
		return np.array(points), np.array(colors)

	@staticmethod
	def cubeData(n, dim):
		colors = []
		points = []
		for i in range(n):
			p = []
			for j in range(dim):
				p.append(random())
			points.append(p)
			colors.append("dodgerblue")
		return np.array(points), np.array(colors)

	@staticmethod
	def threeClustersData(n, dim=50):
		colors = [ ["gold", "dodgerblue", "red"][i%3] for i in range(3 * n)]
		points = np.zeros((3 * n, dim))
		for i in range(n):
			for j in range(dim):
				points[3 * i, j] = np.random.normal()
				points[3 * i + 1, j] = np.random.normal() + (10 if j == 0 else 0)
				points[3 * i + 2, j] = np.random.normal() + (50 if j == 0 else 0)
				
		return np.array(points), np.array(colors)

def animate(tsne, fig, C):
	# Construct the scatter which we will update during animation
	# as the raindrops develop.
	ax = fig.add_subplot(1, 2, 2)

	Y = tsne.getSolution()
	ax.set_ylim((-5, 5))
	ax.set_xlim((-5, 5))
	scat = ax.scatter(Y[:, 0], Y[:, 1], c=C)
					#s=rain_drops['size'], lw=0.5, edgecolors=rain_drops['color'],
					#facecolors='none')

	def update(frame_number):
		# Update the scatter collection, with the new colors, sizes and positions.
		#scat.set_edgecolors(rain_drops['color'])
		#scat.set_sizes(rain_drops['size'])
		tsne.step()
		Y = tsne.getSolution()
		scat.set_offsets(Y)
		ax.set_title(f'Iteration n°{tsne.iter}')
		return scat, 

	# Construct the animation, using the update function as the animation director.
	animation = FuncAnimation(fig, update, interval=1, frames=500)
	plt.show()

if __name__ == '__main__':
	tsne = TSNE()
	fig = plt.figure()

	axData = fig.add_subplot(1, 2, 1, projection='3d')

	D, C = tsne.threeClustersData(50, 3)
	axData.scatter(D[:, 0], D[:, 1], D[:, 2], c=C)

	
	tsne.initDataRaw(D)
	anim = animate(tsne, fig, C)

	plt.show()