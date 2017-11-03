#This is all wrong in its own way. Ask yourself how many can be wrong like this.
import numpy as np
import random
class CNN:
	def __init__(self, filters, image_size, pool = (2,2)):
		'''filters is a tuple: (((3,3),2),((5,5),4),...)
		let all pool be the same for now
		the first layer doesn't have w and b, only p_ave
		'''
		self.L = len(filters)
		self.filters = filters
		self.weights = [[np.random.randn(f[0][0],f[0][1]) for j in range(f[1])] for f in filters[1:]]
		self.biases = [[random.random() for j in range(f[1])] for f in filters[1:]]
		self.zs = []
		for i in range(self.L):
			self.zs
		#create the form of zs so that we don't have to recreate z everytime
		h = image_size(0)
		w = image_size(1)
		for i in range(1,len(filters)):
			f=filters[i]
			zs.append([])
			h=h+1-f[0][0]
			w=w+1-f[0][1]
			for j in range(f[1]):
				self.zs[i-1].append(np.random.randn(w,h))

	def feedforward(self, in_array):
		p_ave = in_array
		for w,b,z in zip(self.weights,self.biases, self.zs):
			for k in range(len(w)):
				for i in range(len(z)):
					for j in range(len(z[0])):
						z[k][i][j] = (p_ave[i:i+len(w[k]), j:j+len(w[k][0])]*w[k]).sum()+b[k]
			#We still need pooling here. This is too messy, we need methods. We'll trash this.

#there are 2 options: put everything in a list. This way, it's less clear but you don't have to connect things
class ConvPoolLayer:
	def __init__(self, num_in, num_out, filter_shape, pool=[2,2]):
		self.z = [np.random.randn()]
		self.a = []
		self.p = 
		self.weights = [np.random.randn(filter_shape[0], filter_shape[1]) for i in range(num_out)]
		self.biases = [random.random() for i in range(num_out)]
		self.pool = pool
	def 

