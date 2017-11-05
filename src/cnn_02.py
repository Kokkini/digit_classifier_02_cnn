import numpy as np 
import random

class CNN:
	def __init__(self, sizes, pool = 2):
		#in layers, p of the first layer is the input image, the last layer will be dots, each dot connect to a single sigmoid neuron in the fcn
		#thus the number of z will be 1 less than the number of layers
		#let's just do everything square
		#let's every stride = 1
		#the form of sizes is [size_image, [size_of_filter, num_filters],[size_of_filter, num_filters],[...],...]
		self.sizes = sizes
		self.layers = []
		self.image_size = sizes[0]
		self.pool = pool
		#We'll store every z so that we don't have to create them again
		self.create_layers(sizes)

	def create_layers(self,sizes):
		image_size = sizes[0]
		#first layer
		self.layers.append(ConvPoolLayer((0,1), image_size*self.pool)) #in the first layer, size_z=image_size*pool so that size_p=image_size
		#the rest of layers
		for size in sizes[1:]:
			size_z = len(self.layers[-1].p_ave) + 1 - size[0]
			self.layers.append(ConvPoolLayer(size, size_z))


	def feedforward(self, in_matrix):
		#feed in first layer
		self.layers[0].p_ave = in_matrix
		#feed through the rest of the layers
		for i in range(1,len(self.layers)):
			self.layers[i].feedforward(self.layers[i-1].p_ave)
		
	def sgd(self): ########################### HERE is where we gave up ##################################################
		#We won't connect the end of this to the beginning of fcn here
		#We'll just take the output from fcn.sgd() to be the input here

class ConvPoolLayer:
	def __init__(self, size, size_z, pool = 2):
		#ConvPoolLayer will have input and output that handle everything.
		#z: after applying weights and biases to input
		#a: after applying ReLU to z
		#p: after applying pooling to a
		#p_ave: average of the p's
		self.size_filter, self.num_filters = size
		self.weights = [np.random.randn(self.size_filter,self.size_filter) for i in range(self.num_filters)]
		self.biases = [random.random() for i in range(self.num_filters)]
		self.pool = pool
		self.size_z = size_z;
		self.z = [np.zeros((size_z,size_z)) for i in range(self.num_filters)]
		self.a = [np.zeros((size_z,size_z)) for i in range(self.num_filters)]

		size_p = (int) (size_z/pool)
		self.size_p = size_p
		self.p = [np.zeros((size_p,size_p)) for i in range(self.num_filters)]
		self.p_ave = np.zeros((size_p, size_p))

	def feedforward(self, in_matrix):
		self.p_ave.fill(0)
		for i in range(self.num_filters):
			self.update_pi(in_matrix, i)
			self.p_ave = self.p_ave + self.p[i]/self.num_filters


	def update_zi_ai(self, in_matrix, i):
		for a in range(self.size_z):
			for b in range(self.size_z):
				self.z[i][a][b] = (in_matrix[a:a+self.size_filter,b:b+self.size_filter]*self.weights[i]).sum()+self.biases[i]
				self.a[i][a][b] = max(self.z[i][a][b],0)

	def update_pi(self, in_matrix, i):
		self.update_zi_ai(in_matrix,i)
		for a in range(self.size_p):
			for b in range(self.size_p):
				self.p[i][a][b] = np.max(self.a[i][a*self.pool:(a+1)*self.pool, b*self.pool:(b+1)*self.pool])


'''
Where to store z? If we store z in the ConvPoolLayer, how to init it? Give the size of z in the constructor.

'''