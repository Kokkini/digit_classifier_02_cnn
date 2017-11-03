import numpy as np
class FCN:
	def __init__(self, sizes):
		self.num_layers = len(sizes)
		self.sizes = sizes
		self.biases = [np.random.randn(y,1) for y in sizes[1:]]
		self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
	def forward(self, a):
		for w,b in zip(self.weights, self.biases):
			a = sigmoid(np.dot(w,a)+b)
		return a
	def backprop(self, x, y):
		#feed forward
		activation = x
		activations = [x]
		zs = []
		for b,w in zip(self.biases, self.weights):
			z = np.dot(w,activation)+b
			zs.append(z)
			activation = sigmoid(z)
			activations.append(activation)
		
		#backprop
		nabla_b = [np.zeros(b.shape) for b in self.biases]
		nabla_w = [np.zeros(w.shape) for w in self.weights]

		delta = self.cost_derivative(y, activations[-1])*sigmoid_prime(zs[-1])
		nabla_b[-1] = delta
		nabla_w[-1] = np.dot(delta,activations[-2].transpose())
		for i in xrange(2, self.num_layers):
			z = zs[-i]
			delta = np.dot(self.weights[-i+1].transpose(), delta)*sigmoid_prime(z)
			nabla_b[-i] = delta
			nabla_w[-i] = np.dot(delta, activations[-i-1].transpose())
		return [nabla_b, nabla_w]

	def SGD_single(self, training_data, test_data, numEpoches, eta):
		counter = 0
		for i in range(numEpoches):
			for x,y in training_data:
				nabla_b, nabla_w = self.backprop(x, y)
				self.biases = [-nb*eta+b for b,nb in zip(self.biases, nabla_b)]
				self.weights = [-nw*eta+w for w,nw in zip(self.weights, nabla_w)]
				counter+=1
			print("Epoch {0}: {1}".format(i, self.evaluate(test_data)))
				#print counter

	def evaluate(self, test_data):
		test_result = [(np.argmax(self.forward(x)),y) for x,y in test_data]
		return sum(int(output==y) for (output, y) in test_result)
	def cost_derivative(self, y, a):
		return (a-y)/(a*(1-a))
		
def sigmoid(z):
	return 1/(1+np.exp(-z))
def sigmoid_prime(z):
	return sigmoid(z)*(1-sigmoid(z))
