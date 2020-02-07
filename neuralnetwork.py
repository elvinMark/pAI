import numpy as np
def sigmoid(x,diff):
	if diff:
		return x*(1-x)
	return 1/(1+np.exp(-x))
def tanh(x,diff):
	if diff:
		return (1-x**2)/2
	return (1-np.exp(-x))/(1+np.exp(-x))
def relu(x,diff):
	if diff:
		return np.where(x>=0,1,0.001)
	return np.where(x>=0,x,0.001*x)
def linear(x,diff):
	if diff:
		return np.ones(x.shape)
	return x
class Layer:
	def __init__(self,nin,nout,act_fun = "sigmoid"):
		self.w = np.random.random([nin,nout]) #Creating the weight matrix with random values
		self.bias = np.random.random([1,nout]) #Create the bias vector (kind of the threshold)
		self.ones_matrix = None
		self.nin = nin
		self.nout = nout
		self.act_fun = act_fun
		self.in_data = None
		self.out_data = None
		self.delta = None
	def set_ones_matrix(self,N):
		self.ones_matrix = np.ones([N,1])
	def activation_function(self,x,diff=False):
		if self.act_fun == "sigmoid":
			return sigmoid(x,diff)
		elif self.act_fun == "tanh":
			return tanh(x,diff)
		elif self.act_fun == "relu":
			return relu(x,diff)
		elif self.act_fun == "linear":
			return linear(x,diff)
		else:
			return sigmoid(x,diff)
	def forward(self,in_data):
		self.in_data = in_data
		o = self.in_data.dot(self.w) + self.ones_matrix.dot(self.bias)
		self.out_data = self.activation_function(o,diff=False)
		return self.out_data
	def backward(self,err):
		self.delta = self.activation_function(self.out_data,diff=True)*err
		return self.delta.dot(self.w.T)
	def update(self,alpha=1):
		self.w = self.w - alpha*self.in_data.T.dot(self.delta)
		self.bias = self.bias - alpha*np.sum(self.delta,axis=0,keepdims=True)

class MLP:
	def __init__(self):
		self.layers = []
		self.in_data_set = None
		self.out_data_set = None
	def add_layer(self,nin,nout,act_fun="sigmoid"):
		layer = Layer(nin,nout,act_fun = act_fun)
		self.layers.append(layer)
	def forward(self,in_data):
		o = in_data
		for l in self.layers:
			l.set_ones_matrix(len(in_data))
			o = l.forward(o)
		return o
	def backward(self,err):
		o = err
		for l in reversed(self.layers):
			o = l.backward(o)
	def update(self,alpha=1):
		for l in self.layers:
			l.update(alpha=alpha)
	def set_data_set(self,in_data,out_data):
		self.in_data_set = in_data
		self.out_data_set = out_data
		N = len(in_data)
		for l in self.layers:
			l.set_ones_matrix(N)
	def train(self,alpha=1,maxIt=1000):
		for i in range(maxIt):
			o = self.forward(self.in_data_set)
			err = o - self.out_data_set
			self.backward(err)
			self.update(alpha=alpha)

