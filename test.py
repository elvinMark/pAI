import neuralnetwork as nn
import numpy as np 

indata = np.array([[0,0],[0,1],[1,0],[1,1]])
outdata = np.array([[-1],[2],[3],[-4]])

mlp = nn.MLP()
mlp.add_layer(2,5,act_fun="relu")
mlp.add_layer(5,3,act_fun="relu")
mlp.add_layer(3,1,act_fun="linear")
mlp.set_data_set(indata,outdata)
mlp.train(alpha=0.01,maxIt=3000)

print(mlp.forward(indata[:4]))