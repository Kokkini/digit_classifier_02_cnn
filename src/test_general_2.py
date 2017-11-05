import mnist_loader_enhanced as ml
training_data, validation_data, test_data = ml.load_data_wrapper_matrix()

import cnn_02 as cnn 

sizes = [28, [3,2],[3,4],[3,3]]
a = cnn.CNN(sizes)
a.feedforward(training_data[0][0])
for layer in a.layers:
	print (layer.z[0].shape)
	print (layer.p_ave)
