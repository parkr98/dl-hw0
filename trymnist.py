from uwnet import *

# 4.2: Similar to what was suggested in the iPython notebook, I trained with decreasing learning rate.
#      I also lowered the iters since the model was converging quickly and overfitting. I achieved a test accuracy of 96.84%.

def softmax_model():
    l = [make_connected_layer(784, 10, SOFTMAX)]
    return make_net(l)

def neural_net():
    l = [   make_connected_layer(784, 32, LRELU),
            make_connected_layer(32, 10, SOFTMAX)]
    return make_net(l)

print("loading data...")
train = load_image_classification_data("mnist/mnist.train", "mnist/mnist.labels")
test  = load_image_classification_data("mnist/mnist.test", "mnist/mnist.labels")
print("done")
print

print("making model...")
batch = 128
iters = 5000
rate = .01
momentum = .9
decay = .001

m = neural_net()
print("training...")
train_image_classifier(m, train, batch, 2000, .1, momentum, decay)
train_image_classifier(m, train, batch, 2000, 0.001, momentum, decay)
print("done")
print

print("evaluating model...")
print("training accuracy: %f", accuracy_net(m, train))
print("test accuracy:     %f", accuracy_net(m, test))

