import numpy as np
import pandas as pd

file_path = "C:/Users/artdude/Documents/conda_spyder/mnist/mnist_test.csv"
df = pd.read_csv(file_path, header=None)
Y = df.iloc[1:, 0].values
X = df.iloc[1:, 1:].values.astype("float32") / 255.0


def sigmoid(z): return 1 / (1 + np.exp(-z))
def relu(z): return np.maximum(0, z)
def softmax(z): 
    exp_z = np.exp(z - np.max(z, axis=1, keepdims=True))
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)
def linear(z): return z

def sigmoid_derivative(x): return sigmoid(x) * (1 - sigmoid(x))
def relu_derivative(x): return (x > 0).astype(float)
def linear_derivative(x): return np.ones_like(x)


activations = {
    'relu': relu,
    'sigmoid': sigmoid,
    'softmax': softmax,
    'linear': linear
}
activations_derivative = {
    'relu': relu_derivative,
    'sigmoid': sigmoid_derivative,
    'softmax': None,
    'linear': linear_derivative
}

# Losses
def binary_crossentropy(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

def binary_crossentropy_grad(y_true, y_pred):
    return y_pred - y_true

def categorical_crossentropy(y_true, y_pred):
    eps = 1e-8
    y_pred = np.clip(y_pred, eps, 1 - eps)
    return -np.sum(y_true * np.log(y_pred)) / y_true.shape[0]

def categorical_crossentropy_grad(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

def mean_squared_error(y_true, y_pred):
    return 0.5 * np.sum((y_true - y_pred) ** 2) / y_true.shape[0]

def mean_squared_error_grad(y_true, y_pred):
    return (y_pred - y_true) / y_true.shape[0]

activations_gradient = {
    'sigmoid': (binary_crossentropy, binary_crossentropy_grad),
    'softmax': (categorical_crossentropy, categorical_crossentropy_grad),
    'linear': (mean_squared_error, mean_squared_error_grad)
}


class Dense:
    def forward_prop(self, a_in, W, b):
        return np.dot(a_in, W.T) + b

    def back_prop(self, da, a_prev, W, b, z, activation):
        dz = da if activation == 'softmax' else da * activations_derivative[activation](z)
        dw = np.dot(dz.T, a_prev) / dz.shape[0]
        db = np.mean(dz, axis=0)
        da_prev = np.dot(dz, W)
        return dw, db, da_prev

# Sequential network
class Sequential:
    def __init__(self):
        self.num_layers = 0
        self.layersList = {}
        self.cache = {}
        self.cache_z = {}

    def addInput(self, X):
        self.layersList[0] = X
        self.cache[0] = X
        self.num_layers = 1

    def addDense(self, num_input, num_features, bias=True, activation=None):
        W = np.random.randn(num_features, num_input) * np.sqrt(2. / num_input)
        b = np.random.randn(num_features,) if bias else np.zeros((num_features,))
        self.layersList[self.num_layers] = (W, b, activation)
        self.num_layers += 1

    def compile(self, x, y_true):
        DenseObject = Dense()
        self.cache[0] = x
        for i in range(1, self.num_layers):
            W, b, activation_name = self.layersList[i]
            a_in = self.cache[i - 1]
            z = DenseObject.forward_prop(a_in, W, b)
            self.cache_z[i] = z
            self.cache[i] = activations[activation_name](z)

        i = self.num_layers - 1
        activation = self.layersList[i][2]
        loss_func, grad_func = activations_gradient[activation]
        loss = loss_func(y_true, self.cache[i])
        da = grad_func(y_true, self.cache[i])

        gradients = {}
        for i in reversed(range(1, self.num_layers)):
            W, b, activation_name = self.layersList[i]
            a_prev = self.cache[i - 1]
            z = self.cache_z[i]
            dw, db, da = Dense().back_prop(da, a_prev, W, b, z, activation_name)
            gradients[i] = {'dw': dw, 'db': db}
        return gradients, loss

    def update_parameters(self, W, b, dw, db, learning_rate=0.01):
        return W - learning_rate * dw, b - learning_rate * db

    def compileForward(self, X):
        self.cache[0] = X
        for i in range(1, self.num_layers):
            W, b, activation_name = self.layersList[i]
            z = Dense().forward_prop(self.cache[i - 1], W, b)
            self.cache_z[i] = z
            self.cache[i] = activations[activation_name](z)
        return self.cache, self.cache_z

    def predict(self, X):
        outputs, _ = self.compileForward(X)
        return np.argmax(outputs[self.num_layers - 1], axis=1)

    def fit(self, X_val, X_train, y_val, y_train, epoch=200, learning_rate=0.01, batch_size=32):
        num_inputs = X_train.shape[0]
        for ep in range(epoch):
            indices = np.arange(num_inputs)
            np.random.shuffle(indices)
            X_train = X_train[indices]
            y_train = y_train[indices]
            epoch_loss = 0

            for start in range(0, num_inputs, batch_size):
                end = min(start + batch_size, num_inputs)
                x_batch = X_train[start:end]
                y_batch = y_train[start:end]
                gradients, loss = self.compile(x_batch, y_batch)
                epoch_loss += loss
                for i in range(1, self.num_layers):
                    W, b, activation = self.layersList[i]
                    dw = gradients[i]['dw']
                    db = gradients[i]['db']
                    W, b = self.update_parameters(W, b, dw, db, learning_rate)
                    self.layersList[i] = (W, b, activation)
            print(f"Epoch {ep+1}/{epoch} - Loss: {epoch_loss:.4f}")
