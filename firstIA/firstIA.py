import numpy as np

x_entrer = np.array(([3, 1.5], [2, 1], [4, 1.5], [3, 1], [3.5, 0.5], [2, 0.5], [5.5, 1], [1, 1], [2.5, 0.8]), float)
y = np.array(([1], [0], [1], [0], [1], [0], [1], [0]), float)  # 1 -> rouge, 0 -> bleu

x_entrer = x_entrer / np.amax(x_entrer, 0)

x = np.split(x_entrer, [8])[0]
xPrediction = np.split(x_entrer, [8])[1]


class Reaseau(object):
    def __init__(self):
        self.z2_delta = None
        self.o_delta = None
        self.z2_error = None
        self.delta_error = None
        self.o_error = None
        self.z3 = None
        self.z2 = None
        self.z = None
        self.inputSize = 2
        self.outputSize = 1
        self.hiddenSize = 3

        self.W1 = np.random.randn(self.inputSize, self.hiddenSize)  # 2x3
        self.W2 = np.random.randn(self.hiddenSize, self.outputSize)  # 3x1

    def forward(self, X):
        self.z = np.dot(X, self.W1)
        self.z2 = self.sigmoid(self.z)
        self.z3 = np.dot(self.z2, self.W2)
        out = self.sigmoid(self.z3)
        return out

    @staticmethod
    def sigmoid(s):
        return 1 / (1 + np.exp(-s))

    @staticmethod
    def sigmoidPrime(s):
        return s * (1 - s)

    def backward(self, X, y, o):
        self.o_error = y - o
        self.o_delta = self.o_error * self.sigmoidPrime(o)

        self.z2_error = self.o_error.dot(self.W2.T)
        self.z2_delta = self.z2_error * self.sigmoidPrime(self.z2)

        self.W1 += X.T.dot(self.z2_delta)
        self.W2 += self.z2.T.dot(self.o_delta)

    def train(self, x, y):
        o = self.forward(x)
        self.backward(x, y, o)

    def predict(self):
        print("\n\nDonnee predite : \n")
        print(f"Entree : {xPrediction}")
        print(f"Sortie : {self.forward(xPrediction)}")

        if self.forward(xPrediction) > 0.5:
            print("La fleur est ROUGE !")
        else:
            print("La fleur est BLEU !")


R = Reaseau()

for i in range(30000):
    print(f"\n# {i} \n")
    print(f"Valeur Actuelle : {x} ")
    print(f"Sortie Actuelle : {y}")
    print(f"Sortie prédite : {np.matrix.round(R.forward(x), 2)}")
    R.train(x, y)

R.predict()