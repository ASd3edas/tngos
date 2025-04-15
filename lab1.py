import pandas as pd
import random


class Perceptron:
    def __init__(self, learning_rate, epoch):
        self.weights = [random.uniform(0.376, 0.756) for _ in range(len(x_train[0]))] # 0.376, 0.756
        self.bias = random.uniform(-0.241, 0.278) # -0.241, 0.278
        self.epoch = epoch
        self.global_train = int()
        self.global_test = int()
        self.learning_rate = learning_rate



    def predict(self, x_train):
        r = 0
        for j in range(len(self.weights)):
            r += x_train[j] * self.weights[j]
        r += self.bias
        return r

    def activation_function(self, net):
        return 1 if net >= 0 else 0

    def train(self, x_train, y_train):
        for epoch in range(self.epoch):
            global_error = int()

            for x, d in zip(x_train, y_train):
                net = self.predict(x)
                y = self.activation_function(net)
                error = 0.5 * (d - y) ** 2  # MSE
                global_error += error/len(y_train)

                delta = -(d - y) * (1 if net > 0 else 0)
                for j in range(len(self.weights)):
                    self.weights[j] -= self.learning_rate * delta * x[j]
                self.bias -= self.learning_rate * delta
                if y == d:
                    self.global_train += 1

            print(f'Эпоха {epoch}')
        print(f'Train Accuracy -> {(self.global_train / (len(x_train) * self.epoch)) * 100:.2f}%')

    def test(self, x_test, y_test):
        for x, d in zip(x_test, y_test):
            y = self.activation_function(self.predict(x))
            print(f"{y=}, {d=}")
            if y == d:
                self.global_test += 1
        print(f'Train Accuracy -> {(self.global_train / (len(x_train) * self.epoch)) * 100:.2f}%')
        print(f"Test Accuracy: {self.global_test / len(x_test) * 100:.2f}%")

df_train = pd.read_csv("train1000.csv", sep=',', encoding='utf-8')
x_train = df_train.iloc[:, :-1].values.tolist()
y_train = df_train.iloc[:, -1].tolist()

df_test = pd.read_csv("test.csv", sep=',', encoding='utf-8')
x_test = df_test.iloc[:, :-1].values.tolist()
y_test = df_test.iloc[:, -1].tolist()

perceptron = Perceptron(epoch=2634, learning_rate=0.00001) # 2634, 0.00001 // 80%+
perceptron.train(x_train, y_train)
perceptron.test(x_test, y_test)




