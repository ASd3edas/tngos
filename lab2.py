import pandas as pd
import random

class Perceptron:
    def __init__(self, epoch):
        self.weights = [random.uniform(-0.1, 0.1) for _ in range(len(x_train[0]))]
        self.bias = random.uniform(-0.1, 0.1)
        self.threshold = random.uniform(-0.2, 0.2)
        self.epoch = epoch
        self.global_train = int()
        self.global_test = int()

    def predict(self, x_train):
        r = 0
        for j in range(len(self.weights)):
            r += x_train[j] * self.weights[j]
        r += self.bias
        return r

    def activation_function(self, net):
        return 1 if net >= self.threshold else 0

    def train(self, x_train, y_train):
        for epoch in range(self.epoch):
            global_error = 0
            for x, d in zip(x_train, y_train):
                net = self.predict(x)
                y = self.activation_function(net)
                if y != d:
                    global_error += 1
                    if y == 0:
                        for j in range(len(self.weights)):
                            self.weights[j] += x[j]
                        self.bias += 1
                    elif y == 1:
                        for j in range(len(self.weights)):
                            self.weights[j] -= x[j]
                        self.bias -= 1
                else:
                    self.global_train += 1
            print(f'Сейчас {global_error} ошибок')
            if epoch > 1 and global_error == 0:
                print(f'Завершил досрочно на {epoch} эпохе')
                break
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
x_train = df_train.iloc[:, :-1].values.tolist()  # все колонки кроме последней
y_train = df_train.iloc[:, -1].tolist()          # последняя колонка

df_test = pd.read_csv("test.csv", sep=',', encoding='utf-8')
x_test = df_test.iloc[:, :-1].values.tolist()
y_test = df_test.iloc[:, -1].tolist()

perceptron = Perceptron(epoch=10)
perceptron.train(x_train, y_train)
perceptron.test(x_test, y_test)