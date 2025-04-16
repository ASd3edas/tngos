import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Nadam


# W1 =  {0.4 -0.7; 1.2 0.6; 0.1 0.5; -1.4 0.5}
# W2 =  {-0.8;0.3;0.5}

# Архитектура модели: 3-2-1, а весов на 4-3-1


X = np.array([[0.4, -0.7, 1.3]])
Y = np.array([[0.7]])

W1 = np.array([[0.4, -0.7], [1.2, 0.6], [0.1, 0.5]])
W2 = np.array([[-0.8], [0.3]])

optimizer = Nadam(learning_rate=0.2)
model = Sequential()
model.add(Dense(2, input_dim=3, activation='sigmoid', use_bias=False))
model.add(Dense(1, activation='sigmoid', use_bias=False))

print(model.layers)

model.layers[0].set_weights([W1])
model.layers[1].set_weights([W2])
model.compile(loss='mean_squared_error', optimizer=optimizer)
model.fit(X, Y, epochs=33, verbose=2) # nadam, 33ep

loss = model.evaluate(X, Y)

pred = model.predict(X)
print(f"Predicted value: {pred[0][0]:.4f}")
