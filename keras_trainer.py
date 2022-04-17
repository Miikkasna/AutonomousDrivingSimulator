
trainer = np.load('C:\\Users\\miikk\\Documents\\timeopt.npy', allow_pickle=True)[79]
train_x, train_y = NN.create_training_set(10000, trainer, 6+1)
from tensorflow.keras import Sequential, layers, losses
model = Sequential()
model.add(layers.Dense(16, input_dim=7, activation='sigmoid'))
model.add(layers.Dense(16))
model.add(layers.Dense(2))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])
model.fit(train_x, train_y, epochs=30)
weights = model.get_weights()
hybrid = NN.NeuralNetwork(7, [16, 16], 2)
hybrid.keras_weightswap(weights)