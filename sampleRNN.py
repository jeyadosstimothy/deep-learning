import numpy as np
from tensorflow import keras
from tensorflow.python.keras.models import load_model


TRAIN = False
model_file = 'models/othello.h5'
data = open('datasets/othello/othello.txt').read().lower()

uniqueChars = sorted(list(set(data)))
totalChars = len(data)
totalUniqueChars = len(uniqueChars)

charToId = {c:i for i, c in enumerate(uniqueChars)}
idToChar = {i:c for i, c in enumerate(uniqueChars)}

charWindow = 100

inputData, outputData = [], []

for i in range(totalChars - charWindow):
	currWindow = data[i:i + charWindow]
	nextChar = data[i + charWindow]
	inputData.append([charToId[c] for c in currWindow])
	outputData.append(charToId[nextChar])

inputData = np.reshape(inputData, (len(inputData), charWindow, 1))
inputData = inputData / float(totalUniqueChars)

outputData = keras.utils.to_categorical(outputData)

if TRAIN:
	model = keras.models.Sequential()
	model.add(keras.layers.LSTM(256, return_sequences=True, input_shape=(inputData.shape[1], inputData.shape[2])))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.LSTM(256))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(64, activation='relu'))
	model.add(keras.layers.Dropout(0.2))
	model.add(keras.layers.Dense(outputData.shape[1], activation='softmax'))
	model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	print(model.summary())
	model.fit(inputData, outputData, epochs=50, batch_size=32)
	model.save(model_file)
else:
	model = load_model(model_file)

randomPos = np.random.randint(0, len(inputData) - 1)
sentence = inputData[randomPos]
prediction = list([int(totalUniqueChars * i[0]) for i in sentence])
print('Seed:')
print(''.join([idToChar[i] for i in prediction]))
print('Predicted:')
for i in range(500):
	x = np.reshape(sentence, (1, len(sentence), 1))
	x = x / float(totalUniqueChars)
	pred = model.predict(x)
	index = np.argmax(pred)
	sentence = np.append(sentence, index)
	prediction.append(index)
	sentence = sentence[1:len(sentence)]

print(''.join([idToChar[i] for i in prediction]))