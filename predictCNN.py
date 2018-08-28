from tensorflow.python.keras.models import load_model
import cv2, os
import numpy as np

model_file = 'models/shapes.h5'
loss = 'categorical_crossentropy'
test_dir = 'test'
test_files = ['circle.png', 'square.png', 'star.png', 'triangle.png']
input_shape = (200, 200)


model = load_model(model_file)

model.compile(loss=loss,
              optimizer='adam',
              metrics=['accuracy'])


def getPath(filename):
	return os.path.join(test_dir, filename)


for filename in test_files:
	img = cv2.imread(getPath(filename))
	img = cv2.resize(img,input_shape)
	img = np.reshape(img,[1,*input_shape,3])
	classes = model.predict_classes(img)
	print(classes)