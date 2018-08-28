import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator

batch_size = 16
class_mode = 'categorical'  # categorical or binary
num_classes = 2
input_dir = 'datasets/shapes'
model_file = 'models/shapes.h5'
input_shape = (200, 200, 3)
learn_rate = 0.001
epochs = 10

datagen = ImageDataGenerator(validation_split=0.2)
train_generator = datagen.flow_from_directory(input_dir,
											  target_size=input_shape[:2],
											  batch_size=batch_size,
											  subset='training',
											  class_mode=class_mode)
validation_generator = datagen.flow_from_directory(input_dir,
												   target_size=input_shape[:2],
												   batch_size=batch_size,
												   subset='validation',
											  	   class_mode=class_mode)

model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters=20, kernel_size=5, input_shape=input_shape))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Conv2D(filters=50, kernel_size=5))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))
model.add(keras.layers.MaxPooling2D())
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(500))
model.add(keras.layers.BatchNormalization())
model.add(keras.layers.Activation('relu'))

if class_mode == 'categorical':
	model.add(keras.layers.Dense(num_classes))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('softmax'))
	loss = keras.losses.categorical_crossentropy
elif class_mode == 'binary':
	model.add(keras.layers.Dense(1))
	model.add(keras.layers.BatchNormalization())
	model.add(keras.layers.Activation('sigmoid'))
	loss = keras.losses.binary_crossentropy

model.compile(optimizer=keras.optimizers.Adam(lr=learn_rate),
			  loss=loss,
			  metrics=['accuracy'])

print(model.summary())

model.fit_generator(train_generator,
					steps_per_epoch=len(train_generator),
					epochs=epochs,
					validation_data=validation_generator,
					validation_steps=len(validation_generator))

model.save(model_file)
