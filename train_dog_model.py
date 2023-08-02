import tensorflow as tf
assert tf.__version__.startswith('2')

from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the path to the new dog_images folder
gen_data = 'dog_images_formatted'

# Define the paths for the training, validation, and test data folders
train_dir = f'./{gen_data}/train'
validation_dir = f'./{gen_data}/validation'
test_dir = f'./{gen_data}/test'

# Set image size and batch size
img_height, img_width = 224, 224
batch_size = 32

# Create ImageDataGenerator for training, validation, and test data
data_gen = ImageDataGenerator(rescale=1./255)

train_data = data_gen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size
)

validation_data = data_gen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size
)

test_data = data_gen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size
)

# Build the model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(img_height, img_width, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Train the model
num_epochs = 10
steps_per_epoch = train_data.samples // batch_size
validation_steps = validation_data.samples // batch_size

model.fit(
    train_data,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_data,
    validation_steps=validation_steps
)

# Save the model
model.save('data/dog_classifier_model.h5')

# Convert and save the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('data/dog_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Evaluate the model
test_data = data_gen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation'
)

loss, accuracy = model.evaluate(test_data)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
