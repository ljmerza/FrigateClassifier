import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import VGG16

# Define the path to the new dog_images folder
gen_data = 'dog_images_formatted'

# Define the paths for the training, validation, and test data folders
train_dir = f'./{gen_data}/train'
validation_dir = f'./{gen_data}/validation'
test_dir = f'./{gen_data}/test'

# Set image size and batch size
img_height, img_width = 224, 224
batch_size = 32

# Data augmentation
train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

train_data = train_data_gen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size
)

validation_data_gen = ImageDataGenerator(rescale=1./255)
validation_data = validation_data_gen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size
)

# Transfer Learning with VGG16 base model
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))
base_model.trainable = False

model = models.Sequential([
    base_model,
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(train_data.class_indices), activation='softmax')
])

# Compile the model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Learning Rate Scheduling
def lr_schedule(epoch):
    initial_learning_rate = 0.001
    decay_steps = 1000
    decay_rate = 0.9
    return initial_learning_rate * decay_rate**(epoch // decay_steps)

lr_callback = tf.keras.callbacks.LearningRateScheduler(lr_schedule)

# Early Stopping
early_stopping = tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

# Train the model
num_epochs = 1
steps_per_epoch = train_data.samples // batch_size
validation_steps = validation_data.samples // batch_size

model.fit(
    train_data,
    epochs=num_epochs,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_data,
    validation_steps=validation_steps,
    callbacks=[lr_callback, early_stopping]
)

# Save the model
model.save('data/dog_classifier_model.h5')

# Convert and save the model to TFLite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('data/dog_classifier_model.tflite', 'wb') as f:
    f.write(tflite_model)

# Evaluate the model
test_data_gen = ImageDataGenerator(rescale=1./255)
test_data = test_data_gen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    subset='validation'
)

loss, accuracy = model.evaluate(test_data)
print(f'Test loss: {loss}, Test accuracy: {accuracy}')
