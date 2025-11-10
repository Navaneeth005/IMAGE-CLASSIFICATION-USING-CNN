from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Use absolute paths to your data folders
train_dir = r"D:\cnn_prj\data\train"
val_dir = r"D:\cnn_prj\data\validation"

# Data augmentation for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,       # randomly rotate images by 0-40 degrees
    width_shift_range=0.2,   # randomly translate images horizontally
    height_shift_range=0.2,  # randomly translate images vertically
    shear_range=0.2,         # random shearing transformations
    zoom_range=0.2,          # randomly zoom in/out
    horizontal_flip=True,    # randomly flip images horizontally
    fill_mode='nearest'      # fill missing pixels
)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# Validation data should not be augmented, only rescaled
validation_datagen = ImageDataGenerator(rescale=1./255)
validation_generator = validation_datagen.flow_from_directory(
    val_dir,
    target_size=(128, 128),
    batch_size=32,
    class_mode='binary'
)

# CNN model architecture
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train model
history = model.fit(
    train_generator,
    epochs=35,
    validation_data=validation_generator
)

# Train the model
history = model.fit(
    train_generator,
    epochs=30,                 # your chosen number of epochs
    validation_data=validation_generator
)

# Save the trained model
model.save(r"D:\cnn_prj\outputs\my_model.h5")
