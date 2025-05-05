import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define CNN Model

#test
def create_model():
    model = Sequential([
        Conv2D(32, (3,3), activation='relu', input_shape=(128, 128, 1)),
        MaxPooling2D(pool_size=(2,2)),
        Conv2D(64, (3,3), activation='relu'),
        MaxPooling2D(pool_size=(2,2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # Binary classification (Normal vs Anomaly)
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

# Load dataset
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=16,
    color_mode="grayscale",
    class_mode='binary',
    # subset='training'
)
val_generator = train_datagen.flow_from_directory(
    'dataset/',
    target_size=(128, 128),
    batch_size=16,
    color_mode="grayscale",
    class_mode='binary',
    # subset='validation'
)

# Train model
model = create_model()
model.fit(train_generator, validation_data=val_generator, epochs=10)
model.save("models/healthscan_cnn.h5")
