# src/prepare_data.py
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_data(img_size=(150, 150), batch_size=32):
    train_gen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,
        zoom_range=0.2,
        shear_range=0.1,
        horizontal_flip=True
    )

    test_gen = ImageDataGenerator(rescale=1./255)

    train_data = train_gen.flow_from_directory(
        '../chest_xray/train',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    val_data = test_gen.flow_from_directory(
        '../chest_xray/val',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary'
    )

    test_data = test_gen.flow_from_directory(
        '../chest_xray/test',
        target_size=img_size,
        batch_size=batch_size,
        class_mode='binary',
        shuffle=False
    )

    return train_data, val_data, test_data
