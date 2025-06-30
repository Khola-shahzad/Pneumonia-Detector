# src/train_model.py

from prepare_data import load_data
from build_model import build_cnn_model
import matplotlib.pyplot as plt

def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(acc, label='Train Accuracy')
    plt.plot(val_acc, label='Val Accuracy')
    plt.legend()
    plt.title("Accuracy")

    plt.subplot(1, 2, 2)
    plt.plot(loss, label='Train Loss')
    plt.plot(val_loss, label='Val Loss')
    plt.legend()
    plt.title("Loss")

    plt.show()

def main():
    train_data, val_data, test_data = load_data()
    model = build_cnn_model()
    
    # ✅ Force build the model to define input/output tensors
    model.build(input_shape=(None, 150, 150, 3))

    history = model.fit(
        train_data,
        epochs=10,
        validation_data=val_data
    )

    model.save("../saved_models/pneumonia_cnn_model.keras")
    plot_history(history)

if __name__ == '__main__':
    main()
