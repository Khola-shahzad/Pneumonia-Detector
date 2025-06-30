# src/evaluate_model.py

from tensorflow.keras.models import load_model
from prepare_data import load_data
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def main():
    model = load_model('../saved_models/pneumonia_cnn_model.keras')
    _, _, test_data = load_data()

    predictions = model.predict(test_data)
    threshold = 0.6
    y_pred = (predictions > threshold).astype(int).reshape(-1)

    y_true = test_data.classes

    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=['Normal', 'Pneumonia']))

    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=['Normal', 'Pneumonia'],
                yticklabels=['Normal', 'Pneumonia'], cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()

if __name__ == '__main__':
    main()
