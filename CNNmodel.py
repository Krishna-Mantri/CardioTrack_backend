import numpy as np
import pandas as pd
import wfdb  # For loading MIT-BIH dataset
from wfdb import processing
import glob
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, LSTM, Dense, Dropout, Flatten, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.utils import to_categorical

# --------------------------
# 1️⃣ Load MIT-BIH Dataset (All Records)
# --------------------------
def load_mitbih_data(data_path):
    """
    Load ECG signal data and labels from all available MIT-BIH records.
    """
    records = [f[:-4] for f in os.listdir(data_path) if f.endswith(".dat")]  # Get all records
    signals, labels = [], []

    for record in records:
        try:
            # Load ECG signal and annotations
            signal, _ = wfdb.rdsamp(os.path.join(data_path, record))
            annotation = wfdb.rdann(os.path.join(data_path, record), 'atr')

            # Extract R-peaks (heartbeat locations)
            r_peaks = annotation.sample

            # Extract 200-sample segments around R-peaks
            for peak in r_peaks:
                if peak - 100 >= 0 and peak + 100 < len(signal):
                    segment = signal[peak - 100:peak + 100, 0]  # Single lead
                    signals.append(segment)
                    labels.append(annotation.symbol[np.where(r_peaks == peak)[0][0]])  # Corresponding label
        except Exception as e:
            print(f"Skipping {record} due to error: {e}")

    return np.array(signals), np.array(labels)

# --------------------------
# 2️⃣ Preprocess Data
# --------------------------
def preprocess_data(X, y):
    """
    Normalize ECG signals and encode labels.
    """
    # Normalize signal
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    return X, y_categorical, label_encoder

# --------------------------
# 3️⃣ Prepare Training & Testing Data
# --------------------------
data_path = r"C:\\Users\\Krishna\\Documents\\miniproject\\mit-bih"  # Replace with your dataset folder
X, y = load_mitbih_data(data_path)
X, y, label_encoder = preprocess_data(X, y)

# Reshape for CNN & LSTM
X = X.reshape(X.shape[0], X.shape[1], 1)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --------------------------
# 4️⃣ Build CNN + LSTM Model
# --------------------------
def build_cnn_lstm_model(input_shape, num_classes):
    model = Sequential()

    # CNN Layers
    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(BatchNormalization())  # Added for stability
    model.add(MaxPooling1D(pool_size=2))

    model.add(Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))  # Reduced dropout to retain more patterns

    model.add(Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(BatchNormalization())
    model.add(MaxPooling1D(pool_size=2))
    model.add(Dropout(0.2))

    # LSTM Layers
    model.add(LSTM(128, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dropout(0.3))

    # Fully Connected Layers
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(num_classes, activation='softmax'))  # Multi-class classification

    # Compile model with Adam optimizer
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(learning_rate=0.0005),  # Reduced learning rate for better convergence
        metrics=['accuracy']
    )

    return model

# --------------------------
# 5️⃣ Train the Model (with Checkpoint & Early Stopping)
# --------------------------
input_shape = (X_train.shape[1], 1)
num_classes = y_train.shape[1]
model = build_cnn_lstm_model(input_shape, num_classes)

# Model Saving Strategy
checkpoint = ModelCheckpoint(
    "best_ecg_cnn_lstm_model.h5",  # Save best model based on validation accuracy
    monitor='val_accuracy',
    save_best_only=True,
    mode='max'
)

# Early Stopping to Prevent Overfitting
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=5,  # Stop if val_loss doesn't improve for 5 epochs
    restore_best_weights=True
)

# Train model
history = model.fit(
    X_train, y_train,
    epochs=15,  # Increased epochs for better learning
    batch_size=32,
    validation_data=(X_test, y_test),
    callbacks=[checkpoint, early_stopping]
)

# --------------------------
# 6️⃣ Save Final Trained Model
# --------------------------
model.save("final_ecg_cnn_lstm_model.h5")

# --------------------------
# 7️⃣ Evaluate Model
# --------------------------
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"✅ Test Accuracy: {test_acc:.4f}")

# --------------------------
# 8️⃣ Plot Training Performance
# --------------------------
plt.figure(figsize=(12, 5))

# Plot Accuracy
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Model Accuracy')

# Plot Loss
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Model Loss')

plt.show()
