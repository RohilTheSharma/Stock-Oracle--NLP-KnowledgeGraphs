import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import BatchNormalization
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import lime
import lime.lime_tabular

from training import StockTrendPredictor


class CNNModel:
    def __init__(self, input_shape, num_classes=3):
        self.model = Sequential([
            Conv1D(64, kernel_size=3, activation='relu', padding='same', input_shape=input_shape),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            Conv1D(128, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            Conv1D(256, kernel_size=3, activation='relu', padding='same'),
            BatchNormalization(),
            MaxPooling1D(pool_size=2),

            Flatten(),
            Dense(256, activation='relu'),
            Dropout(0.6),
            Dense(num_classes, activation='softmax')
        ])

        # Compile with a lower learning rate
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

    def prob(self, data):
        if len(data.shape) == 2:
            data = np.expand_dims(data, axis=2)
        return self.model.predict(data)

    def feature_names(self):
        # Return the list of feature names used in your model
        return self.feature_names

    def train_and_evaluate(self, X_train, X_test, y_train, y_test):
        # Early stopping
        early_stopping = EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5
        )

        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_split=0.2,
            callbacks=[early_stopping, reduce_lr],
            verbose=1
        )

        # Evaluate on test data
        y_pred = self.model.predict(X_test).argmax(axis=1)

        # Calculate metrics
        print("\nCNN Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Downward', 'Neutral', 'Upward']))

        accuracy = accuracy_score(y_test, y_pred)
        print(f"\nCNN Test Set Accuracy: {accuracy:.4f}")

        # Plot confusion matrix
        self.plot_confusion_matrix(y_test, y_pred)

        # Plot performance metrics
        self.plot_performance_metrics(y_test, y_pred)

        return history

    def plot_confusion_matrix(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Downward', 'Neutral', 'Upward'],
                    yticklabels=['Downward', 'Neutral', 'Upward'], ax=ax)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title('Confusion Matrix - CNN')
        plt.tight_layout()
        fig.savefig('confusion_matrix_CNN.png')
        plt.close(fig)

    def plot_performance_metrics(self, y_test, y_pred):
        performance_metrics = classification_report(
            y_test, y_pred, target_names=['Downward', 'Neutral', 'Upward'], output_dict=True
        )
        metrics_df = pd.DataFrame(performance_metrics).transpose()

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.heatmap(metrics_df.iloc[:-3][['precision', 'recall', 'f1-score']],
                    annot=True, cmap='YlOrRd', ax=ax)
        ax.set_title('CNN Performance Metrics')
        plt.tight_layout()
        fig.savefig('performance_metrics_CNN.png')
        plt.close(fig)


def main():
    # Initialize predictor for CNN
    predictor = StockTrendPredictor()
    X, y = predictor.load_data('News+Stock data.csv', 'final_features.csv')

    # Ensure X can be reshaped for CNN
    if X is None or y is None:
        print("Failed to load data. Exiting...")
        return

    # Reshape data for CNN
    # Assuming X has shape (samples, features) and converting to (samples, timesteps, features)
    X_reshaped = np.expand_dims(X.values, axis=2)  # Reshape for Conv1D (add "channels" dimension)
    input_shape = X_reshaped.shape[1:]  # (timesteps, features)

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_reshaped, y, test_size=0.2, random_state=42, stratify=y
    )

    # Initialize and train CNN
    cnn_model = CNNModel(input_shape=input_shape, num_classes=3)
    cnn_history = cnn_model.train_and_evaluate(X_train, X_test, y_train, y_test)

    # Save CNN model
    cnn_model.model.save('cnn_model.h5')
    print("CNN model saved as 'cnn_model.h5'")

    # Create LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X.values,  # Original non-reshaped data
        mode='classification',
        training_labels=y,
        feature_names=X.columns,
        class_names=['Downward', 'Neutral', 'Upward']
    )

    # Get explanation for a specific instance
    i = 1  # Index of instance to explain
    exp = explainer.explain_instance(
        X.iloc[i].values,  # Use non-reshaped data
        cnn_model.prob,
        num_features=len(X.columns)
    )

    # Visualize the explanation
    exp.show_in_notebook()  # If using Jupyter notebook
    # Or save to file:
    exp.save_to_file('lime_explanation.html')


if __name__ == "__main__":
    main()