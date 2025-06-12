import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_score
import numpy as np
from sklearn.model_selection import learning_curve

from main import rf_model, knn_model, dt_model, nb_model, X_train, y_train, X_test, y_test

class ModelVisualizer:
    def __init__(self, models, X_train, y_train, X_test, y_test):
        self.models = models
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

    def plot_learning_curve(self, model, model_name, ax):
        train_sizes, train_scores, valid_scores = learning_curve(
            model, self.X_train, self.y_train, cv=3, scoring='accuracy', n_jobs=-1
        )
        train_mean = np.mean(train_scores, axis=1)
        valid_mean = np.mean(valid_scores, axis=1)

        ax.plot(train_sizes, train_mean, label="Training Accuracy", color="blue", marker="o")
        ax.plot(train_sizes, valid_mean, label="Validation Accuracy", color="red", marker="o")
        ax.set_title(f'Learning Curve: {model_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('Training Set Size', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid()

    def plot_cross_validation_metrics(self, model, model_name, ax):
        cv_scores = cross_val_score(model, self.X_train, self.y_train, cv=3, scoring='accuracy')
        train_accuracy = model.score(self.X_train, self.y_train)
        test_accuracy = model.score(self.X_test, self.y_test)

        ax.plot([1, 2, 3], cv_scores, label="Cross-Validation Accuracy", color="purple", marker="o")
        ax.axhline(train_accuracy, color="green", linestyle="--", label="Train Accuracy")
        ax.axhline(test_accuracy, color="orange", linestyle="--", label="Test Accuracy")
        ax.set_title(f'Train & Validation Metrics: {model_name}', fontsize=16, fontweight='bold')
        ax.set_xlabel('CV Fold', fontsize=14)
        ax.set_ylabel('Accuracy', fontsize=14)
        ax.legend(fontsize=12)
        ax.grid()

    def plot_model_metrics(self):
        fig, axes = plt.subplots(len(self.models), 3, figsize=(24, 7 * len(self.models)))

        class_labels = ["healthy", "late", "early"]

        for i, (model_name, model) in enumerate(self.models.items()):
            # Test set predictions
            y_pred = model.predict(self.X_test)
            cm = confusion_matrix(self.y_test, y_pred)

            # Confusion Matrix
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                        xticklabels=class_labels,
                        yticklabels=class_labels, ax=axes[i, 0])
            axes[i, 0].set_title(f'Confusion Matrix: {model_name}', fontsize=16, fontweight='bold')
            axes[i, 0].set_xlabel('Predicted', fontsize=14)
            axes[i, 0].set_ylabel('True', fontsize=14)

            # Learning Curve
            self.plot_learning_curve(model, model_name, axes[i, 1])

            # Cross-validation and Train-Test Accuracy
            self.plot_cross_validation_metrics(model, model_name, axes[i, 2])

        plt.tight_layout(rect=[0, 0, 1, 0.97])
        plt.suptitle('Model Performance Metrics', fontsize=18, fontweight='bold')
        plt.show()


models = {
    "Random Forest": rf_model,
    "KNN": knn_model,
    "Decision Tree": dt_model,
    "Naive Bayes": nb_model
}

visualizer = ModelVisualizer(models, X_train, y_train, X_test, y_test)
visualizer.plot_model_metrics()