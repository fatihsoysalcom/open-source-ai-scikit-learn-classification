import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# This example demonstrates a simple machine learning workflow using scikit-learn,
# a widely used open-source library for machine learning in Python.
# It embodies the spirit of "Open Source AI" by making algorithms, models,
# and tools accessible and modifiable, aligning with the article's theme.

def main():
    print("--- Open Source AI Example: Simple Classification ---")

    # 1. Data Generation (simulating a 'dataset' mentioned in the article)
    # Open source AI often relies on publicly available or easily generated datasets.
    X, y = make_classification(
        n_samples=100,      # 100 data points
        n_features=2,       # 2 features for simplicity
        n_informative=2,    # Both features are informative
        n_redundant=0,      # No redundant features
        n_clusters_per_class=1, # One cluster per class
        random_state=42     # For reproducibility
    )
    print(f"\nGenerated dataset with {X.shape[0]} samples and {X.shape[1]} features.")
    print(f"First 5 samples (features):\n{X[:5]}")
    print(f"First 5 labels:\n{y[:5]}")

    # 2. Data Splitting
    # Splitting data into training and testing sets is a standard practice.
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    print(f"\nTraining data size: {X_train.shape[0]} samples")
    print(f"Testing data size: {X_test.shape[0]} samples")

    # 3. Model Definition and Training (illustrating an 'algorithm' and 'model')
    # DecisionTreeClassifier is an open-source algorithm readily available in scikit-learn.
    # The trained 'classifier' object is our AI 'model'.
    print("\nTraining a Decision Tree Classifier...")
    classifier = DecisionTreeClassifier(random_state=42)
    classifier.fit(X_train, y_train)
    print("Model training complete.")

    # 4. Prediction
    # Using the trained model to make predictions on new, unseen data.
    y_pred = classifier.predict(X_test)
    print(f"\nFirst 5 actual test labels: {y_test[:5]}")
    print(f"First 5 predicted labels:   {y_pred[:5]}")

    # 5. Evaluation
    # Open source tools allow for transparent evaluation of model performance.
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on test set: {accuracy:.2f}")

    # 6. Making a prediction for a new, arbitrary data point
    # This demonstrates how the trained 'model' can be used in practice.
    new_sample = np.array([[0.5, -0.5]]) # Example new data point
    new_prediction = classifier.predict(new_sample)
    print(f"\nPrediction for new sample {new_sample[0]}: Class {new_prediction[0]}")

    print("\n--- End of Example ---")

if __name__ == "__main__":
    main()
