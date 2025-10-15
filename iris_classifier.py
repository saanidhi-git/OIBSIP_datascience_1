import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def get_data_frame():
    """Loads the Iris data safely."""
    try:
        df = pd.read_csv('Iris.csv')
    except FileNotFoundError:
        print("Error: Iris.csv file not found.")
        return None
    
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
    
    return df

def get_trained_model():
    """Trains the KNN model on the full dataset and returns the model, names, and features."""
    df = get_data_frame()
    if df is None:
        return None, None, None

    X = df.drop('Species', axis=1)
    y = df['Species']

    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    target_names = le.classes_ 
    feature_columns = X.columns.tolist() 

    # Train model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y_encoded)
    
    return knn, target_names, feature_columns

# This block executes ONLY when you run the script directly.
if __name__ == '__main__':
    df = get_data_frame()
    if df is None:
        exit()

    print("--- 1. Exploratory Data Analysis (EDA) ---")
    print(df.head())
    print("\nMissing values per column:\n", df.isnull().sum())

    X = df.drop('Species', axis=1)
    y = df['Species']
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)
    
    print("\n--- 2. Model Training and Evaluation ---")
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    y_pred = knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy Score: {accuracy:.4f}")
    print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Show plots when run in terminal
    plt.figure(figsize=(7, 5))
    sns.countplot(x='Species', data=df)
    plt.title('Species Distribution')
    plt.show()

    sns.pairplot(df, hue='Species', markers=["o", "s", "D"])
    plt.suptitle('Pair Plot of Iris Features by Species', y=1.02)
    plt.show()

