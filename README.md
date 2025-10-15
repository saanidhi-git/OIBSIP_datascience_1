# 🌸 Iris Flower Classification with K-Nearest Neighbors (KNN)

## Table of Contents
1.  [Project Overview](#project-overview)
2.  [Algorithm Used (KNN)](#algorithm-used-knn)
3.  [Dataset](#dataset)
4.  [Project Structure](#project-structure)
5.  [Setup and Installation](#setup-and-installation)
6.  [How to Run the Streamlit App](#how-to-run-the-streamlit-app)
7.  [Results and Visualization](#results-and-visualization)
8.  [Conclusion](#conclusion)

***

## 1. Project Overview

This project demonstrates a classic Machine Learning classification task: identifying the species of an **Iris flower** based on its physical measurements. We utilize the **K-Nearest Neighbors (KNN)** algorithm to build a robust model and package the solution in an interactive web application using **Streamlit**.

### Key Features:
* **Data Analysis:** Exploratory Data Analysis (EDA) of the Iris dataset.
* **Model Training:** KNN implementation using `scikit-learn`.
* **Interactive App:** A Streamlit interface for real-time classification input.

***

## 2. Algorithm Used (KNN)

The core of this project is the **K-Nearest Neighbors (KNN)** algorithm.

* **How it Works:** KNN is a non-parametric, lazy learning algorithm. It classifies a new data point based on the majority class among its $K$ nearest neighbors. The 'distance' (Euclidean distance is typically used) is calculated between the new point and all existing data points to find the closest ones.
* **Hyperparameter:** The value of **K** (the number of neighbors) was chosen to be **[Insert your K value, e.g., 5]** after initial testing showed optimal performance.

***

## 3. Dataset

This project uses the famous **Iris flower dataset**, which is often called the "Hello World" of Machine Learning.

| Feature | Description | Unit |
| :--- | :--- | :--- |
| `sepal_length` | Length of the sepal | cm |
| `sepal_width` | Width of the sepal | cm |
| `petal_length` | Length of the petal | cm |
| `petal_width` | Width of the petal | cm |
| `species` | The target class (Setosa, Versicolor, or Virginica) | N/A |

***

## 4. Project Structure

The repository is organized as follows:
iris-classification-knn/
├── .gitignore
├── README.md
├── requirements.txt           # Lists all necessary Python libraries
├── iris_classifier.py         # Main ML code: loads data, trains KNN, saves model.
└── streamlit_app.py           # Streamlit code for the interactive web interface.

***

## 5. Setup and Installation

To run this project locally, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourUsername/iris-classification-knn.git](https://github.com/YourUsername/iris-classification-knn.git)
    cd iris-classification-knn
    ```

2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required libraries:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the ML Training Script:** This will train the KNN model and create the necessary artifacts (like a serialized model file).
    ```bash
    python iris_classifier.py
    ```

***

## 6. How to Run the Streamlit App

The project includes an interactive Streamlit application for demonstration.

1.  **Ensure you have completed the Setup steps above.**

2.  **Run the Streamlit app:**
    ```bash
    streamlit run streamlit_app.py
    ```

3.  The app will automatically open in your web browser at a local address (usually `http://localhost:8501`).

### 📷 Streamlit Application Preview

<p align="center">
 <img width="1910" height="825" alt="Screenshot 2025-10-15 202057" src="https://github.com/user-attachments/assets/930cd04f-3cc5-4255-9f92-dbe435aa5482" />
  <br>
  <em>Figure 1: Streamlit App Interface showing sliders for input features and the predicted species.</em>
</p>

***

## 7. Results and Visualization

### Model Performance

The K-Nearest Neighbors (KNN) classifier achieved the following performance metrics on the test set:

* **Accuracy:** **[Insert your calculated Accuracy Score]%**

### Data Visualization

A key step in classification is visualizing the data to understand class separability. The pairplot below illustrates how the three species cluster based on the features.

<p align="center">
 <img width="1881" height="949" alt="IFC1" src="https://github.com/user-attachments/assets/5a166107-921c-4faf-84b8-206434445aaf" />
  <br>
  <em>Figure 2: Scatter plot of Petal Length vs. Petal Width, clearly separating the three Iris species, confirming the data's separability for the KNN model.</em>
</p>

***

## 8. Conclusion

The KNN algorithm proved highly effective for classifying the Iris species, achieving high accuracy. The Streamlit app provides a simple, intuitive way to interact with the trained model, making this a complete and accessible Machine Learning project for demonstration and learning.
