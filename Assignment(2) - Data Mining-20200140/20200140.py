import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QTextEdit
import pandas as pd
import numpy as np


# craete function for dectect outliers using IQR method is send data have two cloumn  name movie name and IMDB rating
def detect_outliers_iqr(data):
    # Calculate the first quartile (Q1) and third quartile (Q3)
    Q1 = data["IMDB Rating"].quantile(0.25)
    Q3 = data["IMDB Rating"].quantile(0.75)

    # Calculate the interquartile range (IQR)
    IQR = Q3 - Q1

    # Define the lower and upper bounds
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR

    # Get the outliers
    outliers = data[(data["IMDB Rating"] < lower_bound) | (data["IMDB Rating"] > upper_bound)]

    return outliers["Movie Name"].values


def k_means_clustering(X, k, max_iter=100000, tol=1e-4):
    # Initialize centroids randomly
    np.random.seed(42)
    centroids = X[np.random.choice(X.shape[0], k, replace=False)]

    #print(f"Initial centroids: {centroids}")
    
    prev_centroids = np.zeros_like(centroids)

    # Iterate until the centroids not change much


    for _ in range(max_iter):
        # Assign each point to the nearest centroid
        distances = np.sqrt(((X - centroids[:, np.newaxis]) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=0)

        # Update centroids
        prev_centroids = centroids.copy()
        for i in range(k):
            centroids[i] = np.mean(X[labels == i], axis=0)
        print(f"i Centroids: {centroids}")


        # Check convergence
        if np.allclose(centroids, prev_centroids, atol=tol):
           

            break

    #print(f" f Centroids: {centroids}")


    return labels, centroids


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("K-Means Clustering")
        self.setGeometry(100, 100, 600, 400)

        layout = QVBoxLayout()

        self.percentage_label = QLabel("Enter the percentage of data to read (0-100):")
        layout.addWidget(self.percentage_label)

        self.percentage_input = QLineEdit()
        layout.addWidget(self.percentage_input)

        self.k_label = QLabel("Enter the number of clusters (k):")
        layout.addWidget(self.k_label)

        self.k_input = QLineEdit()
        layout.addWidget(self.k_input)

        self.run_button = QPushButton("Run Clustering")
        self.run_button.clicked.connect(self.run_clustering)
        layout.addWidget(self.run_button)

        self.result_text = QTextEdit()
        layout.addWidget(self.result_text)

        widget = QWidget()
        widget.setLayout(layout)
        self.setCentralWidget(widget)

    def run_clustering(self):
        percentage = float(self.percentage_input.text())
        k = int(self.k_input.text())

        # Read the dataset
        data = pd.read_csv("D:/senior2024/Data Mining/new assig/Assignment(2) - Data Mining/Assignment(2) - Data Mining-20200140/imdb_top_2000_movies.csv")

        # Preprocess the data if necessary

        # Select a percentage of the data based on user input
        sample_size = int(len(data) * (percentage / 100))
        data_sample = data.sample(n=sample_size, random_state=42)

        # Convert "IMDB Rating" column to numeric
        data_sample["IMDB Rating"] = pd.to_numeric(data_sample["IMDB Rating"], errors="coerce")

        # Drop all columns except "IMDB Rating" and "Movie Name"
        data_sample = data_sample[["Movie Name", "IMDB Rating"]]

        # Drop rows with missing "IMDB Rating" values
        data_sample = data_sample.dropna()

        # Detect outliers in the "IMDB Rating" column
        outliers = detect_outliers_iqr(data_sample)

        new_data = data_sample[
            ~data_sample["Movie Name"].isin(outliers)].copy()  # Use copy() to avoid SettingWithCopyWarning

        # Apply k-means clustering
        labels, centroids = k_means_clustering(new_data[["IMDB Rating"]].values, k)

        # Add cluster labels to the dataset
        new_data['Cluster'] = labels

        # Display cluster content and outlier records
        result_text = ""

        # Print outliers
        result_text += "Outliers:\n"
        result_text += str(outliers) + "\n\n"
        # Print count of outliers
        result_text += f"Number of outliers: {len(outliers)}\n\n"
        

        # Print clustering results
        for cluster in range(k):
            cluster_data = new_data[new_data['Cluster'] == cluster][["Movie Name", "IMDB Rating"]]
            result_text += f"Cluster {cluster + 1}:\n{cluster_data}\n\n"
            # Print count of movies in each cluster
            result_text += f"Number of movies in cluster {cluster + 1}: {len(cluster_data)}\n\n"
              

        self.result_text.setPlainText(result_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())