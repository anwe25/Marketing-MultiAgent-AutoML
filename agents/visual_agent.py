import matplotlib.pyplot as plt
import pandas as pd

class VisualAgent:
    def __init__(self):
        pass

    def plot_column(self, data, column):
        plt.figure(figsize=(8,5))
        plt.plot(data[column])
        plt.title(f"Trend of {column}")
        plt.xlabel("Index")
        plt.ylabel(column)
        plt.savefig(f"{column}_trend.png")
        plt.close()

    def bar_chart(self, data, column):
        plt.figure(figsize=(8,5))
        data[column].value_counts().plot(kind='bar')
        plt.title(f"Distribution of {column}")
        plt.xlabel(column)
        plt.ylabel("Count")
        plt.savefig(f"{column}_bar.png")
        plt.close()

    def visualize_predictions(self, y_test, predictions):
        plt.figure(figsize=(8,5))
        plt.plot(list(y_test), label="Actual")
        plt.plot(list(predictions), label="Predicted")
        plt.legend()
        plt.title("Actual vs Predicted")
        plt.savefig("prediction_comparison.png")
        plt.close()
