import pandas as pd

class DataAgent:
    def __init__(self):
        pass

    def load_data(self, file_path):
        try:
            data = pd.read_csv(file_path)
            print("🔹 Data Loaded Successfully")
            return data
        except Exception as e:
            print("Error loading data:", e)
            return None

    def clean_data(self, data):
        print("Cleaning Data...")

        # Remove duplicates
        data = data.drop_duplicates()

        # Fill missing numeric values with mean
        numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
        data[numeric_cols] = data[numeric_cols].fillna(data[numeric_cols].mean())

        # Fill missing categorical values with mode
        categorical_cols = data.select_dtypes(include=['object']).columns
        data[categorical_cols] = data[categorical_cols].fillna(data[categorical_cols].mode().iloc[0])

        print("Data Cleaning Completed")
        return data

    def basic_analysis(self, data):
        print("Running Basic Analysis...")
        summary = {
            "Total Rows": len(data),
            "Total Columns": len(data.columns),
            "Column Names": list(data.columns),
            "Describe": data.describe().to_dict()
        }
        return summary
