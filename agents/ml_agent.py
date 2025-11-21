from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
import pandas as pd

class MLAgent:
    def __init__(self):
        pass

    def choose_task(self, data, target_column):
        if data[target_column].dtype in ['int64', 'float64']:
            return "regression"
        else:
            return "classification"

    def preprocess(self, data):
        for col in data.columns:
            if data[col].dtype == 'object':
                data[col] = LabelEncoder().fit_transform(data[col])
        return data

    def train_model(self, data, target_column):
        data = self.preprocess(data)

        X = data.drop(target_column, axis=1)
        y = data[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

        task = self.choose_task(data, target_column)

        if task == "regression":
            model = LinearRegression()
        else:
            model = RandomForestClassifier()

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        if task == "regression":
            score = model.score(X_test, y_test)
        else:
            score = accuracy_score(y_test, predictions)

        return {
            "model": model,
            "task": task,
            "score": score
        }
