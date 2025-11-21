import os
from agents.data_agent import DataAgent
from agents.ml_agent import MLAgent
from agents.visual_agent import VisualAgent

DATA_PATH = "data/cleaned_sales_data.csv"

def main():

    print("\n===== STEP 1: DATA AGENT =====")
    data_agent = DataAgent()
    data = data_agent.load_data(DATA_PATH)

    if data is None:
        print(" Failed to load data.")
        return

    cleaned_data = data_agent.clean_data(data)
    print("Basic Analysis:", data_agent.basic_analysis(cleaned_data))

    print("\n===== STEP 2: ML AGENT =====")
    ml_agent = MLAgent()

    # ⛔ CHANGE THIS TO ANY TARGET YOU WANT TO PREDICT
    target_column = cleaned_data.columns[-1]

    model_info = ml_agent.train_model(cleaned_data, target_column)

    print(f"ML Task: {model_info['task']}")
    print(f"Model Score: {model_info['score']}")

    print("\n===== STEP 3: VISUALIZATION AGENT =====")
    visual_agent = VisualAgent()

    # Plot all numeric columns trend
    for col in cleaned_data.select_dtypes(include=['int64', 'float64']).columns:
        visual_agent.plot_column(cleaned_data, col)

    print("📊 Trend plots saved.")

    print("\n🎉 Pipeline Completed Successfully!")


if __name__ == "__main__":
    main()

