from agents.data_agent import DataAgent
from agents.ml_agent import MLAgent
from agents.visual_agent import VisualAgent
import pandas as pd

def main():

    print("\n=== Marketing Portfolio Multi-Agent System Started ===\n")

    # --- AGENT 1: DATA AGENT ---
    data_agent = DataAgent()
    data = data_agent.load_data("data/sales.csv")   # <-- your sales file goes here
    
    if data is None:
        print("Dataset not found. Please add sales.csv to /data folder.")
        return

    cleaned_data = data_agent.clean_data(data)
    analysis = data_agent.basic_analysis(cleaned_data)

    print("\n🔹 Basic Data Analysis:")
    print(analysis)

    # --- AGENT 2: ML AGENT ---
    ml_agent = MLAgent()
    
    target_column = cleaned_data.columns[-1]  # last column assumed as prediction column
    print(f"\n📌 Target Column Selected Automatically: {target_column}")

    ml_results = ml_agent.train_model(cleaned_data, target_column)

    print("\n🔹 ML Results:")
    print(f"Task: {ml_results['task']}")
    print(f"Score: {ml_results['score']}")

    # --- AGENT 3: VISUAL AGENT ---
    visual_agent = VisualAgent()
    
    # Simple visualization of the first numeric column
    numeric_col = cleaned_data.select_dtypes(include=['int64','float64']).columns[0]
    
    print(f"\n📊 Generating Trend Chart for: {numeric_col}")
    visual_agent.plot_column(cleaned_data, numeric_col)

    print("\n📌 All graphs saved successfully!")
    print("\n=== Multi-Agent System Finished ===")

if __name__ == "__main__":
    main()
