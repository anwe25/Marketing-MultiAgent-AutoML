import os
import sys
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import streamlit.components.v1 as components

# Prevent running the Streamlit app with "python app/dashboard.py" (common mistake)
try:
    # streamlit 1.18+ provides this helper
    from streamlit.runtime.scriptrunner import get_script_run_ctx
    if get_script_run_ctx() is None:
        sys.exit("ERROR: Run this app with: streamlit run app\\dashboard.py (do NOT run with 'python').")
except Exception:
    # If import fails, still check argv for 'streamlit' indicator and bail otherwise
    if "streamlit" not in " ".join(sys.argv).lower():
        sys.exit("ERROR: Run this app with: streamlit run app\\dashboard.py (do NOT run with 'python').")

# Ensure project root is importable
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from agents.data_agent import DataAgent
from agents.visual_agent import VisualAgent
from agents.ml_agent import MLAgent

st.set_page_config(page_title="Marketing Analytics Dashboard", layout="wide")
st.title("📊 Marketing Multi-Agent Dashboard")

# Data path (default)
DEFAULT_DATA = os.path.join(PROJECT_ROOT, "data", "cleaned_sales_data.csv")

# Sidebar: data source
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload CSV (optional)", type=["csv"])
use_default = False
if uploaded is None and os.path.exists(DEFAULT_DATA):
    use_default = st.sidebar.checkbox(f"Use default data: {os.path.basename(DEFAULT_DATA)}", value=True)

data_agent = DataAgent()
if uploaded:
    try:
        df = pd.read_csv(uploaded)
        st.sidebar.success("Loaded uploaded CSV")
    except Exception as e:
        st.sidebar.error(f"Failed to load uploaded CSV: {e}")
        st.stop()
elif use_default:
    df = data_agent.load_data(DEFAULT_DATA)
    if df is None:
        st.sidebar.error("Failed to load default data.")
        st.stop()
else:
    st.info("Upload a CSV or enable the default dataset in the sidebar.")
    st.stop()

# Clean data (button) and show basic analysis
if st.sidebar.button("Clean data"):
    df = data_agent.clean_data(df)
    st.sidebar.success("Data cleaned (in-memory)")

with st.expander("Preview data"):
    st.dataframe(df.head(200))

with st.expander("Basic analysis"):
    try:
        summary = data_agent.basic_analysis(df)
        st.json(summary)
    except Exception as e:
        st.error(f"Basic analysis failed: {e}")

# Filters
st.sidebar.header("Filters")
cols = list(df.columns)
date_col = None
# try to guess a date-like column
for c in cols:
    if "date" in c.lower():
        date_col = c
        break

if date_col:
    try:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        min_date, max_date = df[date_col].min(), df[date_col].max()
        if pd.notna(min_date) and pd.notna(max_date):
            default_range = (min_date.date(), max_date.date())
            date_range = st.sidebar.date_input("Date range", value=default_range)
            if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
                start_dt = pd.to_datetime(date_range[0])
                end_dt = pd.to_datetime(date_range[1])
                df = df[(df[date_col] >= start_dt) & (df[date_col] <= end_dt)]
    except Exception:
        pass

# Example categorical filter (if Category exists)
if "Category" in df.columns:
    cat_opts = ["All"] + sorted(df["Category"].dropna().unique().tolist())
    cat_sel = st.sidebar.selectbox("Category", options=cat_opts, index=0)
    if cat_sel != "All":
        df = df[df["Category"] == cat_sel]

# Update cols after filtering
cols = list(df.columns)

# Plots area
st.header("Visualizations")
vis = VisualAgent()

# Select column to plot
plot_col = st.selectbox("Choose column to plot", options=cols, index=0)
plot_type = st.selectbox("Plot type", options=["auto", "line", "bar"], index=0)
use_pyecharts = st.sidebar.checkbox("Enable pyecharts (interactive)", value=False)

if st.button("Generate plot"):
    try:
        series = df[plot_col]
        fig, ax = plt.subplots(figsize=(8, 4))
        if plot_type == "auto":
            if hasattr(series, "dtype") and series.dtype.kind in "biufc":
                ax.plot(series.values)
            else:
                vc = series.value_counts().sort_index()
                ax.bar(range(len(vc)), vc.values)
                ax.set_xticks(range(len(vc)))
                ax.set_xticklabels([str(x) for x in vc.index], rotation=45, ha="right")
        elif plot_type == "line":
            ax.plot(series.values)
        else:
            vc = series.value_counts().sort_index()
            ax.bar(range(len(vc)), vc.values)
            ax.set_xticks(range(len(vc)))
            ax.set_xticklabels([str(x) for x in vc.index], rotation=45, ha="right")

        ax.set_title(f"{plot_col} trend")
        ax.set_ylabel(str(plot_col))
        st.pyplot(fig)

        out = vis.plot_column(df, plot_col)
        if isinstance(out, dict) and out.get("png"):
            st.success(f"Saved PNG: {out.get('png')}")
        if use_pyecharts and isinstance(out, dict) and out.get("html"):
            # render the saved html interactive chart
            with open(out.get("html"), "r", encoding="utf-8") as f:
                html = f.read()
            components.html(html, height=500)
        elif use_pyecharts and (not out or not out.get("html")):
            st.warning("pyecharts not available or failed to render. Install pyecharts: pip install pyecharts")
    except Exception as e:
        st.error(f"Failed to generate plot: {e}")

# Modeling area
st.header("Modeling")
ml = MLAgent()

# choose target candidates (non-constant)
candidates = [c for c in df.columns if df[c].nunique() > 1]
if not candidates:
    st.warning("No suitable target found (all columns constant).")
else:
    target = st.selectbox("Select target column", options=candidates)
    if st.button("Train model"):
        try:
            res = ml.train_model(df, target)
            st.success("Training completed")
            st.write("Task:", res.get("task"))
            st.write("Score:", float(res.get("score")))
            # show few predictions
            y_test = res.get("y_test")
            preds = res.get("predictions")
            compare = pd.DataFrame({"y_test": list(y_test)[:50], "pred": list(preds)[:50]})
            st.dataframe(compare)
        except Exception as e:
            st.error(f"Training failed: {e}")

st.caption("Run: streamlit run app\\dashboard.py (from project root).")