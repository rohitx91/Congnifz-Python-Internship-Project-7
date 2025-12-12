"""
data_viz_tool.py
Streamlit-based interactive data visualization tool.

Features:
- Upload CSV
- Quick dataset preview & summary
- Column selection
- Plot types: Line, Scatter, Bar, Histogram, Box, Heatmap, Pairplot
- Optional grouping & aggregation
- Interactive plots (Plotly) + static seaborn/matplotlib where useful
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

st.set_page_config(layout="wide", page_title="Data Visualization Tool")

# --- Sidebar ---
st.sidebar.title("Data Visualization Tool")
st.sidebar.markdown("Upload a CSV and choose plot options.")

uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv", "txt", "xlsx"])
sample_datasets = {
    "No sample": None,
    "Iris (builtin sample)": "iris"
}
sample_choice = st.sidebar.selectbox("Or choose a sample dataset (for quick demo)", list(sample_datasets.keys()))

# Helper to load sample iris if chosen
def load_sample(name):
    if name == "iris":
        from sklearn import datasets
        iris = datasets.load_iris(as_frame=True)
        df = iris.frame
        return df
    return None

# Load dataframe
df = None
if uploaded_file is not None:
    try:
        if uploaded_file.name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Could not read file: {e}")
elif sample_choice != "No sample":
    df = load_sample("iris")

if df is None:
    st.header("Welcome 👋")
    st.write("Upload a CSV from the sidebar or choose a sample dataset to start.")
    st.write("Small tips:")
    st.write("- Make sure your CSV has a header row.")
    st.write("- For best results, numeric columns should be properly typed.")
    st.stop()

# --- Main UI ---
st.header("Dataset preview")
st.write(f"**Rows:** {df.shape[0]}, **Columns:** {df.shape[1]}")
with st.expander("Show raw data (first 200 rows)"):
    st.dataframe(df.head(200))

st.markdown("---")
st.subheader("Quick summary")
col1, col2, col3 = st.columns([1,1,1])
with col1:
    st.write("**Data types**")
    st.write(pd.DataFrame(df.dtypes, columns=["dtype"]))
with col2:
    st.write("**Missing values**")
    st.write(df.isna().sum())
with col3:
    st.write("**Numeric summary**")
    st.write(df.describe().T)

st.markdown("---")
st.subheader("Plot settings")

# Choose numeric and categorical columns
all_columns = df.columns.tolist()
numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
categorical_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

plot_type = st.selectbox("Plot type", ["Line", "Scatter", "Bar", "Histogram", "Box", "Heatmap (corr)", "Pairplot"])

# Axis selectors (only show relevant ones)
x_col = None; y_col = None; color_col = None; agg_func = None; group_col = None
if plot_type in ["Line", "Scatter", "Bar"]:
    x_col = st.selectbox("X axis", [None] + all_columns, index=0)
    y_col = st.selectbox("Y axis", [None] + all_columns, index=0)
    color_col = st.selectbox("Color / Hue (optional)", [None] + categorical_cols, index=0)
    # Grouping & aggregation
    group_col = st.selectbox("Group by (optional)", [None] + categorical_cols, index=0)
    if group_col is not None:
        agg_func = st.selectbox("Aggregation function", ["mean", "sum", "median", "count", "min", "max"])
elif plot_type == "Histogram":
    x_col = st.selectbox("Column (numeric)", [None] + numeric_cols, index=0)
    bins = st.slider("Bins", 5, 200, 30)
elif plot_type == "Box":
    x_col = st.selectbox("X (categorical, optional)", [None] + categorical_cols, index=0)
    y_col = st.selectbox("Y (numeric)", [None] + numeric_cols, index=0)
    color_col = st.selectbox("Color (optional)", [None] + categorical_cols, index=0)
elif plot_type == "Heatmap (corr)":
    corr_method = st.selectbox("Correlation method", ["pearson", "spearman", "kendall"])
elif plot_type == "Pairplot":
    pair_cols = st.multiselect("Select columns for pairplot (numeric)", numeric_cols, default=numeric_cols[:4])

st.markdown("---")
st.subheader("Build plot")

# Prepare data (apply grouping/aggregation if chosen)
plot_df = df.copy()

if group_col and agg_func and x_col and y_col:
    try:
        # If x is numeric, we group by group_col and aggregate y
        grouped = plot_df.groupby(group_col).agg({y_col: agg_func}).reset_index()
        plot_df = grouped
        st.info(f"Applied grouping by `{group_col}` and `{agg_func}` on `{y_col}`.")
    except Exception as e:
        st.error(f"Grouping failed: {e}")

# Draw plots
def show_plotly(fig):
    st.plotly_chart(fig, use_container_width=True)

def show_seaborn_matplotlib(fig):
    st.pyplot(fig, bbox_inches="tight")

if st.button("Generate plot"):
    try:
        if plot_type == "Line":
            if x_col is None or y_col is None:
                st.error("Select both X and Y for Line plot.")
            else:
                fig = px.line(plot_df, x=x_col, y=y_col, color=color_col, title=f"Line: {y_col} vs {x_col}")
                show_plotly(fig)

        elif plot_type == "Scatter":
            if x_col is None or y_col is None:
                st.error("Select both X and Y for Scatter plot.")
            else:
                fig = px.scatter(plot_df, x=x_col, y=y_col, color=color_col, title=f"Scatter: {y_col} vs {x_col}", hover_data=all_columns)
                show_plotly(fig)

        elif plot_type == "Bar":
            if x_col is None or y_col is None:
                st.error("Select both X and Y for Bar plot.")
            else:
                # If x is categorical, automatically aggregate if necessary
                if plot_df[x_col].dtype == "object" or plot_df[x_col].dtype.name == "category":
                    agg = plot_df.groupby(x_col)[y_col].sum().reset_index()
                    fig = px.bar(agg, x=x_col, y=y_col, color=color_col, title=f"Bar (aggregated): {y_col} by {x_col}")
                else:
                    fig = px.bar(plot_df, x=x_col, y=y_col, color=color_col, title=f"Bar: {y_col} vs {x_col}")
                show_plotly(fig)

        elif plot_type == "Histogram":
            if x_col is None:
                st.error("Choose a numeric column for Histogram.")
            else:
                fig = px.histogram(plot_df, x=x_col, nbins=bins, title=f"Histogram: {x_col}")
                show_plotly(fig)

        elif plot_type == "Box":
            if y_col is None:
                st.error("Choose Y (numeric) for Box plot.")
            else:
                if x_col:
                    fig = px.box(plot_df, x=x_col, y=y_col, color=color_col, title=f"Box: {y_col} by {x_col}")
                else:
                    fig = px.box(plot_df, y=y_col, title=f"Box: {y_col}")
                show_plotly(fig)

        elif plot_type == "Heatmap (corr)":
            num_df = plot_df.select_dtypes(include=np.number)
            if num_df.shape[1] < 2:
                st.error("Need at least two numeric columns for correlation heatmap.")
            else:
                corr = num_df.corr(method=corr_method)
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(corr, annot=True, fmt=".2f", ax=ax, cmap="vlag")
                ax.set_title("Correlation matrix")
                show_seaborn_matplotlib(fig)

        elif plot_type == "Pairplot":
            if not pair_cols or len(pair_cols) < 2:
                st.error("Select at least two numeric columns for a pairplot.")
            else:
                fig = sns.pairplot(plot_df[pair_cols].dropna())
                st.pyplot(fig.fig)

        else:
            st.error("Unknown plot type selected.")

    except Exception as e:
        st.exception(f"Failed to create plot: {e}")

st.markdown("---")
st.subheader("Download & Export")

buffer = io.BytesIO()
st.download_button("Download current data (CSV)", data=plot_df.to_csv(index=False).encode("utf-8"), file_name="data_export.csv", mime="text/csv")

st.caption("Tip: You can preprocess the data (filter/sort) in your spreadsheet or use Python to clean before uploading for best visuals. ✨")

# Optional: small quick filters (bonus)
with st.expander("Quick filter (optional)"):
    col_to_filter = st.selectbox("Choose column to filter (optional)", [None] + all_columns, index=0)
    if col_to_filter:
        vals = plot_df[col_to_filter].unique().tolist()
        chosen = st.multiselect("Filter values", vals, default=vals[:5])
        if st.button("Apply filter"):
            filtered_df = plot_df[plot_df[col_to_filter].isin(chosen)]
            st.write("Filtered preview:")
            st.dataframe(filtered_df.head(100))
            # Allow download
            st.download_button("Download filtered data", data=filtered_df.to_csv(index=False).encode("utf-8"), file_name="filtered_data.csv", mime="text/csv")
