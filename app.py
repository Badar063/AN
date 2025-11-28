import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json
import os

# Set page configuration for a wide, attractive layout
st.set_page_config(
    page_title="ArchNet Deep Learning Benchmark",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Aesthetic Configuration (for plots) ---
MAIN_COLOR = '#1f77b4' # Plotly blue
SECONDARY_COLOR = '#ff7f0e' # Plotly orange

# --- Data Loading (using Streamlit caching for performance) ---

@st.cache_data
def load_data(results_path, history_path):
    """Loads and preprocesses the results and history data."""
    try:
        results_df = pd.read_csv(results_path)
    except FileNotFoundError:
        # Crucial error handling for deployment
        st.error(f"❌ Data file not found: {results_path}. Please ensure both CSV and JSON are uploaded alongside app.py.")
        return pd.DataFrame(), {}
    except Exception as e:
        st.error(f"Error loading {results_path}: {e}")
        return pd.DataFrame(), {}

    try:
        with open(history_path, 'r') as f:
            history_data = json.load(f)
    except FileNotFoundError:
        st.error(f"❌ History file not found: {history_path}. Please ensure both CSV and JSON are uploaded alongside app.py.")
        return results_df, {}
    except Exception as e:
        st.error(f"Error loading {history_path}: {e}")
        return results_df, {}

    return results_df, history_data

# Load data files (assuming they are in the same directory)
RESULTS_FILE = 'archnet_real_results.csv'
HISTORY_FILE = 'training_histories.json'
results_df, history_data = load_data(RESULTS_FILE, HISTORY_FILE)

# --- Sidebar Filters ---
st.sidebar.title("Dashboard Controls")
st.sidebar.markdown("Use the filters below to focus the analysis.")

if results_df.empty:
    st.title("Data Loading Error")
    st.warning("Please check the error message in the main content area above.")
    st.stop()
    
# Proceed only if data loaded successfully
datasets = results_df['dataset_name'].unique()
selected_dataset = st.sidebar.selectbox("🎯 Select Dataset", datasets)

models = results_df['model_name'].unique()
selected_model = st.sidebar.selectbox("🧠 Select Model", models)

# Filter DataFrame based on selection
filtered_df = results_df[results_df['dataset_name'] == selected_dataset]
focus_model_df = filtered_df[filtered_df['model_name'] == selected_model].iloc[0]

# Find the best models for KPIs
best_acc = filtered_df.loc[filtered_df['test_accuracy'].idxmax()]
best_f1 = filtered_df.loc[filtered_df['f1_score'].idxmax()]
fastest_inf = filtered_df.loc[filtered_df['inference_time'].idxmin()]


# =============================================================================
# --- Main Dashboard Content ---
# =============================================================================

st.title("🧬 ArchNet Benchmark: CNN Architectures for X-ray Analysis")
st.markdown(f"**Analyzing performance of various CNN models on the `{selected_dataset.upper()}` dataset.**")

st.markdown("---")

## 1. Key Performance Indicators (KPIs)
st.header("🏆 Dataset Overview (KPIs)")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Best Test Accuracy",
        value=f"{best_acc['test_accuracy']:.4f}",
        delta=f"Model: {best_acc['model_name']}"
    )
with col2:
    st.metric(
        label="Best F1-Score (Weighted)",
        value=f"{best_f1['f1_score']:.4f}",
        delta=f"Model: {best_f1['model_name']}"
    )
with col3:
    st.metric(
        label="Fastest Inference Time (per image)",
        value=f"{fastest_inf['inference_time']:.4f} s",
        delta=f"Model: {fastest_inf['model_name']}"
    )
with col4:
    # Smallest model in MB
    smallest_size = filtered_df.loc[filtered_df['model_size_mb'].idxmin()]
    st.metric(
        label="Smallest Model Size",
        value=f"{smallest_size['model_size_mb']:.2f} MB",
        delta=f"Model: {smallest_size['model_name']}"
    )

st.markdown("---")

## 2. Model Comparison Plots (Plotly Bar Charts)
st.header("📊 Performance Comparison on Selected Dataset")

comparison_col1, comparison_col2 = st.columns(2)

with comparison_col1:
    # Accuracy/F1 Plot
    metrics_df = filtered_df[['model_name', 'test_accuracy', 'f1_score']].melt(
        id_vars='model_name',
        var_name='Metric',
        value_name='Score'
    )
    fig_acc_f1 = px.bar(
        metrics_df,
        x='model_name',
        y='Score',
        color='Metric',
        barmode='group',
        title='Accuracy vs. F1-Score by Architecture',
        color_discrete_map={'test_accuracy': MAIN_COLOR, 'f1_score': SECONDARY_COLOR}
    )
    fig_acc_f1.update_layout(xaxis_title="Model Architecture", yaxis_title="Score (0.0 - 1.0)", legend_title="Metric")
    st.plotly_chart(fig_acc_f1, use_container_width=True)

with comparison_col2:
    # Speed/Size Plot
    
    # Dual-axis chart for Speed vs Size (advanced visualization)
    fig_speed_size = px.bar(
        filtered_df,
        x='model_name',
        y='inference_time',
        title='Inference Time & Model Size',
        color_discrete_sequence=[MAIN_COLOR],
        opacity=0.6,
        labels={'inference_time': 'Inference Time (s)'}
    )
    # Add secondary line plot for model size
    fig_speed_size.add_scatter(
        x=filtered_df['model_name'],
        y=filtered_df['model_size_mb'],
        mode='lines+markers',
        name='Model Size (MB)',
        yaxis='y2',
        marker=dict(color=SECONDARY_COLOR, size=10)
    )
    fig_speed_size.update_layout(
        xaxis_title="Model Architecture",
        yaxis=dict(title="Inference Time (s)", showgrid=False),
        yaxis2=dict(title="Model Size (MB)", overlaying='y', side='right', showgrid=True),
        legend_title="Metric"
    )
    st.plotly_chart(fig_speed_size, use_container_width=True)

st.markdown("---")

## 3. Training History Analysis
st.header(f"📈 Training History: {selected_model} on {selected_dataset.upper()}")

if selected_dataset in history_data and selected_model in history_data[selected_dataset]:
    hist = history_data[selected_dataset][selected_model]
    epochs = range(1, len(hist['loss']) + 1)
    
    # Create a DataFrame for plotting history
    history_df = pd.DataFrame({
        'Epoch': epochs,
        'Training Loss': hist.get('loss', [0]),
        'Validation Loss': hist.get('val_loss', [0]),
        'Training Accuracy': hist.get('accuracy', [0]),
        'Validation Accuracy': hist.get('val_accuracy', [0])
    })
    
    # Melt the DataFrame for easy plotting with Plotly
    loss_df = history_df[['Epoch', 'Training Loss', 'Validation Loss']].melt(
        id_vars='Epoch', var_name='Metric', value_name='Loss'
    )
    acc_df = history_df[['Epoch', 'Training Accuracy', 'Validation Accuracy']].melt(
        id_vars='Epoch', var_name='Metric', value_name='Accuracy'
    )

    history_col1, history_col2 = st.columns(2)
    
    with history_col1:
        fig_loss = px.line(
            loss_df,
            x='Epoch',
            y='Loss',
            color='Metric',
            title=f"{selected_model} Loss over Epochs",
            markers=True
        )
        fig_loss.update_layout(yaxis_title="Loss", legend_title="Loss Metric")
        st.plotly_chart(fig_loss, use_container_width=True)

    with history_col2:
        fig_acc = px.line(
            acc_df,
            x='Epoch',
            y='Accuracy',
            color='Metric',
            title=f"{selected_model} Accuracy over Epochs",
            markers=True
        )
        fig_acc.update_layout(yaxis_title="Accuracy", legend_title="Accuracy Metric")
        st.plotly_chart(fig_acc, use_container_width=True)

else:
    st.warning("⚠️ Training history data not available for this model/dataset combination.")

st.markdown("---")

## 4. Model Efficiency Analysis (Pareto Front)
st.header("📦 Efficiency Trade-off (Accuracy vs. Size)")
st.markdown("Analyze the **Pareto Front** to find models that offer the best accuracy-to-size trade-off across **all** datasets.")

# Scatter plot for all data
fig_scatter = px.scatter(
    results_df,
    x='model_size_mb',
    y='test_accuracy',
    color='model_name',
    symbol='dataset_name',
    size='f1_score', # Use F1-score to size the markers (advanced feature)
    hover_data=['model_name', 'dataset_name', 'f1_score', 'inference_time'],
    title='Model Accuracy vs. Size: Finding the Pareto Front',
    template='plotly_white'
)

fig_scatter.update_traces(
    marker=dict(line=dict(width=1, color='DarkSlateGrey')),
    selector=dict(mode='markers')
)

fig_scatter.update_layout(
    xaxis_title='Model Size (MB)',
    yaxis_title='Test Accuracy (Score)',
    legend_title='Model/Dataset'
)

st.plotly_chart(fig_scatter, use_container_width=True)

st.markdown("""
<style>
    /* Styling to make the dashboard more attractive and eye-catching */
    .st-metric div {
        font-size: 1.2rem;
    }
    .st-metric > div > div:nth-child(2) {
        font-size: 2.5rem;
        color: #e30b5d; /* Custom bold color for main value */
    }
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #1f77b4; /* Deep blue for title */
    }
</style>
""", unsafe_allow_html=True)
