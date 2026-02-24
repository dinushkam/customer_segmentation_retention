import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.io as pio
import yaml
import base64

import plotly.graph_objects as go

# ----------------------
# Plotly theme: dark-glass (matches custom CSS)
# ----------------------
_CUSTOM_PLOTLY_TEMPLATE = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(
            color="#E8EEFF",
            family="Inter, system-ui, -apple-system, Segoe UI, Roboto, sans-serif",
        ),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)"),
        yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.12)"),
        colorway=[
            "#00E5FF", "#FF3D81", "#7C4DFF", "#00E676",
            "#FFD54F", "#FF6D00", "#64FFDA", "#B388FF",
        ],
    )
)
pio.templates["customer360_dark"] = _CUSTOM_PLOTLY_TEMPLATE
pio.templates.default = "customer360_dark"

from utils import (
    load_customer_data,
    metric_card,
    process_uploaded_data,
)

st.set_page_config(page_title="Customer 360 Analytics", layout="wide")

# Load config
with open(Path(__file__).parent / 'config.yaml') as f:
    config = yaml.safe_load(f)

# Load custom CSS
with open(Path(__file__).parent / 'assets' / 'style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)

# ======================
# MAIN HEADER
# ======================
st.markdown(f"""
<div class="header" style="margin-bottom: 1rem;">
    <h1 style="font-size: 3rem; font-weight: 700; color: {config['dashboard']['theme']['primary_color']};">📊 {config['dashboard']['title']}</h1>
    <h3 style="color: rgba(232,238,255,0.75); font-weight: 400;">Empower your marketing decisions with data-driven customer insights.</h3>
    <hr style="border: 2px solid {config['dashboard']['theme']['primary_color']}; opacity: 0.3;">
</div>
""", unsafe_allow_html=True)

# ======================
# SIDEBAR
# ======================
with st.sidebar:
    # Logo
    assets_dir = Path(__file__).parent / 'assets'
    logo_path = assets_dir / 'logo.jpg'


    def load_logo(path):
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode()


    if logo_path.exists():
        logo_base64 = load_logo(logo_path)
        st.markdown(
            f"""
            <div class="sidebar-logo">
                <img src="data:image/png;base64,{logo_base64}" width="180">
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        st.warning("Logo not found. Using placeholder.")
        st.image("https://via.placeholder.com/200x80?text=Your+Logo", width=200)

    st.markdown("---")
    st.markdown("## 📁 Data Source")

    with st.expander("📋 CSV Format Requirements", expanded=False):
        st.markdown("""
        **Your CSV file must contain the following columns:**

        | Column         | Required | Description                                                                 |
        |----------------|----------|-----------------------------------------------------------------------------|
        | `CustomerID`   | ✅ Yes   | Unique identifier for each customer (text or number).                       |
        | `InvoiceDate`  | ✅ Yes   | Date of transaction (e.g., `2023-01-15` or `15/01/2023 10:30`).            |
        | `Quantity`     | ⚠️ One of | Number of items purchased (positive integer).                               |
        | `UnitPrice`    | ⚠️ these | Price per unit (positive number).                                           |
        | `TotalPrice`   | ⚠️ one   | Pre-computed transaction total (`Quantity × UnitPrice`).                    |

        > **Either provide `Quantity` + `UnitPrice` OR `TotalPrice`.** If both are given, `TotalPrice` will be recomputed from `Quantity * UnitPrice`.

        **Example rows:**
        ```
        CustomerID,InvoiceDate,Quantity,UnitPrice
        123,2023-01-15,2,25.50
        123,2023-02-10,1,15.00
        456,2023-01-20,3,10.00
        ```

        **Or with `TotalPrice`:**
        ```
        CustomerID,InvoiceDate,TotalPrice
        123,2023-01-15,51.00
        123,2023-02-10,15.00
        456,2023-01-20,30.00
        ```

        **Notes:**
        - Rows with missing `CustomerID` are automatically removed.
        - Transactions with zero or negative `Quantity` / `UnitPrice` / `TotalPrice` are filtered out.
        - Extra columns (e.g., product description) are ignored – only those listed above are used.
        """)

        # Template download
        template_df = pd.DataFrame({
            'CustomerID': [123, 123],
            'InvoiceDate': ['2023-01-15', '2023-02-10'],
            'Quantity': [2, 1],
            'UnitPrice': [25.50, 15.00]
        })
        csv_template = template_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📎 Download CSV template",
            data=csv_template,
            file_name="upload_template.csv",
            mime='text/csv',
        )

    # --- FILE UPLOAD ---
    uploaded_file = st.file_uploader(
        "Upload your own transaction data (CSV)",
        type=['csv'],
        help="Upload a CSV with columns: CustomerID, InvoiceDate, Quantity, UnitPrice (or TotalPrice)."
    )

    if uploaded_file is not None:
        # Process the uploaded file
        with st.spinner("Processing your data..."):
            raw_df = pd.read_csv(uploaded_file)
            # Attempt to parse dates (adjust column name if needed)
            date_col = 'InvoiceDate' if 'InvoiceDate' in raw_df.columns else 'Date'
            if date_col in raw_df.columns:
                raw_df[date_col] = pd.to_datetime(raw_df[date_col], errors='coerce')
            else:
                st.error("Uploaded file must contain a date column (InvoiceDate or Date).")
                st.stop()

            # Process to customer-level features
            customer_df = process_uploaded_data(raw_df)
            if customer_df is not None:
                st.session_state['customer_data'] = customer_df
                st.session_state['data_source'] = "uploaded"
                st.success(f"✅ Data processed! {len(customer_df)} customers identified.")
            else:
                st.error("Data processing failed. Check file format.")
                st.stop()
    else:
        # No upload: use default dataset
        if 'customer_data' not in st.session_state or st.session_state.get('data_source') != "default":
            default_df = load_customer_data()
            if not default_df.empty:
                st.session_state['customer_data'] = default_df
                st.session_state['data_source'] = "default"

    st.markdown("---")
    st.markdown("## 🎛️ Dashboard Controls")
    st.markdown("Use the filters below to customize the view.")

    # Retrieve current data
    if 'customer_data' not in st.session_state:
        st.warning("No data available. Please upload a file or check default data.")
        st.stop()

    df = st.session_state['customer_data']

    with st.form(key="filter_form"):
        st.markdown("#### 🔍 Filters")
        segments = st.multiselect(
            "Customer Segment",
            options=df['Segment'].unique(),
            default=df['Segment'].unique()
        )
        col1, col2 = st.columns(2)
        with col1:
            min_ltv = st.number_input("Min LTV ($)", value=float(df['HistoricalLTV'].min()))
        with col2:
            max_ltv = st.number_input("Max LTV ($)", value=float(df['HistoricalLTV'].max()))
        min_churn, max_churn = st.slider(
            "Churn Probability",
            min_value=0.0,
            max_value=1.0,
            value=(0.0, 1.0)
        )
        st.markdown("---")
        apply_filters = st.form_submit_button("Apply Filters", type="primary")

    st.markdown("---")
    if st.session_state.get('data_source') == "uploaded":
        st.info("📢 Using your uploaded data. Some predictions are rule-based (no pre-trained models).")

# Apply filters
filtered_df = df[
    (df['Segment'].isin(segments)) &
    (df['HistoricalLTV'].between(min_ltv, max_ltv)) &
    (df['ChurnProb'].between(min_churn, max_churn))
]

if filtered_df.empty:
    st.warning("⚠️ No customers match the selected filters. Adjust filters.")
    st.stop()

# Key Metrics
st.subheader("📈 Key Performance Indicators")
cols = st.columns(4)

with cols[0]:
    st.markdown(metric_card(
        "Total Customers",
        f"{len(filtered_df):,}",
        help_text="Active customers in selection"
    ), unsafe_allow_html=True)

with cols[1]:
    avg_churn = filtered_df['ChurnProb'].mean()
    overall_churn = df['ChurnProb'].mean()
    delta_churn = f"{avg_churn - overall_churn:+.2%}"
    st.markdown(metric_card(
        "Avg Churn Probability",
        f"{avg_churn:.2%}",
        delta=delta_churn,
        help_text="vs overall average"
    ), unsafe_allow_html=True)

with cols[2]:
    avg_ltv = filtered_df['HistoricalLTV'].mean()
    overall_ltv = df['HistoricalLTV'].mean()
    delta_ltv = f"${avg_ltv - overall_ltv:+,.0f}"
    st.markdown(metric_card(
        "Avg Historical LTV",
        f"${avg_ltv:,.0f}",
        delta=delta_ltv,
        help_text="vs overall average"
    ), unsafe_allow_html=True)

with cols[3]:
    avg_pred = filtered_df['PredictedLTV_Next6Months'].mean()
    overall_pred = df['PredictedLTV_Next6Months'].mean()
    delta_pred = f"${avg_pred - overall_pred:+,.0f}"
    st.markdown(metric_card(
        "Avg Predicted LTV (6m)",
        f"${avg_pred:,.0f}",
        delta=delta_pred,
        help_text="vs overall average"
    ), unsafe_allow_html=True)

# Tabs
tab1, tab2, tab3 = st.tabs(["📊 Segment Overview", "📈 Risk & Value", "📋 Raw Data"])

with tab1:
    col_left, col_right = st.columns([1, 2])
    with col_left:
        fig_seg = px.pie(
            filtered_df,
            names='Segment',
            hole=0.4,
            color_discrete_sequence=px.colors.qualitative.Pastel,
            title="Segment Distribution"
        )
        fig_seg.update_layout(showlegend=False)
        st.plotly_chart(fig_seg, use_container_width=True)
    with col_right:
        st.subheader("Segment Profiles")
        profiles = filtered_df.groupby('Segment')[['Recency', 'Frequency', 'Monetary']].mean().round(1)
        st.dataframe(profiles.style.background_gradient(cmap='Purples'), use_container_width=True)

with tab2:
    fig_scatter = px.scatter(
        filtered_df,
        x='ChurnProb',
        y='HistoricalLTV',
        color='Segment',
        size='PredictedLTV_Next6Months',
        hover_data=['CustomerID'],
        title='Customers by Churn Risk and Lifetime Value',
        color_discrete_sequence=px.colors.qualitative.Pastel,
        labels={'ChurnProb': 'Churn Probability', 'HistoricalLTV': 'Historical LTV ($)'}
    )
    fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)')
    st.plotly_chart(fig_scatter, use_container_width=True)

with tab3:
    st.dataframe(filtered_df, use_container_width=True)
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Download data as CSV",
        data=csv,
        file_name='filtered_customers.csv',
        mime='text/csv',
    )

# Footer
st.markdown("""
<div class="footer">
    <hr>
    <p>© 2026 Customer 360 Analytics. All rights reserved.</p>
</div>
""", unsafe_allow_html=True)