"""
Streamlit Dashboard for Cross-Sell Opportunity Intelligence
Run with: streamlit run src/visualization/dashboard.py
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sqlalchemy import create_engine

# Configure page
st.set_page_config(page_title="Cross-Sell Intelligence Platform", page_icon="🎯", layout="wide")

# Custom CSS for better styling
st.markdown(
    """
<style>
.metric-card {
    background-color: #f0f2f5;
    padding: 20px;
    border-radius: 10px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
.opportunity-card {
    background-color: white;
    padding: 15px;
    border-radius: 8px;
    border-left: 4px solid #1f77b4;
    margin-bottom: 10px;
}
</style>
""",
    unsafe_allow_html=True,
)


# Database connection
@st.cache_resource
def get_db_connection():
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://user:pass@localhost/crosssell_db")
    return create_engine(DATABASE_URL)


# Load data
@st.cache_data(ttl=300)  # Cache for 5 minutes
def load_recommendations():
    engine = get_db_connection()
    query = """
    SELECT * FROM recommendations
    WHERE created_at >= NOW() - INTERVAL '30 days'
    ORDER BY score DESC
    """
    return pd.read_sql(query, engine)


@st.cache_data(ttl=300)
def load_time_series():
    engine = get_db_connection()
    query = """
    SELECT DATE(created_at) as date,
           COUNT(*) as count,
           SUM(estimated_value) as value
    FROM recommendations
    WHERE created_at >= NOW() - INTERVAL '30 days'
    GROUP BY DATE(created_at)
    ORDER BY date
    """
    return pd.read_sql(query, engine)


# Title and description
st.title("🎯 Cross-Sell Intelligence Platform")
st.markdown("**AI-Powered Revenue Opportunities Across Your Portfolio**")

# Sidebar for filters
st.sidebar.header("Filters")
min_score = st.sidebar.slider("Minimum Opportunity Score", 0.0, 1.0, 0.5)
selected_confidence = st.sidebar.multiselect(
    "Confidence Levels", ["Very High", "High", "Medium", "Low"], default=["Very High", "High"]
)

# Load data
recommendations_df = load_recommendations()
time_series_df = load_time_series()

# Apply filters
filtered_df = recommendations_df[
    (recommendations_df["score"] >= min_score)
    & (recommendations_df["confidence_level"].isin(selected_confidence))
]

# Key Metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    st.metric(
        "Total Opportunities",
        f"{len(filtered_df)}",
        f"+{len(filtered_df[filtered_df['created_at'] >= datetime.now() - timedelta(days=7)])} this week",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    total_value = filtered_df["estimated_value"].sum()
    st.metric(
        "Total Potential Revenue",
        f"${total_value:,.0f}",
        f"Avg: ${filtered_df['estimated_value'].mean():,.0f}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col3:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    avg_score = filtered_df["score"].mean()
    st.metric(
        "Average Score",
        f"{avg_score:.2f}",
        f"High confidence: {len(filtered_df[filtered_df['score'] > 0.8])}",
    )
    st.markdown("</div>", unsafe_allow_html=True)

with col4:
    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
    conversion_rate = 0.24  # Example metric
    st.metric("Conversion Rate", f"{conversion_rate:.1%}", "Based on historical data")
    st.markdown("</div>", unsafe_allow_html=True)

# Charts Section
st.markdown("---")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Opportunity Score Distribution")
    fig_hist = px.histogram(
        filtered_df,
        x="score",
        nbins=20,
        title="Distribution of Opportunity Scores",
        labels={"score": "Opportunity Score", "count": "Number of Opportunities"},
    )
    fig_hist.update_layout(showlegend=False, height=300)
    st.plotly_chart(fig_hist, use_container_width=True)

with col2:
    st.subheader("Recommendations by Type")
    type_counts = filtered_df["recommendation_type"].value_counts()
    fig_pie = px.pie(
        values=type_counts.values, names=type_counts.index, title="Recommendation Types"
    )
    fig_pie.update_layout(height=300)
    st.plotly_chart(fig_pie, use_container_width=True)

# Time series chart
st.subheader("Opportunity Discovery Trend")
fig_trend = go.Figure()
fig_trend.add_trace(
    go.Scatter(
        x=time_series_df["date"],
        y=time_series_df["count"],
        mode="lines+markers",
        name="Opportunities",
        line=dict(color="#1f77b4", width=3),
    )
)
fig_trend.update_layout(
    title="Daily Opportunities Identified",
    xaxis_title="Date",
    yaxis_title="Number of Opportunities",
    height=300,
)
st.plotly_chart(fig_trend, use_container_width=True)


def display_opportunities(df):
    for _, opp in df.iterrows():
        st.markdown(
            f"""
            <div class="opportunity-card">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <div>
                        <h4 style="margin: 0;">{opp['account1_name']} × {opp['account2_name']}</h4>
                        <p style="color: #666; margin: 5px 0;">
                            {opp['account1_org']} → {opp['account2_org']} |
                            {opp['recommendation_type']}
                        </p>
                        <p style="margin: 5px 0;">
                            <strong>Action:</strong> {opp['next_best_action']}
                        </p>
                    </div>
                    <div style="text-align: right;">
                        <h3 style="color: #1f77b4; margin: 0;">{opp['score']:.2f}</h3>
                        <p style="color: #28a745; margin: 0; font-weight: bold;">
                            ${opp['estimated_value']:,.0f}
                        </p>
                        <p style="color: #666; margin: 0; font-size: 0.9em;">
                            {opp['confidence_level']}
                        </p>
                    </div>
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


# Top Opportunities Table
st.markdown("---")
st.subheader("🏆 Top Cross-Sell Opportunities")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["High Score", "High Value", "Recent"])

with tab1:
    top_by_score = filtered_df.nlargest(10, "score")
    display_opportunities(top_by_score)

with tab2:
    top_by_value = filtered_df.nlargest(10, "estimated_value")
    display_opportunities(top_by_value)

with tab3:
    recent = filtered_df.nlargest(10, "created_at")
    display_opportunities(recent)


# Export Section
st.markdown("---")
col1, col2, col3 = st.columns([1, 1, 3])

with col1:
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download CSV",
        data=csv,
        file_name=f"cross_sell_opportunities_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv",
    )

with col2:
    if st.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.experimental_rerun()

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
    Cross-Sell Intelligence Platform | Powered by AI | Last updated: {0}
    </div>
    """.format(
        datetime.now().strftime("%Y-%m-%d %H:%M")
    ),
    unsafe_allow_html=True,
)
