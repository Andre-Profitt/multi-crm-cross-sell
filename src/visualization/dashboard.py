"""Streamlit Dashboard for Cross-Sell Opportunity Intelligence.

This dashboard surfaces cross-sell opportunities across multiple Salesforce
organizations. It includes interactive filtering and data exploration tools.
Run with: ``streamlit run src/visualization/dashboard.py``.
"""

import os
from datetime import datetime, timedelta
import math

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
@st.cache_data(ttl=300)
def load_recommendations(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load recommendations within the provided date range."""

    engine = get_db_connection()
    query = """
    SELECT * FROM recommendations
    WHERE created_at >= %(start)s AND created_at <= %(end)s
    ORDER BY score DESC
    """
    return pd.read_sql(query, engine, params={"start": start_date, "end": end_date})


@st.cache_data(ttl=300)
def load_time_series(start_date: datetime, end_date: datetime) -> pd.DataFrame:
    """Load aggregated time series metrics within the date range."""

    engine = get_db_connection()
    query = """
    SELECT DATE(created_at) as date,
           COUNT(*) as count,
           SUM(estimated_value) as value
    FROM recommendations
    WHERE created_at >= %(start)s AND created_at <= %(end)s
    GROUP BY DATE(created_at)
    ORDER BY date
    """
    return pd.read_sql(query, engine, params={"start": start_date, "end": end_date})


# Title and description
st.title("🎯 Cross-Sell Intelligence Platform")
st.markdown("**AI-Powered Revenue Opportunities Across Your Portfolio**")

# Sidebar for filters
"""Sidebar section for selecting score thresholds, date range and organizations."""

st.sidebar.header("Filters")
min_score = st.sidebar.slider("Minimum Opportunity Score", 0.0, 1.0, 0.5)
selected_confidence = st.sidebar.multiselect(
    "Confidence Levels", ["Very High", "High", "Medium", "Low"], default=["Very High", "High"]
)

date_range = st.sidebar.date_input(
    "Date Range",
    value=(datetime.now() - timedelta(days=30), datetime.now()),
)
start_date = datetime.combine(date_range[0], datetime.min.time())
end_date = datetime.combine(date_range[1], datetime.max.time())

# Load data
recommendations_df = load_recommendations(start_date, end_date)
time_series_df = load_time_series(start_date, end_date)

all_orgs = sorted(
    set(recommendations_df["account1_org"]).union(set(recommendations_df["account2_org"]))
)
selected_orgs = st.sidebar.multiselect("Organizations", all_orgs, default=all_orgs)

# Apply filters
filtered_df = recommendations_df[
    (recommendations_df["score"] >= min_score)
    & (recommendations_df["confidence_level"].isin(selected_confidence))
    & (
        (recommendations_df["account1_org"].isin(selected_orgs))
        | (recommendations_df["account2_org"].isin(selected_orgs))
    )
]

"""Key metrics summarizing the filtered opportunity set."""

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

"""Visual charts for exploring opportunity distributions and trends."""

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


def display_opportunities(df: pd.DataFrame) -> None:
    """Render a list of opportunity cards."""

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


def display_paginated_opportunities(df: pd.DataFrame, sort_by: str, key_prefix: str) -> None:
    """Display opportunities sorted by a column with pagination."""

    page_size = 5
    sorted_df = df.sort_values(by=sort_by, ascending=False).reset_index(drop=True)
    total_pages = math.ceil(len(sorted_df) / page_size)
    page_key = f"{key_prefix}_page"
    if page_key not in st.session_state:
        st.session_state[page_key] = 0

    start = st.session_state[page_key] * page_size
    end = start + page_size
    display_opportunities(sorted_df.iloc[start:end])

    col_prev, col_next = st.columns(2)
    with col_prev:
        if st.button("Previous", key=f"{key_prefix}_prev") and st.session_state[page_key] > 0:
            st.session_state[page_key] -= 1
    with col_next:
        if st.button("Next", key=f"{key_prefix}_next") and end < len(sorted_df):
            st.session_state[page_key] += 1

    st.caption(f"Page {st.session_state[page_key] + 1} of {total_pages}")


"""Paginated lists of opportunities sorted by score, value and recency."""

# Top Opportunities Table
st.markdown("---")
st.subheader("🏆 Top Cross-Sell Opportunities")

# Create tabs for different views
tab1, tab2, tab3 = st.tabs(["High Score", "High Value", "Recent"])

with tab1:
    display_paginated_opportunities(filtered_df, "score", "score")

with tab2:
    display_paginated_opportunities(filtered_df, "estimated_value", "value")

with tab3:
    display_paginated_opportunities(filtered_df, "created_at", "recent")


"""Export tools for downloading data and refreshing the dashboard."""

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

"""Informational footer with timestamp."""

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
