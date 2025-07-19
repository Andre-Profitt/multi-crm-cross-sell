import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='Test Dashboard', page_icon='🎯')

st.title('🎯 Cross-Sell Intelligence Platform')
st.markdown('**Testing Streamlit Setup**')

# Simple metrics
col1, col2, col3 = st.columns(3)
with col1:
    st.metric('Total Opportunities', '42')
with col2:
    st.metric('Potential Revenue', '$1.2M')
with col3:
    st.metric('Avg Score', '0.85')

# Simple chart
st.subheader('Sample Data')
df = pd.DataFrame({
    'Date': pd.date_range('2024-01-01', periods=10),
    'Opportunities': np.random.randint(5, 20, 10)
})
st.line_chart(df.set_index('Date'))

st.success('✅ If you can see this, Streamlit is working!')
