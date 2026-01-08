import streamlit as st
import pandas as pd

# 1. Add a title
st.title("My First Local App")

# 2. Add some text
st.write("Welcome! This app is running on my own laptop.")

# 3. Create some sample data
data = {
    'Sales': [10, 20, 15, 25],
    'Month': ['Jan', 'Feb', 'Mar', 'Apr']
}
df = pd.DataFrame(data)

# 4. Display a chart
st.subheader("Monthly Sales")
st.bar_chart(df.set_index('Month'))