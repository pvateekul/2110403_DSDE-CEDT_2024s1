import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# Example 1: Basic Text Elements
# Reference: https://docs.streamlit.io/library/api-reference/text
def text_elements():
    st.title("Welcome to Streamlit!")
    st.header("This is a header")
    st.subheader("This is a subheader")
    st.text("This is plain text")
    st.markdown("**This** is *markdown*")
    st.write("Streamlit's write function can handle multiple data types")
    st.code("print('Hello World')", language='python')

# Example 2: Data Display
# Reference: https://docs.streamlit.io/library/api-reference/data
def data_display():
    df = pd.DataFrame({
        'name': ['John', 'Mary', 'Bob'],
        'age': [25, 30, 35],
        'city': ['New York', 'London', 'Paris']
    })
    
    st.dataframe(df)  # Interactive dataframe
    st.table(df)      # Static table
    st.json({'foo': 'bar'})  # JSON viewer

# Example 3: Advanced Charts and Plots
# Reference: https://docs.streamlit.io/library/api-reference/charts
def charts():
    st.header("Basic Streamlit Charts")
    # Generate sample data
    chart_data = pd.DataFrame(
        np.random.randn(20, 3),
        columns=['Line 1', 'Line 2', 'Line 3']
    )
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Line Chart")
        st.line_chart(chart_data)
    
    with col2:
        st.subheader("Area Chart")
        st.area_chart(chart_data)

    # Plotly Charts
    st.header("Plotly Charts")
    
    # Sample data for more complex charts
    dates = pd.date_range(start='2024-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'Date': dates,
        'Sales': np.random.randint(100, 1000, size=100),
        'Revenue': np.random.uniform(1000, 5000, size=100),
        'Category': np.random.choice(['A', 'B', 'C'], size=100)
    })

    # Plotly Scatter Plot
    fig_scatter = px.scatter(
        data, x='Sales', y='Revenue',
        color='Category',
        title="Sales vs Revenue"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Plotly Time Series
    fig_time = px.line(
        data, x='Date', y='Sales',
        color='Category',
        title="Time Series Analysis"
    )
    st.plotly_chart(fig_time, use_container_width=True)

    # Custom Plotly Chart with Multiple Traces
    st.header("Custom Plotly Chart")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Sales'],
        name='Sales', mode='lines'
    ))
    fig.add_trace(go.Scatter(
        x=data['Date'], y=data['Revenue'],
        name='Revenue', mode='lines',
        yaxis='y2'
    ))
    
    fig.update_layout(
        title='Sales and Revenue Over Time',
        yaxis=dict(title='Sales'),
        yaxis2=dict(title='Revenue', overlaying='y', side='right')
    )
    
    st.plotly_chart(fig, use_container_width=True)

    # Plotly Bar Chart
    fig_bar = px.bar(
        data.groupby('Category').mean().reset_index(),
        x='Category',
        y=['Sales', 'Revenue'],
        title="Average Sales and Revenue by Category",
        barmode='group'
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# Example 4: Input Widgets
# Reference: https://docs.streamlit.io/library/api-reference/widgets
def input_widgets():
    name = st.text_input("Enter your name")
    age = st.number_input("Enter your age", min_value=0, max_value=120)
    is_happy = st.checkbox("Are you happy?")
    color = st.selectbox("Choose a color", ["red", "green", "blue"])
    
    slider_val = st.slider("Select a value", 0, 100)
    
    if st.button("Submit"):
        st.write(f"Hello {name}, you are {age} years old!")
        st.write(f"Happiness status: {is_happy}")
        st.write(f"Favorite color: {color}")
        st.write(f"Slider value: {slider_val}")

# Example 5: Advanced Layout and Containers
# Reference: https://docs.streamlit.io/library/api-reference/layout
def layout():
    st.header("Basic Column Layout")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("Column 1")
        st.write("This is column 1")
        st.metric(label="Temperature", value="24 °C", delta="1.2 °C")
    
    with col2:
        st.subheader("Column 2")
        st.write("This is column 2")
        st.metric(label="Humidity", value="48%", delta="-2%")
    
    with col3:
        st.subheader("Column 3")
        st.write("This is column 3")
        st.metric(label="Pressure", value="1013 hPa", delta="0.5 hPa")

    st.header("Nested Layouts")
    
    # Tabs
    tab1, tab2, tab3 = st.tabs(["Tab 1", "Tab 2", "Tab 3"])
    
    with tab1:
        st.write("This is content in tab 1")
        col1, col2 = st.columns(2)
        with col1:
            st.write("Nested column 1")
        with col2:
            st.write("Nested column 2")
    
    with tab2:
        st.write("This is content in tab 2")
        with st.expander("Click to expand"):
            st.write("Hidden content in tab 2")
    
    with tab3:
        st.write("This is content in tab 3")
        
   

# Main App Structure
def main():
    st.set_page_config(
        page_title="Streamlit Examples",
        layout="wide"
    )
    
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose an example",
        ["Text Elements", "Data Display", "Charts", 
         "Input Widgets", "Layout"]
    )
    
    if page == "Text Elements":
        text_elements()
    elif page == "Data Display":
        data_display()
    elif page == "Charts":
        charts()
    elif page == "Input Widgets":
        input_widgets()
    elif page == "Layout":
        layout()

if __name__ == "__main__":
    main()
