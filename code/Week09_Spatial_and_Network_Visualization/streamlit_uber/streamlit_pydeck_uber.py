import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk

st.title('Uber pickups in NYC')

DATE_COLUMN = 'date/time'
DATA_URL = ('https://s3-us-west-2.amazonaws.com/streamlit-demo-data/uber-raw-data-sep14.csv.gz')

@st.cache_data
def load_data(nrows):
    data = pd.read_csv(DATA_URL, nrows=nrows)
    lowercase = lambda x: str(x).lower()
    data.rename(lowercase, axis='columns', inplace=True)
    data[DATE_COLUMN] = pd.to_datetime(data[DATE_COLUMN])
    return data

# Load data
data = load_data(10000)

# Add map style selection in sidebar
st.sidebar.header('Map Settings')
map_style = st.sidebar.selectbox(
    'Select Base Map Style',
    options=['Dark', 'Light', 'Road', 'Satellite'],
    index=0
)

# Define map style dictionary
MAP_STYLES = {
    'Dark': 'mapbox://styles/mapbox/dark-v10',
    'Light': 'mapbox://styles/mapbox/light-v10',
    'Road': 'mapbox://styles/mapbox/streets-v11',
    'Satellite': 'mapbox://styles/mapbox/satellite-v9'
}

if st.checkbox('Show raw data'):
    st.subheader('Raw data')
    st.write(data)

# Create hour column for easier filtering
data['hour'] = data[DATE_COLUMN].dt.hour

# Hour range filter with slider
st.subheader('Filter by Hour Range')
hour_range = st.slider(
    'Select hour range',
    min_value=0,
    max_value=23,
    value=(0, 23)  # Default to common business hours
)

# Filter data based on hour range
filtered_data = data[
    (data['hour'] >= hour_range[0]) & 
    (data['hour'] <= hour_range[1])
]

# Show statistics for selected range
st.write(f'Showing {len(filtered_data)} pickups between {hour_range[0]}:00 and {hour_range[1]}:00')


# Add time distribution for selected range
st.subheader('Pickup Distribution in Selected Time Range')
hourly_counts = filtered_data['hour'].value_counts().sort_index()
st.bar_chart(hourly_counts)

# Map visualization options
st.subheader(f'Map of all pickups between {hour_range[0]}:00 and {hour_range[1]}:00')
map_type = st.radio('Select Map Type', ['Points', 'Heatmap'])

# Calculate map center
center_lat = filtered_data['lat'].mean()
center_lon = filtered_data['lon'].mean()

# Create map layers based on selection
def create_map_layers(data, map_type):
    if map_type == 'Points':
        return [
            pdk.Layer(
                "ScatterplotLayer",
                data,
                get_position=['lon', 'lat'],
                get_color=[0, 255, 0, 160],
                get_radius=50,
                opacity=0.8,
                pickable=True
            )
        ]
    else:
        return [
            pdk.Layer(
                "HeatmapLayer",
                data,
                get_position=['lon', 'lat'],
                opacity=0.8,
                radiusPixels=50,
            )
        ]

# Create and display the map
deck = pdk.Deck(
    layers=create_map_layers(filtered_data, map_type),
    initial_view_state=pdk.ViewState(
        latitude=center_lat,
        longitude=center_lon,
        zoom=11,
        pitch=0,
    ),
    map_style=MAP_STYLES[map_style],
    tooltip={
        "html": "<b>Pickup Time:</b> {date/time}<br/>"
                "<b>Hour:</b> {hour}:00"
    }
)

st.pydeck_chart(deck)


