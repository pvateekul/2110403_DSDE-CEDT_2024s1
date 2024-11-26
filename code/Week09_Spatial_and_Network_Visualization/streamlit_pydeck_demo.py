# streamlit_pydeck_demo.py

import streamlit as st
import pandas as pd
import pydeck as pdk
import math

# Function to load data
@st.cache_data
def load_data():
    df = pd.read_json("https://raw.githubusercontent.com/visgl/deck.gl-data/master/website/bart-stations.json")
    df[['longitude', 'latitude']] = pd.DataFrame(df['coordinates'].tolist(), index=df.index)
    df["exits_radius"] = df["exits"].apply(lambda exits_count: math.sqrt(exits_count))
    return df

df = load_data()

# Sidebar controls
map_layer_type = st.sidebar.radio('Map Type', ["ScatterplotLayer", "HeatmapLayer"])

map_style = st.sidebar.radio(
    "Map Style",
    ["dark", "light", "streets", "satellite"]
)
map_style = "mapbox://styles/mapbox/" + map_style + "-v9"

opacity = st.sidebar.slider('Opacity', min_value=0.0, max_value=1.0, value=0.5)
radius_scale = st.sidebar.slider('Radius Scale', min_value=1.0, max_value=10.0, value=5.0)
color_choices = st.sidebar.radio('Color Palette', ['Reds', 'Blues', 'Greens', 'Purples', 'Oranges'])

# Color mapping based on choice
color_mapping = {
    "Reds": [255, 0, 0, 140],
    "Blues": [0, 0, 255, 140],
    "Greens": [0, 255, 0, 140],
    "Purples": [128, 0, 128, 140],
    "Oranges": [255, 165, 0, 140]
}
color = color_mapping[color_choices]


# Main app
st.title('PyDeck Demo')

# Function to create map
def create_map(dataframe):
    if map_layer_type == "ScatterplotLayer":
        layer = pdk.Layer(
            "ScatterplotLayer",
            dataframe,
            get_position=["longitude", "latitude"],
            get_color=color,
            get_radius="exits_radius",
            radius_scale=radius_scale,
            opacity=opacity,
            pickable=True
        )
    elif map_layer_type == "HeatmapLayer":
        layer = pdk.Layer(
            "HeatmapLayer",
            dataframe,
            get_position=["longitude", "latitude"],
            #get_weight="exits",  
            opacity=opacity,
            pickable=True
        )

    view_state = pdk.ViewState(
        longitude=dataframe['longitude'].mean(),
        latitude=dataframe['latitude'].mean(),
        zoom=9
    )

    return pdk.Deck(layers=[layer], initial_view_state=view_state, map_style=map_style, tooltip={"text": "{name}\n{address}"})


# Display Map
st.write('### Map')
map = create_map(df)
st.pydeck_chart(map)

# Display data
st.dataframe(df)


