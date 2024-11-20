import streamlit as st
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from io import StringIO
    

def calculate_centroid_distance(centroid1, centroid2):
    # Calculate Euclidean distance between two centroids.
    return np.sqrt(np.sum((centroid1 - centroid2) ** 2))

def generate_cluster_params(n_clusters, x_range=(-5, 5), y_range=(-5, 5), min_distance=1):
    # Generate random parameters for clusters with minimum separation.
    
    # Create a random number generator
    rng = np.random.default_rng()
    
    # Generate random centroids with minimum separation
    centroids = []
    max_attempts = 100  # Prevent infinite loops
    
    for _ in range(n_clusters):
        attempt = 0
        while attempt < max_attempts:
            # Generate candidate centroid
            new_centroid = np.array([
                rng.uniform(x_range[0], x_range[1]),
                rng.uniform(y_range[0], y_range[1])
            ])
            
            # Check distance from all existing centroids
            if not centroids:  # First centroid
                centroids.append(new_centroid)
                break
            
            # Calculate distances to all existing centroids
            distances = [calculate_centroid_distance(new_centroid, c) for c in centroids]
            
            if min(distances) >= min_distance:
                centroids.append(new_centroid)
                break
                
            attempt += 1
        
        if attempt >= max_attempts:
            # If we can't place the centroid after max attempts, adjust the minimum distance
            min_distance *= 0.8
            attempt = 0
    
    centroids = np.array(centroids)
    
    # Calculate minimum distance between any pair of centroids
    min_observed_distance = float('inf')
    for i in range(len(centroids)):
        for j in range(i + 1, len(centroids)):
            dist = calculate_centroid_distance(centroids[i], centroids[j])
            min_observed_distance = min(min_observed_distance, dist)
    
    # Generate scales based on minimum distance to ensure non-overlapping clusters
    # Use smaller scale for clusters that are closer to others
    max_scale = min_observed_distance / 2  # Limit scale to prevent overlap
    scales = rng.uniform(max_scale * 0.3, max_scale, size=(n_clusters, 2))
    
    # Generate random rotation angles for each cluster
    rotations = rng.uniform(0, 2 * np.pi, size=n_clusters)
    
    # Generate random proportion of samples per cluster. The proportions sum to 1.
    proportions = rng.dirichlet(np.ones(n_clusters))
    
    return centroids, scales, rotations, proportions

def generate_cluster(n_points, centroid, scale, rotation):
    # Generate a single cluster with specified parameters.
    
    # Create a random number generator
    rng = np.random.default_rng()
    
    # Generate normal distribution with small standard deviation (scale=0.5) to keep points clustered tightly
    points = rng.normal(loc=0, scale=0.5, size=(n_points, 2))  # Re standard deviation
    
    # Apply scale
    points *= scale
    
    # Apply rotation using matrix multiplication
    rotation_matrix = np.array([
        [np.cos(rotation), -np.sin(rotation)],
        [np.sin(rotation), np.cos(rotation)]
    ])
    points = points @ rotation_matrix
    
    # Apply translation (move to centroid)
    points += centroid
    
    return points

def generate_sample_data(n_samples, n_clusters, distribution_type='normal', 
                        x_range=(-5, 5), y_range=(-5, 5)):
    # Generate sample data with specified parameters.
    
    # Adjust minimum distance based on the range and number of clusters
    range_size = min(x_range[1] - x_range[0], y_range[1] - y_range[0])
    min_distance = range_size / (n_clusters + 1)  # Dynamic minimum distance
    
    # Generate cluster parameters
    centroids, scales, rotations, proportions = generate_cluster_params(
        n_clusters, x_range, y_range, min_distance
    )
    
    # Calculate number of points per cluster
    points_per_cluster = (proportions * n_samples).astype(int)
    # Adjust last cluster to ensure total equals n_samples
    points_per_cluster[-1] = n_samples - np.sum(points_per_cluster[:-1])
    
    # Generate clusters
    clusters = []
    true_labels = []
    for i in range(n_clusters):
        cluster_points = generate_cluster(
            points_per_cluster[i],
            centroids[i],
            scales[i],
            rotations[i]
        )
        clusters.append(cluster_points)
        true_labels.extend([i] * points_per_cluster[i])
    
    # Combine all clusters
    X = np.vstack(clusters)
    
    return X, np.array(true_labels), centroids

def perform_elbow_analysis(data, max_k=10):
    #Perform elbow analysis for k-means clustering.
    
    inertias = []
    K = range(1, max_k+1)
    
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(data)
        inertias.append(kmeans.inertia_)
    
    return K, inertias

def plot_clustering_comparison(df, centers=None, title_suffix=""):
    # Create side-by-side plots for true vs predicted labels.
    
    # Create figure with subplots
    fig = go.Figure()
    
    # Create subplot layout with increased vertical spacing for legend
    fig = make_subplots(rows=1, cols=2,
                       subplot_titles=(f"True Clusters {title_suffix}", 
                                     f"K-means Clustering Results {title_suffix}"),
                       vertical_spacing=0.2)  # Increase vertical spacing
    
    # Plot true labels
    for label in sorted(df['label'].unique()):
        mask = df['label'] == label
        fig.add_trace(
            go.Scatter(
                x=df[mask]['x'],
                y=df[mask]['y'],
                mode='markers',
                name=f'True Cluster {label}',
                showlegend=True,
                marker=dict(size=8)
            ),
            row=1, col=1
        )
    
    # Plot predicted labels
    for label in sorted(df['predicted'].unique()):
        mask = df['predicted'] == label
        fig.add_trace(
            go.Scatter(
                x=df[mask]['x'],
                y=df[mask]['y'],
                mode='markers',
                name=f'Predicted Cluster {label}',
                showlegend=True,
                marker=dict(size=8)
            ),
            row=1, col=2
        )
    
    # Add cluster centers if provided
    if centers is not None:
        fig.add_trace(
            go.Scatter(
                x=centers[:, 0],
                y=centers[:, 1],
                mode='markers',
                marker=dict(
                    color='black',
                    symbol='x',
                    size=12,
                    line=dict(width=2)
                ),
                name='Cluster Centers'
            ),
            row=1, col=2
        )
    
    # Update layout with more space at the top and adjusted legend position
    fig.update_layout(
        height=700,  # Increased height
        width=1200,
        margin=dict(t=150),  # Increased top margin
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=1.2,  # Moved legend higher
            xanchor="center",
            x=0.5,  # Centered horizontally
            orientation="h"
        ),
        title=dict(y=0.95)  # Adjusted title position
    )
    
    # Update axes labels
    fig.update_xaxes(title_text="x", row=1, col=1)
    fig.update_xaxes(title_text="x", row=1, col=2)
    fig.update_yaxes(title_text="y", row=1, col=1)
    fig.update_yaxes(title_text="y", row=1, col=2)
    
    # Make sure both plots have the same scale
    x_range = [df['x'].min() - 0.5, df['x'].max() + 0.5]
    y_range = [df['y'].min() - 0.5, df['y'].max() + 0.5]
    
    fig.update_xaxes(range=x_range)
    fig.update_yaxes(range=y_range)
    
    return fig

def main():
    st.set_page_config(layout="wide")
    st.title("K-means Clustering Demonstration")

    
    # File upload
    st.sidebar.header("Data Upload")
    uploaded_file = st.sidebar.file_uploader("Upload your CSV file", type=["csv"])
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            # Check if the dataframe has the required columns
            required_columns = {'x', 'y', 'label'}
            if not required_columns.issubset(df.columns):
                st.error("Please upload a CSV file with 'x', 'y', and 'label' columns.")
                return
            
            # Display raw data
            st.subheader("Raw Data Preview")
            st.write(df.head())
            
            # Prepare data for clustering
            features = df[['x', 'y']]
            
            # Normalize the data
            data_normalized = (features - features.mean()) / features.std()
            
            # Sidebar controls
            st.sidebar.header("Clustering Parameters")
            max_k = st.sidebar.slider("Maximum K for Elbow Analysis", min_value=2, max_value=15, value=10)
            
            # Perform elbow analysis
            K, inertias = perform_elbow_analysis(data_normalized, max_k)
            
            # Plot elbow curve
            st.subheader("Elbow Analysis")
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=list(K), y=inertias, mode='lines+markers'))
            fig_elbow.update_layout(
                title="Elbow Method for Optimal k",
                xaxis_title="Number of Clusters (k)",
                yaxis_title="Inertia",
                showlegend=False
            )
            st.plotly_chart(fig_elbow)
            
            # Select number of clusters
            k = st.sidebar.slider("Select number of clusters", min_value=2, max_value=max_k, value=3)
            
            # Perform k-means clustering
            kmeans = KMeans(n_clusters=k, random_state=42)
            cluster_labels = kmeans.fit_predict(data_normalized)
            
            # Add cluster labels to the dataframe
            df['predicted'] = cluster_labels
            
            # Calculate cluster centers in original scale
            centers = kmeans.cluster_centers_ * features.std().values + features.mean().values
            
            # Plot clustering comparison
            st.subheader("Clustering Results")
            fig = plot_clustering_comparison(df, centers)
            st.plotly_chart(fig)
            
            # Display cluster information
            st.subheader("Cluster Information")
            cluster_counts = pd.Series(cluster_labels).value_counts().sort_index()
            st.write("Points per cluster:", cluster_counts.to_dict())
            
        except Exception as e:
            st.error(f"Error processing the file: {str(e)}")
            
    else:
        # Enhanced sample data generation options
        st.sidebar.header("Sample Data Generation")
        
        # Sample data parameters
        n_samples = st.sidebar.slider("Number of points", 100, 1000, 300)
        n_clusters = st.sidebar.slider("Number of clusters", 2, 8, 3)
        distribution_type = "normal"
        
        col1, col2 = st.sidebar.columns(2)
        with col1:
            x_min = st.number_input("X min", value=-5.0)
            y_min = st.number_input("Y min", value=-5.0)
        with col2:
            x_max = st.number_input("X max", value=5.0)
            y_max = st.number_input("Y max", value=5.0)
        
        if st.sidebar.button("Generate Sample Data"):
            # Generate the sample data
            X, true_labels, true_centroids = generate_sample_data(
                n_samples=n_samples,
                n_clusters=n_clusters,
                distribution_type=distribution_type,
                x_range=(x_min, x_max),
                y_range=(y_min, y_max)
            )
            
            # Create DataFrame with standardized column names
            df = pd.DataFrame(X, columns=['x', 'y'])
            df['label'] = true_labels
            
            # Normalize data for k-means
            features = df[['x', 'y']]
            data_normalized = (features - features.mean()) / features.std()
            
            # Perform k-means with same number of clusters as generated
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            df['predicted'] = kmeans.fit_predict(data_normalized)
            
            # Calculate cluster centers in original scale
            centers = kmeans.cluster_centers_ * features.std().values + features.mean().values
            
            # Plot clustering comparison
            st.subheader("Clustering Results")
            fig = plot_clustering_comparison(
                df, 
                centers,
                title_suffix=f"({distribution_type.capitalize()} Distribution)"
            )
            st.plotly_chart(fig)
            
            # Display cluster information
            st.subheader("Cluster Information")
            cluster_counts = df.groupby('predicted').size()
            st.write("Points per cluster:", cluster_counts.to_dict())
            
            # Save to CSV
            csv_buffer = StringIO()
            df.to_csv(csv_buffer, index=False)
            
            # Create download button
            st.download_button(
                label="Download sample data",
                data=csv_buffer.getvalue(),
                file_name="sample_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()
