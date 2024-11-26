## Network Analysis Tools
## Veera Muangsin

import streamlit as st
import networkx as nx
import pandas as pd
import numpy as np
from pyvis.network import Network
import matplotlib.pyplot as plt
import tempfile
import os
from typing import Dict, Optional, Tuple

@st.cache_data
def load_network_from_file(uploaded_file, file_format):
    """Load network from various file formats"""
    try:
        if file_format == 'CSV':
            df = pd.read_csv(uploaded_file, engine="python", sep=r",|\s+")
            if len(df.columns) < 2:
                raise ValueError("CSV file must have at least two columns for source and target nodes")
            source_col, target_col = df.columns[0], df.columns[1]
            G = nx.from_pandas_edgelist(df, source=source_col, target=target_col)
                
        elif file_format == 'GML':
            content = uploaded_file.getvalue().decode('utf-8')
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            G = nx.read_gml(tmp_path)
            os.unlink(tmp_path)
            
        elif file_format == 'GraphML':
            content = uploaded_file.getvalue().decode('utf-8')
            with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp:
                tmp.write(content)
                tmp_path = tmp.name
            G = nx.read_graphml(tmp_path)
            os.unlink(tmp_path)
           
        if len(G.nodes()) == 0:
            raise ValueError("The resulting graph has no nodes")
            
        return G
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        return None

class NetworkVisualizer:
    def __init__(self, G: nx.Graph):
        self.G = G
        self.colors = [
            "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231", 
            "#911eb4", "#46f0f0", "#f032e6", "#bcf60c", "#fabebe",
            "#008080", "#e6beff", "#9a6324", "#fffac8", "#800000"
        ]

    def _get_layout(self, layout_type: str, G: nx.Graph, k_space: float = 2.0):
        """Calculate layout positions with adjustable spacing"""
        if layout_type == "spring":
            k = 1/np.sqrt(len(G.nodes())) * k_space
            return nx.spring_layout(G, k=k, iterations=50, seed=42)
        elif layout_type == "kamada_kawai":
            return nx.kamada_kawai_layout(G)
        elif layout_type == "circular":
            return nx.circular_layout(G)
        elif layout_type == "random":
            return nx.random_layout(G, seed=42)
        else:
            return nx.spring_layout(G)

    def create_interactive_network(
        self, 
        communities: Optional[Dict] = None,
        layout: str = "spring",
        centrality_metric: str = "degree",
        scale_factor: float = 1000,
        node_spacing: float = 2.0,
        node_size_range: Tuple[int, int] = (10, 50),
        show_edges: bool = True, 
        font_size: int = 14 
    ) -> str:
        # Get layout positions with adjustable spacing
        pos = self._get_layout(layout, self.G, node_spacing)
        
        # Scale positions
        pos = {node: (coord[0] * scale_factor, coord[1] * scale_factor) 
               for node, coord in pos.items()}

        # Calculate centrality
        try:
            if centrality_metric == "degree":
                centrality = nx.degree_centrality(self.G)
            elif centrality_metric == "betweenness":
                centrality = nx.betweenness_centrality(self.G)
            elif centrality_metric == "closeness":
                centrality = nx.closeness_centrality(self.G)
            else:  # pagerank
                centrality = nx.pagerank(self.G)
        except:
            centrality = nx.degree_centrality(self.G)
            st.warning(f"Failed to compute {centrality_metric} centrality, using degree centrality instead.")

        # Scale node sizes
        min_cent, max_cent = min(centrality.values()), max(centrality.values())
        min_size, max_size = node_size_range
        if max_cent > min_cent:
            size_scale = lambda x: min_size + (x - min_cent) * (max_size - min_size) / (max_cent - min_cent)
        else:
            size_scale = lambda x: (min_size + max_size) / 2

        # Create a copy of the graph to modify attributes
        G_vis = self.G.copy()

        # Prepare color map for communities if present
        if communities:
            unique_communities = sorted(set(communities.values()))
            color_map = {com: self.colors[i % len(self.colors)] 
                        for i, com in enumerate(unique_communities)}

        # Set node attributes all at once
        for node in G_vis.nodes():
            G_vis.nodes[node].update({
                'label': str(node),
                'size': size_scale(centrality[node]),
                'x': pos[node][0],
                'y': pos[node][1],
                'physics': False,
                'title': (f"Node: {node}\nDegree: {self.G.degree(node)}\nCommunity: {communities[node]}"
                         if communities else
                         f"Node: {node}\nDegree: {self.G.degree(node)}"),
                'color': color_map[communities[node]] if communities else None
            })

        # Create network
        nt = Network(
            height="720px",
            width="100%",
            bgcolor="#ffffff",
            font_color="black",
            directed=self.G.is_directed()
        )

        # Convert from networkx to pyvis
        nt.from_nx(G_vis)
        
        # Disable physics
        nt.toggle_physics(False)

        # Set visualization options
        nt.set_options("""
        {
            "nodes": {
                "font": {"size": %d},
                "borderWidth": 2,
                "borderWidthSelected": 3,
                "shape": "dot"
            },
            "edges": {
                "color": {"color": "#666666"},
                "width": 1.5,
                "smooth": {
                    "type": "continuous",
                    "roundness": 0.5
                },
                "hidden": %s
            },
            "physics": {
                "enabled": false
            },
            "interaction": {
                "hover": true,
                "multiselect": true,
                "navigationButtons": true,
                "tooltipDelay": 100,
                "zoomView": true,
                "dragView": true,
                "zoomSpeed": 0.5,
                "minZoom": 1.0,
                "maxZoom": 2.5
            }
        }
        """ % (font_size, str(not show_edges).lower()))

        with tempfile.NamedTemporaryFile(delete=False, suffix='.html', mode='w', encoding='utf-8') as tmp:
            nt.save_graph(tmp.name)
            return tmp.name

class NetworkAnalyzer:
    def __init__(self, G: nx.Graph):
        self.G = G
        
    def get_basic_stats(self) -> Dict:
        """Calculate basic network statistics"""
        try:
            diameter = nx.diameter(self.G)
        except:
            diameter = None
            
        return {
            "Nodes": len(self.G.nodes()),
            "Edges": len(self.G.edges()),
            "Density": nx.density(self.G),
            "Diameter": diameter
        }
    
    def get_centralities(self) -> pd.DataFrame:
        """Calculate various centrality metrics"""
        centrality_metrics = {}
        try:
            centrality_metrics['Degree'] = nx.degree_centrality(self.G)
            centrality_metrics['Betweenness'] = nx.betweenness_centrality(self.G)
            centrality_metrics['Closeness'] = nx.closeness_centrality(self.G)
            centrality_metrics['PageRank'] = nx.pagerank(self.G)
        except Exception as e:
            st.warning(f"Some centrality metrics couldn't be computed: {str(e)}")
        
        df = pd.DataFrame(centrality_metrics)
        df.index.name = 'Node'
        return df.reset_index()
    
    def plot_degree_distribution(self) -> plt.Figure:
        """Plot degree distribution"""
        degrees = [d for n, d in self.G.degree()]
        fig, ax = plt.subplots(figsize=(6, 4))
        plt.hist(degrees, bins=range(max(degrees) + 2), align='left', rwidth=0.8)
        ax.set_xlabel('Degree')
        ax.set_ylabel('Frequency')
        ax.set_title('Degree Distribution')
        plt.tight_layout()
        return fig

@st.cache_data
def detect_communities(edges_str: str):
    """Community detection"""
    # Recreate graph from edges string
    edges = eval(edges_str)
    G = nx.Graph(edges)
    return list(nx.community.greedy_modularity_communities(G))

def main():
    st.set_page_config(page_title="Network Analysis Tool", layout="wide")
    
    st.title("Network Analysis Tool")
    
    # Create tabs
    tab_viz, tab_analysis = st.tabs(["Network Visualization", "Network Analysis"])
    
    # Input and Visualization tab
    with tab_viz:
        # Move input controls to sidebar
        with st.sidebar:
            st.subheader("Network Input")
            data_source = st.radio(
                "Choose data source",
                ["Sample Networks", "Graph Generator", "Upload Network"]
            )
            
            # Get network based on selection
            G = None  # Initialize G
            if data_source == "Sample Networks":
                sample_option = st.selectbox(
                    "Select sample network",
                    ["Karate Club", "Les Miserables"]
                )
                G = nx.karate_club_graph() if sample_option == "Karate Club" else nx.les_miserables_graph()
                    
            elif data_source == "Graph Generator":
                generator_type = st.selectbox(
                    "Select generator type",
                    ["Complete Graph", "Random (Erdős-Rényi)", "Small World (Watts-Strogatz)", "Scale-free (Barabási-Albert)"]
                )
                
                if generator_type == "Complete Graph":
                    n = st.slider("Number of nodes", 3, 100, 20)
                    G = nx.complete_graph(n)
                    
                elif generator_type == "Random (Erdős-Rényi)":
                    n = st.slider("Number of nodes", 3, 100, 20)
                    p = st.slider("Edge probability", 0.0, 1.0, 0.2)
                    G = nx.erdos_renyi_graph(n, p)
                    
                elif generator_type == "Small World (Watts-Strogatz)":
                    n = st.slider("Number of nodes", 100, 1000, 200)
                    k = st.slider("Number of nearest neighbors", 4, 20, 6)
                    p = st.slider("Rewiring probability", 0.0, 1.0, 0.1)
                    G = nx.watts_strogatz_graph(n, k, p)
                    
                else:  # Barabási-Albert
                    n = st.slider("Number of nodes", 100, 1000, 200)
                    m = st.slider("Number of edges to attach", 1, 10, 3)
                    G = nx.barabasi_albert_graph(n, m)
                    
            else:  # Upload Network
                file_format = st.selectbox(
                    "Choose file format",
                    ["CSV", "GML", "GraphML"]
                )
                
                uploaded_file = st.file_uploader(
                    "Upload network file",
                    type=["csv", "gml", "graphml"]
                )
                
                if uploaded_file:
                    G = load_network_from_file(uploaded_file, file_format)
                    if G is None:
                        st.info("Please ensure your file is properly formatted")
                        st.stop()
                else:
                    st.info("Please upload a network file or use other data sources")
                    st.stop()
            
            if G is not None:  # Only show options if we have a graph
                # Visualization options
                st.markdown("---")  # Add separator
                st.subheader("Visualization Options")
                
                layout_option = st.selectbox(
                    "Layout Algorithm",
                    ["spring", "kamada_kawai", "circular", "random"]
                )
                centrality_option = st.selectbox(
                    "Node Size By", 
                    ["degree", "betweenness", "closeness", "pagerank"]
                )
                
                # Size controls 
                scale_factor = st.slider(
                    "Graph Size", 
                    min_value=500, 
                    max_value=3000, 
                    value=1000,
                    step=100,
                    help="Adjust the overall size of the graph"
                )
                
                if layout_option == "spring":
                    node_spacing = st.slider(
                        "Node Spacing",
                        min_value=1.0,
                        max_value=20.0,
                        value=5.0,
                        step=1.0,
                        help="Adjust the spacing between nodes (only for spring layout)"
                    )
                else:
                    node_spacing = 2.0
                
                node_size_range = st.slider(
                    "Node Size Range",
                    min_value=5,
                    max_value=200,
                    value=(10, 50),
                    step=5,
                    help="Set the minimum and maximum node sizes"
                )
                
                # Node label font size
                font_size = st.slider(
                    "Label Font Size",
                    min_value=8,
                    max_value=40,
                    value=16,
                    step=2,
                    help="Adjust the font size of node labels"
                )                
                
                # Edge visibility
                show_edges = st.toggle(
                    "Show Edges",
                    value=True,  
                    help="Toggle edge visibility"
                )
                
                # Community detection
                st.markdown("---")  # Add separator
                show_communities = st.checkbox("Detect Communities")
                
                # Initialize communities variable
                communities = None
                community_stats_container = st.empty()  # Create a placeholder for stats
                
                if show_communities:
                    try:
                        edges_str = str(list(G.edges()))
                        communities_iter = detect_communities(edges_str)
                        communities = {}
                        
                        # Create a list to store community sizes
                        community_sizes = []
                        
                        for idx, community in enumerate(communities_iter):
                            community_sizes.append(len(community))
                            for node in community:
                                communities[node] = idx
                        
                        # Use the placeholder to display community statistics
                        with community_stats_container.container():
                            st.caption("Community Statistics")
                            st.metric("Number of Communities", len(communities_iter))
                            avg_size = sum(community_sizes) / len(community_sizes)
                            st.metric("Average Community Size", f"{avg_size:.1f}")
                            
                            # Sort communities by size in descending order
                            community_df = pd.DataFrame({
                                'Community': range(len(community_sizes)),
                                'Size': community_sizes
                            }).sort_values('Size', ascending=False)
                            
                            st.caption("Community Sizes (sorted by size):")
                            st.dataframe(
                                community_df,
                                hide_index=True,
                                height=min(len(community_sizes) * 35 + 38, 300)
                            )
                            
                    except Exception as e:
                        st.warning(f"Could not detect communities: {str(e)}")
        
        # Main visualization area (now full width)
        if G is not None:
            # Initialize analyzers
            analyzer = NetworkAnalyzer(G)
            visualizer = NetworkVisualizer(G)
            
            # Display basic stats in a more compact form
            st.text(
                f"Nodes: {len(G.nodes())} | Edges: {len(G.edges())} | "
                f"Density: {nx.density(G):.3f}"
            )
            
            # Create and display visualization with increased height
            html_file = visualizer.create_interactive_network(
                communities=communities,
                layout=layout_option,
                centrality_metric=centrality_option,
                scale_factor=scale_factor,
                node_spacing=node_spacing,
                node_size_range=node_size_range,
                show_edges=show_edges,
                font_size=font_size
            )
            
            with open(html_file, 'r', encoding='utf-8') as f:
                html_content = f.read()
            
            st.components.v1.html(html_content, height=800)
            os.unlink(html_file)
    
    # Analysis tab
    # Analysis tab
    with tab_analysis:
        if G is not None:
            # Initialize analyzer
            analyzer = NetworkAnalyzer(G)
            
            st.header("Network Analysis")
            
            # Get basic statistics
            stats = analyzer.get_basic_stats()
            
            # Basic Statistics
            st.subheader("Basic Statistics")
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Nodes", stats["Nodes"])
            with col2:
                st.metric("Edges", stats["Edges"])
            with col3:
                st.metric("Density", f"{stats['Density']:.3f}")
            with col4:
                if stats["Diameter"]:
                    st.metric("Diameter", stats["Diameter"])
            
            # Create columns for degree distribution and centrality analysis
            st.markdown("---")  # Add separator
            left_col, right_col = st.columns(2)
            
            # Degree Distribution in left column
            with left_col:
                st.subheader("Degree Distribution")
                fig = analyzer.plot_degree_distribution()
                st.pyplot(fig)
            
            # Centrality Analysis in right column
            with right_col:
                st.subheader("Centrality Analysis")
                centrality_df = analyzer.get_centralities()
                st.dataframe(
                    centrality_df,
                    height=400  # Fixed height to match the plot
                )
        else:
            st.info("Please select or upload a network in the Visualization tab")

if __name__ == "__main__":
    main()
