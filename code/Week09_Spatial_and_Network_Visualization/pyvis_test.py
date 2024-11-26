import networkx as nx
from pyvis.network import Network

G = nx.karate_club_graph()
net = Network()  	
net.from_nx(G)
net.write_html('karate_club.html') 
