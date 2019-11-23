import numpy as np 
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.flow import edmonds_karp, shortest_augmenting_path

# MAS711 AY1617 Qn 1
G = nx.DiGraph()
G.add_edge('s','1', capacity=2)
G.add_edge('s','2', capacity=2)
G.add_edge('s','3', capacity=2)

G.add_edge('1','4', capacity=4)

G.add_edge('2','1', capacity=3)
G.add_edge('2','5', capacity=7)
G.add_edge('2','6', capacity=3)
G.add_edge('2','3', capacity=2)

G.add_edge('3','6', capacity=5)

G.add_edge('4','5', capacity=4)
G.add_edge('4','t', capacity=3)

G.add_edge('5','t', capacity=2)

G.add_edge('6','5', capacity=6)
G.add_edge('6','t', capacity=2)

G.nodes['s']['pos'] = (0,0)
G.nodes['1']['pos'] = (1,1)
G.nodes['2']['pos'] = (1,0)
G.nodes['3']['pos'] = (1,-1)
G.nodes['4']['pos'] = (2,1)
G.nodes['5']['pos'] = (2,0)
G.nodes['6']['pos'] = (2,-1)
G.nodes['t']['pos'] = (3,0)

fig = plt.figure(figsize=(10,5))
node_pos = nx.get_node_attributes(G,'pos')
capacities = nx.get_edge_attributes(G,'capacity')
nx.draw_networkx(G, node_pos, node_color='orange', node_size=800)
nx.draw_networkx_edges(G, node_pos,edge_color= 'black')
text = nx.draw_networkx_edge_labels(G, node_pos,edge_color= 'black', edge_labels=capacities, font_size=14)
for _,t in text.items():
    t.set_rotation('horizontal')
ax = plt.gca() # to get the current axis
ax.collections[0].set_edgecolor("#000000")
ax.collections[0].set_linewidth(2)
plt.axis('off')

flow_value, flow_dict = nx.maximum_flow(G, 's', 't')
cut_value, partition = nx.minimum_cut(G, 's', 't')