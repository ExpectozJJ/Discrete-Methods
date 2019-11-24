import numpy as np 
import networkx as nx 
from networkx.drawing.nx_pydot import graphviz_layout
import pylab as plt

# Created by Wee JunJie
# Dated 24th Nov 2019

def Prufer_Code(T):
    # Returns the Prufer Code given a Tree of n vertices 😛

    code = []
    if nx.is_tree(T):
        while len(T.nodes()) != 2:
            leaves = []
            for t in T.degree():
                if t[1] == 1:
                    leaves.append(t[0])
            leaves = np.sort(leaves)
            for l in leaves:
                for node in nx.neighbors(T, l):
                    code.append(node)
                    T.remove_edge(node,l)
                    T.remove_node(l)
                    break
                break
            print(code)
    else:
        raise ValueError("Input is not a Tree!")
    
    return code


def Prufer_Reconstruct(seq):
    # Reconstructs a Tree of n vertices given an input of a Prufer Code of length n-2

    T = nx.Graph()
    n = len(seq) + 2
    id = list(range(1, n+1))
    for j in range(len(seq)):
        for i in range(len(id)):
            if id[i] not in seq[j:]:
                T.add_edge(seq[j], id[i])
                #print(seq[j], id[i])
                del id[i]
                break
    #print(id)
    T.add_edge(id[0],id[1])
    return T

fig = plt.figure(figsize=(10,5))
T = Prufer_Reconstruct([1,1,1,2,2,2,3,3,3])
pos = nx.nx_agraph.graphviz_layout(T, prog='neato')
nx.draw_networkx(T, node_pos=pos, node_color='pink', node_size=800)
ax = plt.gca() # to get the current axis
ax.collections[0].set_edgecolor("#000000")
ax.collections[0].set_linewidth(2)
plt.axis('off')
plt.show()

pc = Prufer_Code(T)
print("Prufer Code: ", pc)
