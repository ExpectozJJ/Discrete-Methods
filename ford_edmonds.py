import networkx as nx 
import numpy as np 
import math

# Created by Wee JunJie
# Dated 24th Nov 2019

def id(n):
    # Finds the index order of nodes
    if n == 's':
        d = 0
    elif n == 't':
        d = 9
    else: d = int(n)
    return d

def bfs(G, u, v):
    for n in G.nodes():
        if n == u:
            G.nodes[n]['bfs'] = 0
        else:
            G.nodes[n]['bfs'] = math.inf
    
    Q = [u]
    while Q != []:
        #print("Queue: ", Q)
        q = Q.pop(0)
        for n in np.sort(np.array(list(nx.neighbors(G, q)))):
            if id(n) > id(q):
                if G.nodes[n]['bfs'] == math.inf and G.edges[(q,n)]['flow'] < G.edges[(q,n)]['capacity']:
                    G.nodes[n]['bfs'] = G.nodes[q]['bfs'] + 1
                    G.nodes[n]['predecessor'] = q
                    Q.append(n)
        #print("Queue: ", Q)
        if len(Q) > 0:
            u = Q[0]
    
    predecessors = nx.get_node_attributes(G, 'predecessor')
    for node, pre in predecessors.items():
        print("G[{}]: {}".format(node,pre), end=" ")
    print("\n")
    return G

def ford_fulkerson(G, u, v):
    # Finds a Maximum Flow of a given network from u to v but with any augmenting path 

    for e in G.edges():
        G.edges[e]['flow'] = 0

    flow = 1
    while flow != 0:
        for path in nx.all_simple_paths(G, u, v):
            feasible = 1
            for i in range(len(path)-1):
                if G.edges[(path[i],path[i+1])]['capacity'] - G.edges[(path[i],path[i+1])]['flow'] <= 0:
                    feasible = 0
            if feasible == 1:
                p = path
                break

            delta = []
            for i in range(len(path)-1):
                delta.append(G.edges[(path[i],path[i+1])]['capacity']-G.edges[(path[i],path[i+1])]['flow'])
            flow = min(delta)
            if flow == 0:
                break
            for i in range(len(path)-1):
                G.edges[(path[i],path[i+1])]['flow'] += flow
            print("Augmenting Path: ", end=" ")
            for i in range(len(path)-1):
                print(path[i]+" ->", end =" ")
            print(path[-1])
            print("Flow Value: ", flow)
    
    maxflow = 0
    for n in nx.neighbors(G, u):
        maxflow += G.edges[(u,n)]['flow']
    return maxflow


def edmonds_karp(G, u, v):
    for e in G.edges():
        G.edges[e]['flow'] = 0

    flow = 1
    niters = 1
    while flow != 0:
        print("Iteration: {}".format(niters))
        G = bfs(G, u, v)

        path = ['t']
        while path[-1] != 's':
            path.append(G.nodes[path[-1]]['predecessor'])
        path = path[::-1]

        delta = []
        for i in range(len(path)-1):
            delta.append(G.edges[(path[i],path[i+1])]['capacity']-G.edges[(path[i],path[i+1])]['flow'])
        flow = min(delta)
        if flow == 0:
            break
        for i in range(len(path)-1):
            G.edges[(path[i],path[i+1])]['flow'] += flow
        print("Augmenting Path: ", end=" ")
        for i in range(len(path)-1):
            print(path[i]+" ->", end =" ")
        print(path[-1])
        print("Flow Value: ", flow)
        print("\n")
    
    maxflow = 0
    for n in nx.neighbors(G, 's'):
        maxflow += G.edges[('s',n)]['flow']
    return maxflow

G = nx.DiGraph()
G.add_edge('0','1', capacity=50)
G.add_edge('0','2', capacity=40)
G.add_edge('0','4', capacity=30)

G.add_edge('1','3', capacity=9)
G.add_edge('1','4', capacity=19)

G.add_edge('2','4', capacity=12)
G.add_edge('2','5', capacity = 34)

G.add_edge('3','6', capacity=12)
G.add_edge('3','4', capacity=15)

G.add_edge('4','6', capacity=77)
G.add_edge('4','7', capacity=32)
G.add_edge('4','8', capacity=12)
G.add_edge('4','5', capacity=5)

G.add_edge('5','8', capacity=18)

G.add_edge('6','4', capacity=17)
G.add_edge('6','7', capacity=33)
G.add_edge('6','9', capacity=20)

G.add_edge('7','9', capacity=40)
G.add_edge('7','8', capacity=26)

G.add_edge('8','7', capacity=12)
G.add_edge('8','9', capacity=60)

G.nodes['0']['pos'] = (-1,0)
G.nodes['1']['pos'] = (1,1)
G.nodes['2']['pos'] = (1,-1)
G.nodes['3']['pos'] = (3,1)
G.nodes['4']['pos'] = (3,0)
G.nodes['5']['pos'] = (3,-1)
G.nodes['6']['pos'] = (5,1)
G.nodes['7']['pos'] = (5,0)
G.nodes['8']['pos'] = (5,-1)
G.nodes['9']['pos'] = (7,0)

import matplotlib.pyplot as plt
fig = plt.figure(figsize=(10,5))
node_pos = nx.get_node_attributes(G,'pos')
capacities = nx.get_edge_attributes(G,'capacity')
nx.draw_networkx(G, node_pos, node_color='pink', node_size=800)
nx.draw_networkx_edges(G, node_pos,edge_color= 'black')
text = nx.draw_networkx_edge_labels(G, node_pos, edge_color= 'black', edge_labels=capacities, font_size=12)
for _,t in text.items():
    t.set_rotation('horizontal')
ax = plt.gca() # to get the current axis
ax.collections[0].set_edgecolor("#000000")
ax.collections[0].set_linewidth(2)
plt.axis('off')
plt.show()

edmonds_karp(G, '0', '9')
ford_fulkerson(G,'0','9')