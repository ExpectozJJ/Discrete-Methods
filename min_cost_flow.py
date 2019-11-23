import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx 
from networkx.algorithms.flow import network_simplex, min_cost_flow

# Created by Wee JunJie 
# Dated 17th Nov 2019

import numpy as np 
import matplotlib.pyplot as plt
import networkx as nx 

def pot_reduc_cost(G,L,U):
    # G = Network, L = Lower Capacity edges, U = Upper Capacity edges
    # Given a Tree Structure with feasible flow, compute all the potentials and reduced costs and the reduced costs that violates the optimality conditions 
    # Set Potential of 1st vertex as 0 

    potentials = {'1': 0}
    reduced_costs = dict()
    T = set(G.edges()) - L - U

    for e in T:
        reduced_costs[(e[0],e[1])] = 0
        if e[0] == '1':
            potentials[e[1]] = G[e[0]][e[1]]['weight']
        elif e[1] == '1':
            potentials[e[0]] = -G[e[0]][e[1]]['weight']

    while len(potentials) < G.number_of_nodes():
        for e in T:
            if e[0] not in potentials.keys() and e[1] in potentials.keys():
                potentials[e[0]] = potentials[e[1]] - G[e[0]][e[1]]['weight']
            if e[1] not in potentials.keys() and e[0] in potentials.keys():
                potentials[e[1]] = potentials[e[0]] + G[e[0]][e[1]]['weight']
            if len(potentials) == G.number_of_nodes():
                break
    
    print("T: ", T)
    print("L: ", L)
    print("U: ", U)
    print("Potentials: ", potentials)

    violated = set()
    for e in L.union(U):
        reduced_costs[e] = G[e[0]][e[1]]['weight'] + potentials[e[0]] - potentials[e[1]]
        if (e in L and reduced_costs[e] < 0) or (e in U and reduced_costs[e] > 0):
            #print(e, " violates optimality conditions.")
            violated.add(e)

    print("Reduced Costs: ", reduced_costs)
    print("Violated Edges: ", violated)
    
    return potentials, reduced_costs, violated

def augment_flow(G,L,U,e):
    # T = Spanning Tree of G
    # e = edge chosen that violated optimality conditionss
    
    print("Chosen Violated Edge: ", e)
    
    T = set(G.edges()) - L - U
    H = nx.Graph.edge_subgraph(G, list(T)+[e])
    C = nx.find_cycle(H, e, orientation='ignore')
    if (e in U and (e[0], e[1], 'forward') in C) or (e in L and (e[0], e[1], 'reverse') in C):
        for i in range(len(C)):
            if C[i][2] == 'reverse': 
                C[i] = (C[i][0], C[i][1], 'forward')
            else:
                C[i] = (C[i][0], C[i][1], 'reverse')
    print("Cycle: ", C)
                
    delta = []
    for edge in C:
        if edge[2]=='reverse':
            delta.append(G[edge[0]][edge[1]]['init_flow']-G[edge[0]][edge[1]]['low_cap']) 
        else:
            delta.append(G[edge[0]][edge[1]]['capacity']-G[edge[0]][edge[1]]['init_flow'])

    chosen = []
    if np.min(delta) == 0:
        for i in range(len(C)):
            if (C[i][0], C[i][1]) != e and delta[i] == 0:
                chosen.append(C[i])
    else:
        loc = np.where(np.array(delta)==np.min(delta))
        loc = loc[0]
        for l in loc:
            chosen.append(C[l])
    
    #print("Set of edges where minimum is attained: ", chosen)
    """
    if len(chosen) >= 2:
        num = int(input("Enter your (last blocking) edge option (Index 1 to "+str(len(chosen))+"): "))
        while (num > len(chosen) or num <= 0):
            print("Invalid Index!")
            num = int(input("Enter your (last blocking) edge option (Index 1 to "+str(len(chosen))+"): "))
    else:
        num = 1
    """
    
    if (e[0], e[1], 'forward') in C:
        chosen = chosen[-1]
    else:
        chosen = chosen[0]
    delta = np.min(delta)
    print("delta: ", delta, "\nEdge where minimum delta is attained: ", chosen, "\n")
    
    if delta > 0:
        for edge in C:
            if edge[2]=='reverse':
                G[edge[0]][edge[1]]['init_flow'] -= delta
            else:
                G[edge[0]][edge[1]]['init_flow'] += delta
                
    try:
        L.remove(e)
    except:
        U.remove(e)
    
    if G[chosen[0]][chosen[1]]['init_flow'] == G[chosen[0]][chosen[1]]['capacity']:
        U.add((chosen[0], chosen[1]))
    else:
        L.add((chosen[0], chosen[1]))
        
    #print(L, U)
    
    return G, L, U

def plot_network(G, L, U, potentials, niters):
    colors = []
    for e in G.edges:
        if (e[0], e[1]) in L:
            colors.append('red')
        elif (e[0], e[1]) in U:
            colors.append('green')
        else:
            colors.append('blue')
    fig = plt.figure(figsize=(10,5))
    node_pos = nx.get_node_attributes(G,'pos')
    label_pos = nx.get_node_attributes(G,'label_pos')
    demands = nx.get_node_attributes(G,'demand')
    costs = nx.get_edge_attributes(G,'weight')
    low_cap = nx.get_edge_attributes(G, 'low_cap')
    starts = nx.get_edge_attributes(G,'init_flow')
    capacities = nx.get_edge_attributes(G,'capacity')
    labels = dict()
    for e in G.edges(): 
        if low_cap[(e[0],e[1])] != 0:
            labels[e] = '{}/{}/{}/{}'.format(low_cap[(e[0],e[1])],starts[(e[0],e[1])],capacities[(e[0], e[1])], costs[(e[0], e[1])])
        else:
            labels[e] = '{}/{}/{}'.format(starts[(e[0],e[1])],capacities[(e[0], e[1])], costs[(e[0], e[1])])
    nx.draw_networkx(G, node_pos,node_color='white', edge_color=colors, node_size=800, arrowsize=20)
    nx.draw_networkx_edges(G, node_pos,edge_color= colors, width=2)
    text = nx.draw_networkx_edge_labels(G, node_pos, edge_labels=labels, font_size=14)
    labels = {v: '{}/{}'.format(potentials[v],demands[v])
    for v in G.nodes}
    nx.draw_networkx_labels(G, label_pos, labels=labels, font_size=14)
    for _,t in text.items():
        t.set_rotation('horizontal')
    ax = plt.gca() # to get the current axis
    ax.collections[0].set_edgecolor("#000000")
    ax.collections[0].set_linewidth(2)
    plt.axis('off')
    plt.savefig("MAS711_Misc_Ex_6_"+str(niters)+"_.eps", format='eps', dpi=500,bbox_inches='tight')
    plt.show()
    
def compute_cost_flow(G):
    sum = 0
    for e in G.edges():
        sum += G[e[0]][e[1]]['init_flow']*G[e[0]][e[1]]['weight']
    print("Cost Flow: ", sum)
    return sum
    

# MAS711 AY1617 Qn 4
G = nx.DiGraph()

# Demand Conditions 
G.add_node('1', demand = -2)
G.add_node('2', demand = -2)
G.add_node('4', demand = -1)
G.add_node('3', demand = 0)
G.add_node('5', demand = 5)

# Flow Capacities and Costs
# Weights are the gamma(e) = costs 
G.add_edge('1', '2', weight = 11, capacity = 5, init_flow = 0, low_cap = 0)
G.add_edge('1', '4', weight = 10, capacity = 3, init_flow = 2, low_cap = 0)
G.add_edge('3', '1', weight = -25, capacity = 5, init_flow = 0, low_cap = 0)
G.add_edge('2', '4', weight = 11, capacity = 5, init_flow = 2, low_cap = 0)
G.add_edge('2', '3', weight = 16, capacity = 4, init_flow = 0, low_cap = 0)
G.add_edge('4', '3', weight = 12, capacity = 3, init_flow = 3, low_cap = 0)
G.add_edge('4', '5', weight = 18, capacity = 4, init_flow = 2, low_cap = 0)
G.add_edge('3', '5', weight = 13, capacity = 3, init_flow = 3, low_cap = 0)

# Change the nodes plotting coordinates in matplotlib
G.nodes['1']['pos'] = (1,0)
G.nodes['2']['pos'] = (2,0)
G.nodes['3']['pos'] = (3,-1)
G.nodes['4']['pos'] = (3,1)
G.nodes['5']['pos'] = (4,0)

# Change the node labelling positions
G.nodes['1']['label_pos'] = (0.9,0.2)
G.nodes['2']['label_pos'] = (1.9,0.2)
G.nodes['3']['label_pos'] = (3.2,-1.1)
G.nodes['4']['label_pos'] = (2.8,1.1)
G.nodes['5']['label_pos'] = (4.2,0.1)

# Change your Initial Tree Structure
L = {('1', '2'), ('2', '3'), ('3', '1')}
U = {('3', '5')}

niters = 0
potentials, reduced_costs, violated = pot_reduc_cost(G,L,U)
plot_network(G, L, U, potentials, niters)
while violated != set():
    niters += 1
    if len(violated) >= 2:
        num = int(input("Enter your violated edge option (Index 1 to "+str(len(violated))+"): "))
        while (num > len(violated) or num <= 0):
            print("Invalid Index!")
            num = int(input("Enter your violated edge option (Index 1 to "+str(len(violated))+"): "))
    else:
        num = 1
    G, L, U = augment_flow(G, L, U, list(violated)[int(num)-1])
    compute_cost_flow(G)
    potentials, reduced_costs, violated = pot_reduc_cost(G,L,U)
    plot_network(G, L, U, potentials, niters)
print("All edges satisfy the optimality conditions.")
print("Number of Iterations: ", niters)

# MAS711 AY1819 Qn 3
G = nx.DiGraph()

G.add_node('1', demand = -2)
G.add_node('2', demand = -5)
G.add_node('4', demand = 0)
G.add_node('3', demand = 2)
G.add_node('5', demand = 0)
G.add_node('6', demand = 5)

# Flow Capacities and Costs
# Weights are the gamma(e) = costs 
G.add_edge('1', '3', weight = 5, capacity = 2, init_flow = 2, low_cap = 0)
G.add_edge('1', '4', weight = 1, capacity = 5, init_flow = 3, low_cap = 0)
G.add_edge('2', '1', weight = 2, capacity = 3, init_flow = 3, low_cap = 0)
G.add_edge('2', '4', weight = 10, capacity = 4, init_flow = 2, low_cap = 0)
G.add_edge('3', '5', weight = 2, capacity = 3, init_flow = 0, low_cap = 0)
G.add_edge('4', '6', weight = 3, capacity = 8, init_flow = 5, low_cap = 0)
G.add_edge('4', '5', weight = 2, capacity = 1, init_flow = 0, low_cap = 0)
G.add_edge('5', '6', weight = 4, capacity = 7, init_flow = 0, low_cap = 0)

# Change the nodes plotting coordinates in matplotlib
G.nodes['1']['pos'] = (1,5)
G.nodes['2']['pos'] = (1.1,5)
G.nodes['3']['pos'] = (1,4)
G.nodes['4']['pos'] = (1.1,4)
G.nodes['5']['pos'] = (1,3)
G.nodes['6']['pos'] = (1.1,3)

# Change the node labelling positions
G.nodes['1']['label_pos'] = (0.99,5)
G.nodes['2']['label_pos'] = (1.11,5)
G.nodes['3']['label_pos'] = (0.99,4)
G.nodes['4']['label_pos'] = (1.11,4)
G.nodes['5']['label_pos'] = (0.99,3)
G.nodes['6']['label_pos'] = (1.11,3)

# Change your Initial Tree Structure
L = {('5', '6')}
U = {('2', '1'), ('1', '3')}

niters = 0
potentials, reduced_costs, violated = pot_reduc_cost(G,L,U)
plot_network(G, L, U, potentials, niters)
while violated != set():
    niters += 1
    if len(violated) >= 2:
        num = int(input("Enter your violated edge option (Index 1 to "+str(len(violated))+"): "))
        while (num > len(violated) or num <= 0):
            print("Invalid Index!")
            num = int(input("Enter your violated edge option (Index 1 to "+str(len(violated))+"): "))
    else:
        num = 1
    G, L, U = augment_flow(G, L, U, list(violated)[int(num)-1])
    compute_cost_flow(G)
    potentials, reduced_costs, violated = pot_reduc_cost(G,L,U)
    plot_network(G, L, U, potentials, niters)
print("All edges satisfy the optimality conditions.")
print("Number of Iterations: ", niters)

# MAS711 Miscellaneous Exercises Qn 6
G = nx.DiGraph()

G.add_node('1', demand = -5)
G.add_node('2', demand = -13)
G.add_node('4', demand = 11)
G.add_node('3', demand = 2)
G.add_node('5', demand = -4)
G.add_node('6', demand = 9)

# Flow Capacities and Costs
# Weights are the gamma(e) = costs 
G.add_edge('1', '3', weight = -5, capacity = 10, init_flow = 5, low_cap = 0)
G.add_edge('1', '2', weight = 1, capacity = 8, init_flow = 0, low_cap = 0)
G.add_edge('4', '1', weight = 1, capacity = 7, init_flow = 0, low_cap = 0)
G.add_edge('5', '1', weight = 1, capacity = 5, init_flow = 0, low_cap = 0)
G.add_edge('2', '4', weight = -1, capacity = 8, init_flow = 4, low_cap = 0)
G.add_edge('2', '6', weight = 1, capacity = 9, init_flow = 9, low_cap = 0)
G.add_edge('3', '4', weight = 2, capacity = 5, init_flow = 0, low_cap = 0)
G.add_edge('3', '6', weight = 4, capacity = 5, init_flow = 3, low_cap = 3)
G.add_edge('5', '4', weight = -1, capacity = 4, init_flow = 4, low_cap = 0)
G.add_edge('6', '4', weight = -1, capacity = 8, init_flow = 3, low_cap = 3)

# Change the nodes plotting coordinates in matplotlib
G.nodes['1']['pos'] = (0.25,0)
G.nodes['2']['pos'] = (0.5,0.5)
G.nodes['3']['pos'] = (0.6,2)
G.nodes['4']['pos'] = (0.65,-0.7)
G.nodes['5']['pos'] = (0.55,-2)
G.nodes['6']['pos'] = (0.9,0)

# Change the node labelling positions
G.nodes['1']['label_pos'] = (0.24,0.5)
G.nodes['2']['label_pos'] = (0.5,0.9)
G.nodes['3']['label_pos'] = (0.63,2.3)
G.nodes['4']['label_pos'] = (0.68,-1.1)
G.nodes['5']['label_pos'] = (0.6,-2.3)
G.nodes['6']['label_pos'] = (0.95,0)


# Change your Initial Tree Structure
L = {('1', '2'), ('3', '6'), ('3', '4')}
U = {('5', '4'), ('2', '6')}

niters = 0
potentials, reduced_costs, violated = pot_reduc_cost(G,L,U)
plot_network(G, L, U, potentials, niters)
while violated != set():
    niters += 1
    if len(violated) >= 2:
        num = int(input("Enter your violated edge option (Index 1 to "+str(len(violated))+"): "))
        while (num > len(violated) or num <= 0):
            print("Invalid Index!")
            num = int(input("Enter your violated edge option (Index 1 to "+str(len(violated))+"): "))
    else:
        num = 1
    G, L, U = augment_flow(G, L, U, list(violated)[int(num)-1])
    compute_cost_flow(G)
    potentials, reduced_costs, violated = pot_reduc_cost(G,L,U)
    plot_network(G, L, U, potentials, niters)
print("All edges satisfy the optimality conditions.")
print("Number of Iterations: ", niters)