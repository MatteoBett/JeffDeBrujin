from collections import defaultdict
from typing import List, Dict
import torch
import numpy as np

from bbuilder import utils
import subprocess

def build_de_bruijn_graph(kmers_list : List[torch.Tensor], k : int):
    """
    Input:
        kmers_list is a list of tensors, where each tensor contains the kmers for a sequence.
        k is the length of the kmers.
    Output:
        Tb is the transition matrix for the De Bruijn graph, where Tb[i, j] is the probability of transitioning from node i to node j.
        Fk is the frequency matrix for the kmers where Fk[i,j] if the frequency of kmers i and j appearing together in the same sequence.
        start is the vector indicating the start probabilities for each node in the De Bruijn graph.
        end is the vector indicating the end probabilities for each node in the De Bruijn graph.
        graph is the De Bruijn graph itself, represented as a dictionary where the keys are the nodes and the values are the list of adjacent nodes.
    """
    graph = defaultdict(list)
    mask = (1 << (2*(k-1))) - 1
    for i, kmers in enumerate(kmers_list):
        utils.progressbar(iteration=i+1, total=len(kmers_list), prefix="Building Brujin graph")
        
        for _, kmer in enumerate(kmers):
            prefix = kmer.item() >> 2
            suffix = kmer.item() & mask
            
            graph[prefix].append(suffix)

    print("De Bruijn graph built with", len(graph), "nodes")

    graph = compact_chocolate(G=graph)

    print("Compacted De Bruijn graph built with", len(graph), "nodes.")

    return graph


def compact_chocolate(G: Dict[int, List[int]]) -> Dict[int, List[int]]:
    """
    Compact the De Bruijn graph by merging nodes that have only one outgoing edge.
    Not currently in use
    """
    new_graph = {}
    visited = set()

    for i, node in enumerate(list(G.keys())):
        utils.progressbar(iteration=i+1, total=len(G), prefix="Compacting Brujin graph")
        if node in visited:
            continue
        path = [node]
        current = node
        while current in G and len(G[current]) == 1:
            next_node = G[current][0]
            if next_node in G and len(G[next_node]) == 1 and next_node not in visited:
                path.append(next_node)
                visited.add(current)
                current = next_node
            else:
                break

        compacted = path[0]
        for nxt in path[1:]:
            compacted = (compacted << 2) | (nxt & 0b11)

        end = G[path[-1]] if path[-1] in G else []
        new_graph[compacted] = end
        for n in path:
            visited.add(n)
    return new_graph