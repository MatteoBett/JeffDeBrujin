from collections import defaultdict
from typing import List, Dict
import torch
import numpy as np

from bbuilder import utils
import psutil
import time

def _build_de_bruijn_graph(kmers_list : List[torch.Tensor], 
                           k : int,
                           bins : int = 1,
                           T : np.matrix = None,
                           startvec : np.ndarray = None):
    graph = defaultdict(list)
    mask = (1 << (2*(k-1))) - 1

    if T is not None:
        for i, kmers in enumerate(kmers_list):  
            utils.progressbar(iteration=i+1, total=len(kmers_list), prefix="Building Brujin graph")
            
            for j, kmer in enumerate(kmers):
                idx = round((j / len(kmers)) * (bins - 1))
                prefix = kmer.item() >> 2
                suffix = kmer.item() & mask

                T[prefix, suffix, idx] += 1
                graph[prefix].append(suffix)

                if j == 0:
                    startvec[prefix] += 1
        return graph, T, startvec
    
    else:
        e = 4**(k-1) + 1
        s = 4**(k-1) + 2
        for i, kmers in enumerate(kmers_list):  
            utils.progressbar(iteration=i+1, total=len(kmers_list), prefix="Building Brujin graph")

            for j, kmer in enumerate(kmers):
                prefix = kmer.item() >> 2
                suffix = kmer.item() & mask
                if suffix not in graph[prefix]:
                    graph[prefix].append(suffix)

                if j == 0:
                    graph[s].append(prefix)
                if j == len(kmers) - 1:
                    graph[suffix].append(e)       
        
        return graph


def seq_Tmat(graph : Dict[int, List[int]]):
    """
    Sequentially build a transition matrix between kmers
    """
    reindex = dict(zip(graph.keys(), range(len(graph))))
    T = np.zeros((len(reindex), len(reindex)), dtype=np.int16)

    for i, (prefix, suffixes) in enumerate(graph.items()):
        utils.progressbar(i+1, total=len(graph))
        idx_prefix = reindex[prefix]

        for suffix in suffixes:
            idx_suffix = reindex[suffix]
            T[idx_prefix, idx_suffix] += 1
    return T


def build_de_bruijn_graph(kmers_list : List[torch.Tensor], k : int, compactable: bool = False, bins : int = 1) -> Dict[int, List[int]]:
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

    if (4**(k-1) * 4**(k-1) * bins * 4 ) > psutil.virtual_memory().total:
        print("Transition matrix too large to fit in Memory. Building it sequentially.")
        graph = _build_de_bruijn_graph(kmers_list=kmers_list, k=k, bins=bins, T=None, startvec=None)
    else:
        print("On-the-fly transition matrix building.")
        T = np.zeros((4**(k-1), 4**(k-1), bins), dtype=np.int16)
        startvec = np.zeros((4**(k-1),), dtype=np.int16)
        graph, T, startvec = _build_de_bruijn_graph(kmers_list=kmers_list, k=k, bins=bins, T=T, startvec=startvec)
    
    print(list(graph.values())[0])
    print("\nDe Bruijn graph built with", len(graph), "nodes\n")

    if compactable:
        graph = compact_chocolate(G=graph, k=k-1)

        print("Compacted De Bruijn graph built with", len(graph), "nodes.")
        if (len(graph) * len(graph) * bins * 2) > psutil.virtual_memory().total:
            raise MemoryError(f"{len(graph) * len(graph) * bins * 2} bytes requested for transition matrix, exceed installed RAM.")
        else:
            print(f"Compation allows Transition matrix building. Beginning...\n")
            seq_Tmat(graph=graph)
        return graph, None, None
    
    else:
        print("De Bruijn graph cannot be compacted. Starting MCMC sampling...")
        sumT = np.sum(T, axis=1, keepdims=True)
        normT = np.divide(T, sumT, out=np.zeros_like(T, dtype=np.float32), where=sumT != 0)
        T = torch.tensor(normT, dtype=torch.float32)
        s = torch.tensor(startvec, dtype=torch.int32)
        startvec = s / s.sum()

        return graph, T, startvec


def compact_chocolate(G: Dict[int, List[int]], k : int) -> Dict[int, List[int]]:
    """
    Compact the De Bruijn graph by merging nodes that have only one outgoing edge.
    """
    new_graph = {}
    visited = set()
    s = 4**k + 2

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

        prefix_A = ((0b00 << path[0].bit_length()) | path[0]) >> 2
        prefix_C = ((0b01 << path[0].bit_length()) | path[0]) >> 2
        prefix_T = ((0b10 << path[0].bit_length()) | path[0]) >> 2
        prefix_G = ((0b11 << path[0].bit_length()) | path[0]) >> 2
        prefixes = set([prefix_A, prefix_C, prefix_T, prefix_G, s])

        prefix_visited = visited & prefixes
        prefix_tovisit = prefixes - prefix_visited

        for prefix in prefix_tovisit:
            if prefix in G:
                if path[0] in G[prefix]:
                    G[prefix].remove(path[0])
                    G[prefix].append(compacted)

        for prefix in prefix_visited:
            if prefix in new_graph:
                if path[0] in new_graph[prefix]:
                    new_graph[prefix].remove(path[0])
                    new_graph[prefix].append(compacted)

        for n in path:
            visited.add(n)

    # Count all suffixes and print their number
    all_suffixes = set()
    for suffixes in new_graph.values():
        all_suffixes.update(suffixes)
    print(f"Number of unique suffixes: {len(all_suffixes)}")
    return new_graph