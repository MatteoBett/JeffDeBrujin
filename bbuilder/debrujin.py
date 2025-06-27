from collections import defaultdict
from typing import List, Dict
import torch

from bbuilder import utils
import subprocess

def build_de_bruijn_graph(kmers_list : List[torch.Tensor], k : int):
    graph = defaultdict(list)
    mask = (1 << (2*(k-2))) - (1 << (2*(k-3)))
    for i, kmers in enumerate(kmers_list):
        utils.progressbar(iteration=i+1, total=len(kmers_list), prefix="Building Brujin graph")
        for kmer in kmers:
            prefix = kmer >> 2
            suffix = kmer & ~mask
            graph[prefix].append(suffix)
    return graph


def brujin_cuttlefish(pathfasta : str, k : int, output : str):
    pass
