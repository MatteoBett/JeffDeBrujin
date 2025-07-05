import os
from typing import List

import torch

from bbuilder import utils

def metropolis_hastings_sampling(Tb: torch.Tensor, start_idx: int, end_idx: int, num_samples: int) -> List[int]:
    """
    Perform Metropolis-Hastings sampling on the De Bruijn graph.
    
    Args:
        Tb: Transition matrix of the De Bruijn graph.
        start_idx: Index of the start node in the transition matrix.
        end_idx: Index of the end node in the transition matrix.
        num_samples: Number of samples to generate.

    Returns:
        A list of sampled nodes from the De Bruijn graph.
    """

    res = torch.multinomial(Tb[start_idx, :-2], num_samples, replacement=True)
    current = res.clone()
    
    while True:
        next_nodes = Tb[current, :-1].multinomial(1).squeeze()
        res = torch.vstack((res, next_nodes))
        current = next_nodes
        nonend_nodes = next_nodes[next_nodes != end_idx]

        utils.progressbar(iteration=num_samples-len(nonend_nodes), total=num_samples, prefix="Sampling", suffix="samples generated")
        if len(nonend_nodes) == 0:
            print(res.shape[1], "samples generated, ending sampling.")
   
            return res