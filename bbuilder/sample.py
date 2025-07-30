import os
from typing import List

import torch
import torch.nn.functional as F

from bbuilder import utils

def MCMC_sampling(T: torch.Tensor, start_idx: torch.Tensor, endvec : torch.Tensor, num_samples: int, k :int) -> List[int]:
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

    end_idx = 4**k + 1
    res = start_idx.multinomial(num_samples=num_samples, replacement=True).squeeze().to(torch.int32)
    current = res.clone().detach()
    end_dist = endvec.multinomial(num_samples=num_samples, replacement=True).squeeze().to(torch.int32)
    nonend_nodes = next_nodes[next_nodes != end_idx]

    n = 0
    while True: 
        next_probs = T.index_select(0, current)[:, :, 0]
        next_nodes = next_probs.multinomial(1).squeeze()
        res = torch.vstack((res, current))
        nonend_nodes = next_nodes[next_nodes != end_idx]
        current = next_nodes
        
        n+=1
        utils.progressbar(iteration=num_samples-len(nonend_nodes), total=num_samples, prefix="Sampling", suffix="samples generated")
        if len(nonend_nodes) == 0:
            print(res.shape[0], "samples generated, ending sampling.")

            return res


