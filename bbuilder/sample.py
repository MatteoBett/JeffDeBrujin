import os
from typing import List

import torch
import torch.nn.functional as F

from bbuilder import utils

def metropolis_hastings_sampling(Tb: torch.Tensor, P : torch.Tensor, start_idx: int, end_idx: int, num_samples: int) -> List[int]:
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

    a = 1
    t = 0
    n = 0
    while True:  
        """  
        accepted = torch.zeros((num_samples, ), dtype=torch.long)
        rejected = torch.ones((num_samples, ), dtype=torch.long)
        next_nodes = accepted.clone()

        while accepted.sum() < num_samples:
            next_nodes += Tb[current, :-1].multinomial(1).squeeze()*rejected
            
            a = torch.clamp((P[next_nodes]*Tb[current, next_nodes])/(P[current]*Tb[next_nodes, current]), min=0, max=1)
            t = torch.rand((num_samples, ))

            accepted[a > t] = 1
            rejected[a > t] = 0
            next_nodes = next_nodes*accepted"""

        next_nodes = Tb[current, :-1].multinomial(1).squeeze()
        current = next_nodes
        res = torch.vstack((res, current))
        nonend_nodes = next_nodes[next_nodes != end_idx]
        
        utils.progressbar(iteration=num_samples-len(nonend_nodes), total=num_samples, prefix="Sampling", suffix="samples generated")
        n+=1
        if len(nonend_nodes) == 0:
            print(res.shape[0], "samples generated, ending sampling.")

            return res


