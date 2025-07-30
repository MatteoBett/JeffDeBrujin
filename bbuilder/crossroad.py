import torch
from typing import List, Dict

def csr_compact(kmers_list: List[torch.Tensor], k: int) -> bool:
    """
    Determine if the De Bruijn graph can be compacted or if all nodes 
    have multiple outgoing edges.
    """

    m = 4**k
    nkmer = torch.unique(torch.cat(kmers_list, dim=0), dim=0)
    if len(nkmer) != m:
        return True
    else:
        print(f"De Bruijn graph cannot be compacted, {len(nkmer)} unique kmers found, expected {m}.")
        return False