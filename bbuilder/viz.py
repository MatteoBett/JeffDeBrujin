import matplotlib.pyplot as plt
from Bio import SeqIO
from typing import List, Dict
import torch
import numpy as np

def seqsize_distribution(sequences, natseqs : str):
    """
    Plot the distribution of sequence sizes.
    
    Args:
        sequences (list): List of sequences.
    """
    sizes = [len(seq) for seq in sequences]
    natsizes = [len(seq) for seq in SeqIO.parse(natseqs, "fasta")]

    plt.hist(sizes, bins=30, alpha=0.7, color='blue', label='Generated Sequences')
    plt.hist(natsizes, bins=30, alpha=0.7, color='red', label='Natural Sequences')
    plt.title('Distribution of Sequence Sizes')
    plt.xlabel('Size of Sequence')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.savefig(r'out/figures/seqsize_distribution.png')

def kmers_diversity(kmers_list : List[torch.Tensor],  bins : int = 100):
    dico = {i: [] for i in range(0, bins+1)}

    for kmers in kmers_list:
        seqsize = len(kmers)
        for i, kmer in enumerate(kmers):
            _bin = round((i/seqsize)*bins)
            dico[_bin].append(kmer.item())

    kdiv = [len(set(dico[i]))/len(dico[i]) for i in range(0, bins+1)]

    return kdiv

def kdiv_dist(kdivs : Dict[int, List]):
    plt.figure(figsize=(8, 6))
    k_values = sorted(kdivs.keys())
    bins = len(next(iter(kdivs.values())))
    data = np.array([kdivs[k] for k in k_values])

    im = plt.imshow(data.T, aspect='auto', origin='lower', cmap='jet')
    plt.colorbar(im, label='Diversity')
    plt.xticks(ticks=range(len(k_values)), labels=k_values)
    plt.yticks(ticks=range(0, bins, 5), labels=range(1, bins + 1, 5))
    plt.xlabel('k value')
    plt.ylabel('Bin number')
    plt.title('K-mer Diversity Heatmap')
    plt.tight_layout()
    plt.savefig('out/figures/kdiv_heatmap.png')
    plt.close()
