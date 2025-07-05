import matplotlib.pyplot as plt
from Bio import SeqIO

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