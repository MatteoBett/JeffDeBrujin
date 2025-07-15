import torch

def kmer_size():
    """
    Computes the optimal kmer size for a given probability of occurence.
    """


def encode_nucl(nucl : str):
    """ 
    Encode a nucleotide into a 2-bit integer.
    Also encode its reverse complement
    """
    encoded = (ord(nucl) >> 1) & 0b11 # Extract the two bits of the ascii code that represent the nucleotide

    return encoded


def stream_kmers(seq : str, k : int):
    """
    Provide a stream of the kmers for a given sequence.
        - first loop: Add the first k-1 nucleotides to the first kmer and its reverse complement
        - yield loop: Sliding window using bit-shift to encode the entire sequence
    """
    singleton_encoder = encode_nucl

    kmer = 0

    for i in range(k-1):
        nucl = singleton_encoder(seq[i])
        kmer |= nucl << (2*(k-2-i))

    mask = (1 << (2*(k-1))) - 1
    for i in range(k-1, len(seq)):
        nucl = singleton_encoder(seq[i])
        kmer &= mask # Shift the kmer to make space for the new nucleotide
        kmer <<= 2 # Add the new nucleotide to the kmer
        kmer |= nucl # remove the rightmost nucleotide by side effect

        yield kmer


def main_kmers(seq : str, k : int, device : str = "cpu"):
    return torch.tensor([kmer for kmer in stream_kmers(seq=seq, k=k)], dtype=torch.int64, device=device)

