import re, os

from Bio import SeqIO

def loadfasta(path : str):
    return [str(record.seq) for record in SeqIO.parse(path, format="fasta-pearson")]
        