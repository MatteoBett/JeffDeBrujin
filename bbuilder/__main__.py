import os, re

from bbuilder import loader, streamkmers
from bbuilder import debrujin


"""
Tb is the transition matrix for the De Bruijn graph, where Tb[i, j] is the probability of transitioning from node i to node j.
Fk is the frequency matrix for the kmers where Fk[i,j] if the frequency of kmers i and j appearing together in the same sequence.
start is the vector indicating the start probabilities for each node in the De Bruijn graph.
end is the vector indicating the end probabilities for each node in the De Bruijn graph.
graph is the De Bruijn graph itself, represented as a dictionary where the keys are the nodes and the values are the list of adjacent nodes.
"""


def main(pathseq : str, 
         k : int, 
         device : str
         ) -> None:
    
    kmers_list = [streamkmers.main_kmers(seq=seq, k=k, device=device) for seq in loader.loadfasta(path=pathseq)]
    print("Loaded", len(kmers_list), "sequences from", pathseq, "with a total of", sum([len(kmers) for kmers in kmers_list]), "kmers.")
    Tb, Fk, start, end, graph = debrujin.build_de_bruijn_graph(kmers_list=kmers_list, k=k)
    print("Compacted De Bruijn graph built with", len(graph), "nodes.")


if __name__ == "__main__":
    pathseq = r'data/GII/GII.fasta'
    k = 7
    device = "cpu"
    main(pathseq=pathseq, k=k, device=device)