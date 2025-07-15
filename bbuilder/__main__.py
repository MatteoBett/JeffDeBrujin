import os, re

from bbuilder import loader, streamkmers
from bbuilder import debrujin
from bbuilder import sample
from bbuilder import utils
from bbuilder import viz


"""
Tb is the transition matrix for the De Bruijn graph, where Tb[i, j] is the probability of transitioning from node i to node j.
Fk is the frequency matrix for the kmers where Fk[i,j] if the frequency of kmers i and j appearing together in the same sequence.
start is the vector indicating the start probabilities for each node in the De Bruijn graph.
end is the vector indicating the end probabilities for each node in the De Bruijn graph.
graph is the De Bruijn graph itself, represented as a dictionary where the keys are the nodes and the values are the list of adjacent nodes.
"""

def main(pathseq : str, 
         output_path : str,
         k : int, 
         num_samples : int,
         device : str
         ) -> None:
    
    kdivs = {}
    for k in range(5, 25):
        kmers_list = [streamkmers.main_kmers(seq=seq, k=k, device=device) for seq in loader.loadfasta(path=pathseq)]
        kdivs[k] = viz.kmers_diversity(kmers_list=kmers_list)
    
    viz.kdiv_dist(kdivs)

    print("Loaded", len(kmers_list), "sequences from", pathseq, "with a total of", sum([len(kmers) for kmers in kmers_list]), "kmers.")
    graph = debrujin.build_de_bruijn_graph(kmers_list=kmers_list, k=k)

    


if __name__ == "__main__":
    pathseq = r'data/GII/GII.fasta'
    output_path = r'out/sequences/seqs.fasta'
    k = 20
    num_samples = 10
    device = "cpu"
    main(pathseq=pathseq, output_path=output_path, k=k, num_samples=num_samples, device=device)