import os, re

from bbuilder import loader, streamkmers
from bbuilder import debrujin
from bbuilder import sample
from bbuilder import utils
from bbuilder import viz
from bbuilder import crossroad

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
         device : str,
         bins : int
         ) -> None:
    
    kdivs = {}
    for _k in range(5, 25):
        kmers_list = [streamkmers.main_kmers(seq=seq, k=_k, device=device) for seq in loader.loadfasta(path=pathseq)]
        kdivs[_k] = viz.kmers_diversity(kmers_list=kmers_list, bins=bins)
    
    kmers_list = [streamkmers.main_kmers(seq=seq, k=k, device=device) for seq in loader.loadfasta(path=pathseq)]
    kmers_list_min = [streamkmers.main_kmers(seq=seq, k=k-1, device=device) for seq in loader.loadfasta(path=pathseq)]
    viz.kdiv_dist(kdivs)

    compactable = crossroad.csr_compact(kmers_list=kmers_list_min, k=k-1)
    print("Loaded", len(kmers_list), "sequences from", pathseq, "with a total of", sum([len(kmers) for kmers in kmers_list]), "kmers.")

    T, graph, startvec = debrujin.build_de_bruijn_graph(kmers_list=kmers_list, k=k, compactable=compactable, bins=bins)
    endvec = utils.enddist(seqpath=pathseq)
    
    if not compactable:
        sample.MCMC_sampling(T=T, start_idx=startvec, endvec=endvec, num_samples=num_samples, k=k-1)
        print("MCMC sampling completed.")
    

if __name__ == "__main__":
    pathseq = r'data/GII/GII.fasta'
    output_path = r'out/sequences/seqs.fasta'
    k = 16
    num_samples = 10
    device = "cpu"
    bins = 1
    main(pathseq=pathseq, 
         output_path=output_path, 
         k=k, 
         num_samples=num_samples, 
         device=device,
         bins=bins)