import os, re

from bbuilder import loader, streamkmers
from bbuilder import debrujin


def main(pathseq : str, 
         k : int, 
         device : str
         ) -> None:
    
    kmers_list = [streamkmers.main_kmers(seq=seq, k=k, device=device) for seq in loader.loadfasta(path=pathseq)]
    print("Loaded", len(kmers_list), "sequences from", pathseq, "with a total of", sum([len(kmers) for kmers in kmers_list]), "kmers.")
    graph = debrujin.build_de_bruijn_graph(kmers_list=kmers_list, k=k)
    print("Compacted De Bruijn graph built with", len(graph), "nodes.")



if __name__ == "__main__":
    pathseq = r'data/GII/GII.fasta'
    k = 11
    device = "cpu"
    main(pathseq=pathseq, k=k, device=device)