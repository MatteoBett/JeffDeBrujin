import os, re

from bbuilder import loader, streamkmers
from bbuilder import debrujin


def main(pathseq : str, 
         k : int, 
         device : str
         ) -> None:
    
    kmers_list = [streamkmers.main_kmers(seq=seq, k=k, device=device) for seq in loader.loadfasta(path=pathseq)]
    graph = debrujin.build_de_bruijn_graph(kmers_list=kmers_list, k=k)
    L = debrujin.Laplacian(G = graph)



if __name__ == "__main__":
    pathseq = r'data/test/GII/GII.fasta'
    k = 7
    device = "cpu"
    main(pathseq=pathseq, k=k, device=device)