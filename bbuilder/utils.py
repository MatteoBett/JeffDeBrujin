import torch

def progressbar(iteration, total, prefix = '', suffix = '', filler = '█', printEnd = "\r") -> None:
    """ Show a progress bar indicating downloading progress """
    percent = f'{round(100 * (iteration / float(total)), 1)}'
    add = int(100 * iteration // total)
    bar = filler * add + '-' * (100 - add)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    if iteration == total: 
        print()

def kmer2str(val, k):
    """ Transform a kmer integer into a its string representation
    :param int val: An integer representation of a kmer
    :param int k: The number of nucleotides involved into the kmer.
    :return str: The kmer string formatted
    """
    letters = ['A', 'C', 'T', 'G']
    str_val = []
    for _ in range(k):
        str_val.append(letters[val & 0b11])
        val >>= 2

    str_val.reverse()
    return "".join(str_val)

def saveseq(seqs, path):
    """ Save a list of sequences to a file """
    with open(path, 'w') as f:
        for i, seq in enumerate(seqs):
            f.write(f">Sequence_{i}\n{seq}\n") 

def bit2seq(samples : torch.Tensor, end_idx : int, k: int, output_path : str) -> str:
    """ Convert a tensor of samples to a sequence string """
    k -= 1
    seqs = []
    start_vec = samples[0, :]
    end_mask = samples == end_idx

    rest = samples[1:, :] & 0b11
    
    startvec_nuc = [kmer2str(val=val, k=k) for val in start_vec]
    
    for j, start in enumerate(startvec_nuc):
        seq = start
        for i in range(rest.shape[0]):
            if end_mask[i, j] == 0:
                seq += kmer2str(val=rest[i, j], k=k)[-1]
            else:
                break
        seqs.append(seq)

    saveseq(seqs=seqs, path=output_path)

    return seqs
