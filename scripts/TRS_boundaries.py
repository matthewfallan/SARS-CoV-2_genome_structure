"""
Compute how many bases on each side of a TRS core sequence should be required for counting the TRS as unambiguously part of gRNA.
"""

from Bio import SeqIO
import numpy as np
import itertools

seq_file = "../seqs/SARS-CoV-2_TRSs.fasta"

seqs = [str(rec.seq) for rec in SeqIO.parse(seq_file, "fasta")]



def hamming_distances_relative(seqs):
    dists = np.array([np.mean([x1 != x2 for x1, x2 in zip(seqs[i1], seqs[i2])]) for i1 in range(len(seqs) - 1) for i2 in range(i1 + 1, len(seqs))])
    return dists

def hamming_distances_relative_to_seq1(seqs):
    dists = np.array([np.mean([x1 != x2 for x1, x2 in zip(seqs[i1], seqs[0])]) for i1 in range(1, len(seqs))])
    return dists

def find_length(seqs, start, direction, threshold=0.25):
    """
    Find the minimum length from start such that no pairs of sequences are < threshold different.
    """
    seq_lengths = [len(seq) for seq in seqs]
    assert len(set(seq_lengths)) == 1  # all sequences are same length
    seq_len = seq_lengths[0]
    length = 1
    while True:
        if direction == "5'":
            seq_sections = [seq[start - length: start] for seq in seqs]
        elif direction == "3'":
            seq_sections = [seq[start: start + length] for seq in seqs]
        dists = hamming_distances_relative_to_seq1(seq_sections)
        if np.all(dists > threshold):
            return length
        if (direction == "5'" and length == start) or (direction == "3'" and start + length == seq_len):
            return
        length += 1

threshold = 0.5
len_5p = find_length(seqs, 30, "5'", threshold)
len_3p = find_length(seqs, 36, "3'", threshold)
print("Minimum distance:", threshold)
print("5':", len_5p)
print("3':", len_3p)

