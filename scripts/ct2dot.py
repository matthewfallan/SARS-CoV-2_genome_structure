"""
Convert an RNA secondary structure from connectivity table to dot-bracket format.

- Connectivity table format example:
4	My_RNA
1	C	0	2	4	1
2	U	1	3	0	2
3	U	2	4	0	3
4	G	3	5	1	4

  - column 1: base index
  - column 2: base type
  - column 3: index - 1
  - column 4: index + 1
  - column 5: index of paired base (0 if unpaired)
  - column 6: natural index

- Dot bracket format example (of the same RNA)
CUUG
(..)

"""

import sys

import pandas as pd

ct_file, dotbracket_file = sys.argv[1:]

# Keep a record of the base pairs.
base_pairs = list()
paired_bases = set()
standard_base_pairs = {'AU', 'CG', 'GU'}
def standardize_base_pair(base1: str, base2: str) -> str:
    return ''.join(sorted([base1, base2]))

open_pair = ['(', '<', '[', '{']
close_pair = [')', '>', ']', '}']

# Read the connectivity table format structure.
with open(ct_file) as f:
    length, name = f.readline().split()
    # Initialize the dot-bracket structure to fully unpaired.
    unpaired = '.'
    dotbracket = [unpaired] * int(length)
    # Determine the seqeunce.
    lines = f.readlines()
    seq = ''.join([line.split()[1] for line in lines])
    # Add base pairs to the dot-bracket structure.
    for line in lines:
        index, base, index_dec, index_inc, index_pair, index_nat = line.split()
        if int(index_pair) == 0:
            # Skip unpaired bases.
            continue
        base1 = int(index) - 1  # subtract 1 because bases are 1-indexed
        base2 = int(index_pair) - 1  # subtract 1 because bases are 1-indexed
        assert(base1 != base2)
        pair = sorted([base1, base2])
        # If the base is already paired, ensure its mate matches.
        if base1 in paired_bases or base2 in paired_bases:
            assert(base1 in paired_bases and base2 in paired_bases)
            assert(pair in base_pairs)
        else:
            paired_bases.add(base1)
            paired_bases.add(base2)
            base_pairs.append(pair)
            # Make sure the base pair is chemically feasible.
            base1_type = seq[base1]
            base2_type = seq[base2]
            assert(standardize_base_pair(base1_type, base2_type) in standard_base_pairs)
            # Label the bases in dot-bracket format.
            dotbracket[base1] = open_pair
            dotbracket[base2] = close_pair

dotbracket = ''.join(dotbracket)
with open(dotbracket_file, 'w') as f:
    f.write(f"{seq}\n{dotbracket}")

