"""
Find the name of the sequence in the ref database that matches the sequence in the ref genome.
"""


from Bio import SeqIO
from tqdm import tqdm


ref_file = "/lab/solexa_rouskin/projects/Tammy/Tammy_corona/Ref_Genome/SARS2MN985325WA.fasta"
ref_seq = str(SeqIO.read(ref_file, "fasta").seq)

db_file = "/lab/solexa_rouskin/projects/mfallan/SARS2_genome_structure/seqs/CoVs_db.fasta"
db_seqs = SeqIO.parse(db_file, "fasta")
for record in tqdm(db_seqs):
    if str(record.seq) == ref_seq:
        print()
        print(record.id)
        break

