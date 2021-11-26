# Secondary structural ensembles of the SARS-CoV-2 RNA genome in infected cells
 
Analyze quality of structural model, structural ensembles, and comparisons to models by Pyle, Incarnato, and Das labs.

## files
- `plots.xlsx`: description of data used in generating the plots in the following figures:
  - Fig 4b
  - Fig 5b

## directories

### scripts
- `run_analysis.py`: performs most of the analyses and generates data and plots for the following figures:
  - Fig 1: b, c, d
  - Fig 2: a, b
  - Fig 3
  - Fig S2: b, c
  - Fig S3
  - Fig S4
  - Fig S5
  - Fig S6: a, b
  - Fig S7: b, c
  - Fig S10: a, b
- `models.py`: a module for loading structural information from our models, Manfredonia et al., Huston et al., and Sun et al.; imported by `run_analysis`
- `compare_to_rangan.py`: compares structure to model from Rangan et al. and generates the following figures:
  - Fig S8: a, b, c, d, e, f
  - Fig S9: a
- `make_files_for_fig5_ldi.py`: make the raw files used in preparing the structure models in Fig 5
- `cm.py`: perform covariation analysis on the structure of the SARS-CoV-2 genome; not used for any figures but appears in our responses to reviewers.
- `get_db_seq_matching_ref_genome.py`: find the sequence in the SARS-CoV-2 sequence database that matches the reference genome
- `fold_genomes.py`: fold the SARS-2 genome using staggered windows to remove edge effects; results were extremely similar to ignoring edge effects, so not used in this paper
- `TRS_boundaries.py`: Compute how many bases on each side of a TRS core sequence should be required for counting the TRS as unambiguously part of gRNA; not used for any results in this paper
- `compare_to_zhang.py`: Compare our results to the model from Zhang et al. (2021). Nat. Comm. 12: 5695; not used for any results in this paper
- `ct2dot.py`: convert an RNA structure in CT format to dot-bracket format

### models
- `controls`: structure models of SARS-CoV-2 SL5, HIV RRE, and U4/U6 snRNA
- `Huston`: SHAPE reactivities and structure model from Huston et al. 
- `Manfredonia`: SHAPE and DMS reactivities and structure model from Manfredonia et al.
- `Sun`: icSHAPE reactivities and structure model from Sun et al.
- `Zhang`: structure models of the SARS-CoV-2 FSE from Zhang et al.
- `Lan`: lists of files of structure models

### fold
structure models of staggered regions of the SARS-CoV-2 based on the Vero and Huh7 DMS-MaPseq reactivities

