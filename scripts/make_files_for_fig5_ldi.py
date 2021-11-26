"""

Make the dot-bracket and DMS files for Figure 5 of the two clusters
one of which is the long-distance interaction.

"""

import argparse
import os


def read_first_dot_entry(dot_file):
    with open(dot_file) as f:
        title = f.readline()
        seq = f.readline().strip()
        struct = f.readline().strip()
    return title, seq, struct

def write_dot_entry(dot_file, title, seq, struct, overwrite=False):
    if os.path.isfile(dot_file) and not overwrite:
        raise ValueError(dot_file)
    with open(dot_file, "w") as f:
        f.write(f"{title}\n{seq}\n{struct}")

def read_dms_values(dms_file):
    with open(dms_file) as f:
        dms_values = list(map(float, f))
    return dms_values

def write_dms_values(dms_file, dms_values, overwrite=False):
    if os.path.isfile(dms_file) and not overwrite:
        raise ValueError(dot_file)
    with open(dms_file, "w") as f:
        f.write("\n".join(map(str, dms_values)))


run_dir = ("/lab/solexa_rouskin/projects/Tammy/Tammy_corona/"
           "EM_Clustering_200b_exp1500/"
           "210511Rou_D21_4517_SARS2MN985325WA_SARS2MN985325WA_13434_13601_"
           "InfoThresh-0.05_SigThresh-0.005_IncTG-NO_DMSThresh-0.5/"
           "K_2/run_1-best")

file_in_template = ("210511Rou_D21_4517_SARS2MN985325WA_SARS2MN985325WA_"
        "13434_13601_InfoThresh-0.05_SigThresh-0.005_IncTG-NO_DMSThresh-0.5-K2_"
        "Cluster{}_expUp_1500_expDown_1500{}")
dot_out_template = os.path.join(run_dir, "FSEstructsFigure_Cluster{}_{}.dot")
dms_out_template = os.path.join(run_dir, "FSEstructsFigure_Cluster{}_{}.dms")
cluster_names = {1: "no-LDI", 2: "LDI"}
#cluster_coords = {1: (13196, 13595), 2: (13256, 14695)}
cluster_coords = {1: (13239, 13595), 2: (13256, 14695)}

nan_value = -1


pipeline_dir = "/lab/solexa_rouskin/mfallan_git/RNA_structure"
projects_dir = "/lab/solexa_rouskin/projects"
project = "Tammy/Tammy_corona"
em_dir = os.path.join(projects_dir, project, "EM_Clustering")
data_start = 13369
data_end = 13597
k = 2
run = 2
run_dir = os.path.join(em_dir, "sars2_30kb_PCRframeshift1_"
        f"SARS2MN985325WA_{data_start}_{data_end}_InfoThresh-0.99_"
        f"SigThresh-0.005_IncTG-NO_DMSThresh-0.5/K_{k}/run_{run}-best")
clusters_mu_file = os.path.join(run_dir, "Clusters_Mu.txt")
exp_start = 13370
exp_end = 14842
expUp = max(data_start - exp_start, 0)
expDown = max(exp_end - data_end, 0)
norm_bases = int(round((data_end - data_start + 1) / 20))

def fold():
    fold_cmd = (f"python {pipeline_dir}/EM_ExpandFold.py "
            f"{project} {clusters_mu_file} {expUp} {expDown} {norm_bases}")
    exstat = os.system(fold_cmd)
    assert exstat == 0


def trim():
    for cluster, name in cluster_names.items():
        print(cluster, name)
        start, end = cluster_coords[cluster]
        trim_start = start - region_start
        trim_end = end - region_start + 1
        # Read sequence and structure.
        cluster_dot_in = os.path.join(run_dir,
                file_in_template.format(cluster, ".dot"))
        title, seq, struct = read_first_dot_entry(cluster_dot_in)
        # Trim sequence and structure.
        seq_trim = seq[trim_start: trim_end]
        struct_trim = struct[trim_start: trim_end]
        print("before", struct[trim_start - 30: trim_start])
        print("middle", struct_trim)
        print("after ", struct[trim_end: trim_end + 30])
        # Write trimmed sequence and structure.
        cluster_dot_out = dot_out_template.format(cluster, name)
        write_dot_entry(cluster_dot_out, title, seq_trim, struct_trim,
                overwrite=True)
        # Read DMS signals (for base colors).
        cluster_dms_in = os.path.join(run_dir,
                file_in_template.format(cluster, "_varna.txt"))
        dms_values = read_dms_values(cluster_dms_in)
        # Nullify bases with no signal.
        for pos, base in enumerate(seq, start=region_start):
            if not (base in "AC" and data_start <= pos <= data_end):
                dms_values[pos - region_start] = nan_value
        # Trim DMS signals.
        dms_values_trim = dms_values[trim_start: trim_end]
        # Write trimmed DMS signals.
        cluster_dms_out = os.path.join(run_dir,
                dms_out_template.format(cluster, name))
        write_dms_values(cluster_dms_out, dms_values_trim, overwrite=True)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", action="store_true")
    parser.add_argument("--trim", action="store_true")
    args = parser.parse_args()
    if args.fold:
        fold()
    if args.trim:
        trim()

