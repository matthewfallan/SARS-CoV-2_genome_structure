import os

import pandas as pd

from models import get_Lan_seq_struct_mus
from rouls.dreem_utils import get_data_structure_agreement, get_sample_and_run, read_clusters_mu, get_folding_filename, plot_data_structure_roc_curve
from rouls.struct_utils import read_dot_file_single, read_ct_file_single
from rouls.seq_utils import read_fasta



def get_Lan_seq_struct_mus_PCR(em_clustering_dir, k, cluster, start, *args, **kwargs):
    run_dir = get_sample_and_run(em_clustering_dir=em_clustering_dir, k=k, *args, **kwargs)
    folding_file = get_folding_filename(run_dir, cluster, 0, 0, ".ct")
    name, pairs, paired, seq = read_ct_file_single(folding_file, start_pos=start, multiple=0)
    clusters_mu_file = os.path.join(run_dir, "Clusters_Mu.txt")
    mus = read_clusters_mu(clusters_mu_file, flatten=False, include_gu=False, seq=seq, start_pos=start)[str(cluster)]
    return pairs, paired, seq, mus


EM_vero = "/lab/solexa_rouskin/projects/Tammy/Tammy_corona/EM_Clustering"

lan_datasets = {
        "Lan-Vero_lib": get_Lan_seq_struct_mus("Vero", "K1"),
        #"Lan-Huh7_PCR": get_Lan_seq_struct_mus("Huh7", "K1"),
}
for k in range(1, 3+1):
    for cluster in range(1, k + 1):
        label = f"Lan-Vero_FSE-{k}.{cluster}"
        data = get_Lan_seq_struct_mus_PCR(EM_vero, sample="sars2_30kb_PCRframeshift1", ref="SARS2MN985325WA", start=13369, end=13597, info=0.1, k=k, cluster=cluster)
        lan_datasets[label] = data


zhang_models_1b = [
        "Fig-1b_pk",
        "Fig-1b_arch1",
]
zhang_models_1c = [
        "Fig-1c_Ziv",
        "Fig-1c_arch2",
        "Fig-1c_arch3",
]
zhang_models = zhang_models_1b + zhang_models_1c


def get_zhang_model(model):
    if model in zhang_models_1b:
        dot_file = os.path.join("../models/Zhang", f"{model}.dot")
        name, paired, seq = read_dot_file_single(dot_file)
        first, last = map(int, name.split("-"))
        paired.index = paired.index + (first - paired.index[0])
        assert paired.index[-1] == last
    elif model in zhang_models_1c:
        paired_file = "../models/Zhang/Fig-1c.xlsx"
        paired = pd.read_excel(paired_file, sheet_name=model, index_col="Position")["Paired"]
    return paired


def compare_dataset_and_model(dataset, model):
    mus = lan_datasets[dataset][3]
    if model in zhang_models:
        paired = get_zhang_model(model)
    elif model in lan_datasets:
        paired = lan_datasets[model][1]
    else:
        raise ValueError(model)
    first, last = paired.index[[0, -1]]
    auroc = get_data_structure_agreement("AUROC", paired, mus)
    print("Data:", dataset, "Model:", model, "AUROC:", auroc)
    return paired


def compare_dataset_and_models(dataset, use_models, plot_file):
    mus = lan_datasets[dataset][3]
    mus = pd.DataFrame(data={model: mus for model in use_models}, index=mus.index)
    paired = pd.DataFrame({model: compare_dataset_and_model(dataset, model) for model in use_models})
    paired.dropna(axis=0, how="any", inplace=True)
    plot_data_structure_roc_curve(paired, mus, plot_file)


for k in range(1, 3+1):
    for cluster in range(1, k + 1):
        lan_model = f"Lan-Vero_FSE-{k}.{cluster}"
        for zhang_model in zhang_models_1b:
            compare_dataset_and_models(lan_model,
                ["Fig-1b_pk", lan_model],
                f"{zhang_model}_{k}.{cluster}.pdf")
for zhang_model in zhang_models_1c:
    compare_dataset_and_models("Lan-Vero_lib",
        [zhang_model, "Lan-Vero_lib"],
        f"{zhang_model}_lib.pdf")

