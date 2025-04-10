import warnings
import torch
import re
import os
import urllib.request
import pandas as pd
import torch.nn.functional as F
import numpy as np
import antifold.esm

from biotite.structure.io import pdb


amino_list = list("ACDEFGHIKLMNPQRSTVWY")

IMGT_dict = {
    "all": range(1, 128 + 1),
    "allH": range(1, 128 + 1),
    "allL": range(1, 128 + 1),
    "FWH": list(range(1, 26 + 1)) + list(range(40, 55 + 1)) + list(range(66, 104 + 1)),
    "FWL": list(range(1, 26 + 1)) + list(range(40, 55 + 1)) + list(range(66, 104 + 1)),
    "CDRH": list(range(26, 33 + 1)) + list(range(51, 57 + 1)) + list(range(96, 111 + 1)),
    # "CDRL": list(range(27, 39)) + list(range(56, 65 + 1)) + list(range(105, 117 + 1)),
    "FW1": range(1, 26 + 1),
    "FWH1": range(1, 26 + 1),
    "FWL1": range(1, 26 + 1),
    # "CDR1": range(27, 39),
    "CDRH1": range(26, 33 + 1), # (resid 26 - 33)
    # "CDRL1": range(27, 39),
    "FW2": range(40, 55 + 1),
    "FWH2": range(40, 55 + 1),
    "FWL2": range(40, 55 + 1),
    # "CDR2": range(56, 65 + 1),
    "CDRH2": range(51, 57 + 1), # (resid 51 - 57)
    # "CDRL2": range(56, 65 + 1),
    "FW3": range(66, 104 + 1),
    "FWH3": range(66, 104 + 1),
    "FWL3": range(66, 104 + 1),
    # "CDR3": range(105, 117 + 1),
    "CDRH3": range(96, 111 + 1), # (resid 96 - 111)
    # "CDRL3": range(105, 117 + 1),
    "FW4": range(118, 128 + 1),
    "FWH4": range(118, 128 + 1),
    "FWL4": range(118, 128 + 1),
}

def seed_everything(seed: int):
    # https://gist.github.com/ihoromi4/b681a9088f348942b01711f251e5f964
    import os
    import random

    import numpy as np
    import torch

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def load_IF1_checkpoint(model, checkpoint_path: str = ""):
    # Load
    print(f'[DEBUG - antiscripts.py load_IF1_checkpoint()] Loading AntiFold model {checkpoint_path} ...')
    # log.debug(f"Loading AntiFold model {checkpoint_path} ...")

    # Check for CPU/GPU load
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint_dict = torch.load(checkpoint_path, map_location=torch.device(device))

    # PYL checkpoint ?
    if "model_state_dict" in checkpoint_dict.keys():
        pretrained_dict = {
            re.sub("model.", "", k): v
            for k, v in checkpoint_dict["model_state_dict"].items()
        }

    # PYL checkpoint ?
    elif "state_dict" in checkpoint_dict.keys():
        # Remove "model." from keys
        pretrained_dict = {
            re.sub("model.", "", k): v for k, v in checkpoint_dict["state_dict"].items()
        }

    # IF1-raw?
    else:
        pretrained_dict = checkpoint_dict

    # Load pretrained weights
    model.load_state_dict(pretrained_dict)

    return model


def load_model(checkpoint_path: str = ""):
    """Load raw/FT IF1 model"""

    # Check that AntiFold weights are downloaded
    root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = f"{root_dir}/models/model.pt"

    if not os.path.exists(model_path):
        print(f'[DEBUG - antiscripts.py load_model()] Downloading AntiFold model weights from https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt to {model_path}')
        # log.warning(
        #     f"Downloading AntiFold model weights from https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt to {model_path}"
        # )
        url = "https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt"
        filename = model_path

        os.makedirs(f"{root_dir}/models", exist_ok=True)
        urllib.request.urlretrieve(url, filename)

    if not os.path.exists(model_path) and not checkpoint_path == "ESM-IF1":
        raise Exception(
            f"Unable to find model weights. File does not exist: {checkpoint_path}"
        )

    # Download IF1 weights
    if checkpoint_path == "ESM-IF1":
        # log.info(
        #     f"NOTE: Loading ESM-IF1 weights instead of fine-tuned AntiFold weights"
        # )
        # Suppress regression weights warning - not needed
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model, _ = antifold.esm.pretrained.esm_if1_gvp4_t16_142M_UR50()

    # Load AntiFold weights locally
    else:
        model, _ = antifold.esm.pretrained._load_IF1_local()
        model = load_IF1_checkpoint(model, model_path)

    # Evaluation mode when predicting
    model = model.eval()

    # Send to CPU/GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _ = model.to(device)
    # log.info(f"Loaded model to {device}.")

    return model

def get_temp_probs(df, t=0.20):
    """Gets temperature scaled probabilities for sampling"""
    print(f'[DEBUG - antiscripts_utils.py get_temp_probs()] Scaling logits by temperature: {t} and softmax', flush=True)
    logits = get_df_logits(df)
    temp_logits = logits / np.max([t, 0.001])  # Lower-bound to prevent div by 0
    temp_probs = F.softmax(temp_logits, dim=1)

    return temp_probs

def calc_pos_perplexity(df):
    cols = list("ACDEFGHIKLMNPQRSTVWY")
    t = torch.tensor(df[cols].values)
    probs = F.softmax(t, dim=1)
    perplexities = torch.pow(2, -(probs * torch.log2(probs)).sum(dim=1))

    return perplexities.numpy()

def sequence_to_onehot(sequence):
    amino_list = list("ACDEFGHIKLMNPQRSTVWY")
    one_hot = np.zeros((len(sequence), len(amino_list)), dtype=int)
    for i, res in enumerate(sequence):
        one_hot[i, amino_list.index(res)] = 1
    return one_hot


def extract_chains_biotite(pdb_file):
    """Extract chains in order"""
    pdbf = pdb.PDBFile.read(pdb_file)
    structure = pdb.get_structure(pdbf, model=1)
    return pd.unique(structure.chain_id)

def pdb_posins_to_pos(pdb_posins):
    # Convert pos+insertion code to numerical only
    return pdb_posins.astype(str).str.extract(r"(\d+)")[0].astype(int).values


def get_imgt_mask(df, imgt_regions=["CDRH1", "CDRH2", "CDRH3"]):
    """Returns e.g. CDRH1+2+3 mask"""

    positions = pdb_posins_to_pos(df["pdb_posins"])
    region_pos_list = list()

    for region in imgt_regions:
        if str(region) not in IMGT_dict.keys():
            region_pos_list.extend(region)
        else:
            region_pos_list.extend(list(IMGT_dict[region]))
            print(f'[DEBUG - antiscripts_utils.py get_imgt_mask()] region: {region} has positions: {list(IMGT_dict[region])}', flush=True)

    region_mask = pd.Series(positions).isin(region_pos_list).values
    return region_mask


def get_df_logits(df):
    cols = list("ACDEFGHIKLMNPQRSTVWY")
    logits = torch.tensor(df[cols].values)

    return logits


def get_temp_probs(df, t=0.20):
    """Gets temperature scaled probabilities for sampling"""
    print(f'[DEBUG - antiscripts_utils.py get_temp_probs()] Scaling logits by temperature: {t} and softmax', flush=True)
    logits = get_df_logits(df)
    temp_logits = logits / np.max([t, 0.001])  # Lower-bound to prevent div by 0
    temp_probs = F.softmax(temp_logits, dim=1)

    return temp_probs


def get_dfs_HL(df):
    """Split df into heavy and light chains"""
    Hchain, Lchain = df["pdb_chain"].unique()[:2] # Assume heavy, light
    return df[df["pdb_chain"] == Hchain], df[df["pdb_chain"] == Lchain]


def get_df_seq(df):
    """Get PDB sequence"""
    return df["pdb_res"].values


def get_df_seq_pred(df):
    """Get PDB sequence"""
    return df["top_res"].values


def get_df_seqs_HL(df):
    """Get heavy and light chain sequences"""
    df_H, df_L = get_dfs_HL(df)
    return get_df_seq(df_H), get_df_seq(df_L)


def visualize_mutations(orig, mut, chain=""):
    """Visualize mutations between two sequences"""

    # Convert to numpy array
    # (whether string, list, Bio.Seq.Seq or np.array)
    orig = np.array(list(orig))
    mut = np.array(list(mut))
    mismatches = "".join(["X" if match else "_" for match in (orig != mut)])

    # Print
    print(f"Mutations ({(orig != mut).sum()}):\t{mismatches}")
    print(f"Original {chain}:\t\t{''.join(orig)}")
    print(f"Mutated {chain}:\t\t{''.join(mut)}\n")
