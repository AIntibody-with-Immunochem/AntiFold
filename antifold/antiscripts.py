import glob
# import logging
import os
import sys
import warnings
import urllib.request
from pathlib import Path
from collections import OrderedDict


import torch
import torch.nn.functional as F

import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq

ROOT_PATH = Path(os.path.dirname(__file__)).parent
sys.path.insert(0, ROOT_PATH)

from biotite.structure.io import pdb
import re
import antifold.esm

from antifold.esm_util_custom import CoordBatchConverter_mask_gpu
from antifold.if1_dataset import InverseData
from antifold.antiscripts_utils import (calc_pos_perplexity, 
                                        df_logits_to_logprobs,
                                        pdb_posins_to_pos,
                                        df_logits_to_logprobs)

# log = logging.getLogger(__name__)

amino_list = list("ACDEFGHIKLMNPQRSTVWY")

def extract_chains_biotite(pdb_file):
    """Extract chains in order"""
    pdbf = pdb.PDBFile.read(pdb_file)
    structure = pdb.get_structure(pdbf, model=1)
    return pd.unique(structure.chain_id)


def generate_pdbs_csv(pdbs_dir, max_chains=10):
    """Generates AntiFOld CSV with PDB/CIF basenames + chains (up to 10)"""

    columns = [
        "pdb",
        "Hchain",
        "Lchain",
        "chain3",
        "chain4",
        "chain5",
        "chain6",
        "chain7",
        "chain8",
        "chain9",
        "chain10",
    ]
    pdbs_csv = pd.DataFrame(columns=columns)
    pdbs = glob.glob(f"{pdbs_dir}/*.[pdb cif]*")

    for i, pdb_file in enumerate(pdbs):
        _pdb = os.path.splitext(os.path.basename(pdb_file))[0]

        # Extract first 10 chains
        chains = extract_chains_biotite(pdb_file)[:max_chains]

        # Put pdb and chains in dataframe
        row = [_pdb] + list(chains)
        _cols = columns[: len(row)]
        pdbs_csv.loc[i, _cols] = row

    return pdbs_csv


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


def get_dataset_pdb_name_chainsname_res_posins_chains(dataset, idx):
    """Gets PDB sequence, position+insertion codes and chains from dataset"""

    # PDB path
    pdb_path = dataset.pdb_info_dict[idx]["pdb_path"]

    # PDB names
    pdb_name = dataset.pdb_info_dict[idx]["pdb"]
    pdb_chainsname = dataset.pdb_info_dict[idx]["pdb_chainsname"]

    # Sequence - gaps
    seq = dataset[idx][2]
    pdb_res = [aa for aa in seq if aa != "-"]

    # Position + insertion codes - gaps
    posinschain = dataset[idx][4]
    posinschain = [p for p in posinschain if p != "nan"]

    # Extract position + insertion code + chain (1-letter)
    pdb_posins = [p[:-1] for p in posinschain]
    pdb_chains = [p[-1] for p in posinschain]

    return pdb_name, pdb_chainsname, pdb_res, pdb_posins, pdb_chains


def logits_to_seqprobs_list(logits, tokens):
    """Convert logits (bs x 35 x L) ot list of L x 20 seqprobs"""

    alphabet = antifold.esm.data.Alphabet.from_architecture("invariant_gvp")

    mask_gap = tokens[:, 1:] != 30  # 30 is gap
    mask_pad = tokens[:, 1:] != alphabet.padding_idx  # 1 is padding
    mask_combined = mask_gap & mask_pad

    # Check that only 10x gap ("-") per sequence!
    # batch_size = logits.shape[0]
    # assert (mask_gap == False).sum() == batch_size * 10

    # Filter out gap (-) and padding (<pad>) positions, only keep 21x amino-acid probs (4:24) + "X"
    seqprobs_list = [logits[i, 4:25, mask_combined[i]] for i in range(len(logits))]

    return seqprobs_list


def get_dataset_dataloader(
    pdbs_csv_or_dataframe, pdb_dir, batch_size, custom_chain_mode=False, num_threads=0
):
    """Prepares dataset/dataoader from CSV file containing PDB paths and H/L chains"""
    print('[DEBUG - antiscripts.py get_dataset_dataloader()] Trying to load PDB coordinates', flush=True)
    print(f'[DEBUG - antiscripts.py get_dataset_dataloader()] pdbs_csv_or_dataframe: {pdbs_csv_or_dataframe}', flush=True)
    print(f'[DEBUG - antiscripts.py get_dataset_dataloader()] pdb_dir: {pdb_dir}', flush=True)
    print(f'[DEBUG - antiscripts.py get_dataset_dataloader()] batch_size: {batch_size}', flush=True)
    print(f'[DEBUG - antiscripts.py get_dataset_dataloader()] custom_chain_mode: {custom_chain_mode}', flush=True)


    # Set number of threads & workers
    if num_threads >= 1:
        torch.set_num_threads(num_threads)
        num_threads = min(num_threads, 4)

    # Load PDB coordinates
    try:
        print('[DEBUG - antiscripts.py get_dataset_dataloader()] Creating InverseData instance', flush=True)
        dataset = InverseData(
            gaussian_noise_flag=False, custom_chain_mode=custom_chain_mode,
        )
        print('[DEBUG - antiscripts.py get_dataset_dataloader()] Calling dataset.populate()', flush=True)
        dataset.populate(pdbs_csv_or_dataframe, pdb_dir)
        print('[DEBUG - antiscripts.py get_dataset_dataloader()] dataset.populate() completed successfully\n', flush=True)
    except Exception as e:
        print(f'[ERROR - antiscripts.py get_dataset_dataloader()] Exception in dataset creation: {e}', flush=True)
        raise

    # Prepare torch dataloader at specified batch size
    alphabet = antifold.esm.data.Alphabet.from_architecture("invariant_gvp")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=CoordBatchConverter_mask_gpu(alphabet),
        num_workers=num_threads,
    )

    return dataset, dataloader


def dataset_dataloader_to_predictions_list(
    model, dataset, dataloader, batch_size=1, extract_embeddings=False
):
    """Get PDB predictions from a dataloader"""

    # Check that dataloader and dataset match, and no random shuffling
    if "random" in str(dataloader.sampler).lower():
        raise ValueError(
            "Torch DataLoader sampler must not be random. Did you forget to set torch.utils.data.DataLoader ... shuffle=False?"
        )
    if dataloader.dataset is not dataset:
        raise ValueError("Dataloader and dataset must match to align samples!")

    # Get all batch predictions
    all_seqprobs_list = []
    all_embeddings_list = []
    for bi, batch in enumerate(dataloader):
        start_index = bi * batch_size
        end_index = min(
            start_index + batch_size, len(dataset)
        )  # Adjust for the last batch
        # log.info(
        #     f"Predicting batch {bi+1}/{len(dataloader)}: PDBs {start_index+1}-{end_index} out of {len(dataset)} total"
        # )  # -1 because the end_index is exclusive

        # Test dataloader
        (
            coords,
            confidence,
            strs,
            tokens,
            padding_mask,
            loss_masks,
            res_pos,
            posinschain_list,
            targets,
        ) = batch

        # Test forward
        with torch.no_grad():
            prev_output_tokens = tokens[:, :-1]
            logits, extra = model.forward(  # bs x 35 x seq_len, exlude bos, eos
                coords,
                padding_mask,  # Includes masked positions
                confidence,
                prev_output_tokens,
                features_only=False,
            )

            logits = logits.detach().cpu().numpy()
            tokens = tokens.detach().cpu().numpy()

            # List of L x 21 seqprobs (20x AA, 21st == "X")
            L = logits_to_seqprobs_list(logits, tokens)
            all_seqprobs_list.extend(L)

            if extract_embeddings:
                # Extract embeddings from batch
                padding_mask = padding_mask.detach().cpu().numpy()  # bs x len
                Es = extra["inner_states"][0].detach().cpu().numpy()  # len x bs x 512
                # L[0] = encoder output
                # L[1] = decoder input
                # L[2] = decoder output layer 1
                # L[3] = decoder output layer 2 ...
                Es = np.transpose(Es, (1, 0, 2))  # bs x len x 512

                # Embeddings without padding and bos/eos
                for e, pad in zip(Es, padding_mask):
                    # e: len x 512
                    e = e[~pad]  # (len - pad) x 512
                    e = e[1:-1]  # seq_len x 512
                    all_embeddings_list.append(e)  # seq_len x 512

    return all_seqprobs_list, all_embeddings_list


def predictions_list_to_df_logits_list(all_seqprobs_list, dataset, dataloader):
    """PDB preds list to PDB DataFrames"""

    # Check that dataloader and dataset match, and no random shuffling
    if "random" in str(dataloader.sampler).lower():
        raise ValueError(
            "Torch DataLoader sampler must not be random. Did you forget to set torch.utils.data.DataLoader ... shuffle=False?"
        )
    if dataloader.dataset is not dataset:
        raise ValueError("Dataloader and dataset must match to align samples!")

    # Create DataFrame for each PDB
    all_df_logits_list = []
    for idx, seq_probs in enumerate(all_seqprobs_list):
        # Get PDB sequence, position+insertion code and H+L chain idxs
        (
            pdb_name,
            pdb_chainsname,
            pdb_res,
            pdb_posins,
            pdb_chains,
        ) = get_dataset_pdb_name_chainsname_res_posins_chains(dataset, idx)

        # Check matches w/ residue probs
        assert len(seq_probs) == len(pdb_posins)

        # Logits to DataFrame
        alphabet = antifold.esm.data.Alphabet.from_architecture("invariant_gvp")
        df_logits = pd.DataFrame(data=seq_probs, columns=alphabet.all_toks[4:25],)

        # Limit to 20x amino-acids probs
        _alphabet = list("ACDEFGHIKLMNPQRSTVWY")
        df_logits = df_logits[_alphabet]

        # Add PDB info
        positions = pdb_posins_to_pos(pd.Series(pdb_posins))
        top_res = np.array((_alphabet))[df_logits[_alphabet].values.argmax(axis=1)]
        perplexity = calc_pos_perplexity(df_logits)

        # Add to DataFrame
        df_logits.name = pdb_chainsname
        df_logits.insert(0, "pdb_posins", pdb_posins)
        df_logits.insert(1, "pdb_chain", pdb_chains)
        df_logits.insert(2, "pdb_res", pdb_res)
        df_logits.insert(3, "top_res", top_res)
        df_logits.insert(4, "pdb_pos", positions)
        df_logits.insert(5, "perplexity", perplexity)

        # Skip if not IMGT numbered - 10 never found in H-chain IMGT numbered PDBs
        Hchain = pdb_chains[0]
        Hpos = positions[pdb_chains == Hchain]
        if 10 in Hpos and not dataset.custom_chain_mode:
            # log.error(
            #     f"WARNING: PDB {pdb_name} seems to not be IMGT numbered! Output probabilities may be affected. See https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/anarci/"
            # )
            print(f"WARNING: PDB {pdb_name} seems to not be IMGT numbered! Output probabilities may be affected. See https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/")
        # Limit to IMGT positions only (only ones trained on)
        # imgt_mask = get_imgt_mask(df_logits, imgt_regions=["all"])
        # df_logits = df_logits[imgt_mask]

        all_df_logits_list.append(df_logits)

    return all_df_logits_list


def df_logits_list_to_logprob_csvs(
    df_logits_list, out_dir, embeddings_list=False, float_format="%.4f"
):
    """Save df_logits_list to CSVs"""
    os.makedirs(out_dir, exist_ok=True)
    # log.info(f"Saving {len(df_logits_list)} CSVs to {out_dir}")

    for i, df in enumerate(df_logits_list):
        # Convert to log-probs
        df_out = df_logits_to_logprobs(df)
        # Save
        outpath = f"{out_dir}/{df.name}.csv"
        # log.info(f"Writing {df.name} per-residue log probs CSV to {outpath}")
        df_out.to_csv(outpath, float_format=float_format, index=False)

        if embeddings_list:
            # Save embeddingsl
            outpath = f"{out_dir}/{df.name}.npy"
            # log.info(f"Writing {df.name} per-residue embeddings to {outpath}")
            np.save(outpath, embeddings_list[i])


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


def get_pdbs_logits(
    model,
    pdbs_csv_or_dataframe,
    pdb_dir,
    out_dir=False,
    batch_size=1,
    extract_embeddings=False,
    custom_chain_mode=False,
    num_threads=0,
    save_flag=False,
    float_format="%.4f",
    seed=42,
):
    """Predict PDBs from a CSV file"""
    # if save_flag:
    #    log.info(f"Saving prediction CSVs to {out_dir}")

    seed_everything(seed)

    # print('[DEBUG - antiscripts.py get_pdbs_logits()] Trying to predict PDBs from a CSV file')
    # print(f'[DEBUG - antiscripts.py get_pdbs_logits()] pdbs_csv_or_dataframe: {pdbs_csv_or_dataframe}')
    # print(f'[DEBUG - antiscripts.py get_pdbs_logits()] pdb_dir: {pdb_dir}')
    # print(f'[DEBUG - antiscripts.py get_pdbs_logits()] batch_size: {batch_size}')
    # print(f'[DEBUG - antiscripts.py get_pdbs_logits()] custom_chain_mode: {custom_chain_mode}')

    # Load PDBs
    try:
        print('\n[DEBUG - antiscripts.py get_pdbs_logits()] Calling get_dataset_dataloader()')
        dataset, dataloader = get_dataset_dataloader(
            pdbs_csv_or_dataframe,
            pdb_dir,
            batch_size=batch_size,
            custom_chain_mode=custom_chain_mode,
            num_threads=num_threads,
        )
        print('[DEBUG - antiscripts.py get_pdbs_logits()] get_dataset_dataloader() completed successfully')
    except Exception as e:
        print(f'[ERROR - antiscripts.py get_pdbs_logits()] Exception in get_dataset_dataloader(): {e}')
        raise

    # Predict PDBs -> df_logits
    try:
        print('[DEBUG - antiscripts.py get_pdbs_logits()] Calling dataset_dataloader_to_predictions_list()')
        predictions_list, embeddings_list = dataset_dataloader_to_predictions_list(
            model,
            dataset,
            dataloader,
            batch_size=batch_size,
            extract_embeddings=extract_embeddings,
        )
        print('[DEBUG - antiscripts.py get_pdbs_logits()] dataset_dataloader_to_predictions_list() completed successfully')
    except Exception as e:
        print(f'[ERROR - antiscripts.py get_pdbs_logits()] Exception in dataset_dataloader_to_predictions_list(): {e}')
        raise

    try:
        print('[DEBUG - antiscripts.py get_pdbs_logits()] Calling predictions_list_to_df_logits_list()')
        df_logits_list = predictions_list_to_df_logits_list(
            predictions_list, dataset, dataloader
        )
        print('[DEBUG - antiscripts.py get_pdbs_logits()] predictions_list_to_df_logits_list() completed successfully')
    except Exception as e:
        print(f'[ERROR - antiscripts.py get_pdbs_logits()] Exception in predictions_list_to_df_logits_list(): {e}')
        raise

    # Save df_logits to CSVs
    if save_flag:
        df_logits_list_to_logprob_csvs(
            df_logits_list,
            out_dir,
            embeddings_list=embeddings_list,
            float_format=float_format,
        )

    if extract_embeddings:
        return df_logits_list, embeddings_list
    else:
        return df_logits_list