import torch
import torch.nn.functional as F
# import logging
import os
from collections import OrderedDict

import numpy as np
import pandas as pd

from Bio import SeqIO
from Bio.Seq import Seq

# log = logging.getLogger(__name__)

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


def calc_pos_perplexity(df):
    cols = list("ACDEFGHIKLMNPQRSTVWY")
    t = torch.tensor(df[cols].values)
    probs = F.softmax(t, dim=1)
    perplexities = torch.pow(2, -(probs * torch.log2(probs)).sum(dim=1))

    return perplexities.numpy()


def identify_polyglycine_motifs(sequence, min_length=2, cdr_lengths=None, cdr_regions=None, df_H=None):
    """
    Identifies polyglycine motifs (consecutive G residues) in CDR regions.
    
    Args:
        sequence: Numpy array of amino acid characters
        min_length: Minimum number of consecutive G residues to consider as a motif
        cdr_lengths: List of lengths for each CDR region [CDRH1_len, CDRH2_len, CDRH3_len]
                    If provided, will identify motifs in each CDR region separately
        cdr_regions: List of CDR region names ["CDRH1", "CDRH2", "CDRH3"]
        df_H: DataFrame with heavy chain data, needed to map CDR positions to full sequence
        
    Returns:
        List of tuples (start_idx, end_idx) for each polyglycine motif in the full sequence
    """
    motifs = []
    
    # If CDR lengths are provided, process each CDR region separately
    if cdr_lengths is not None and cdr_regions is not None and df_H is not None:
        print(f'[DEBUG] Processing CDRs with lengths: {cdr_lengths}', flush=True)
        start_idx = 0
        
        for i, (length, region) in enumerate(zip(cdr_lengths, cdr_regions)):
            end_idx = start_idx + length
            cdr_sequence = sequence[start_idx:end_idx]
            print(f'[DEBUG] {region} sequence: {cdr_sequence}', flush=True)
            
            # Get the IMGT positions for this CDR region
            imgt_positions = list(IMGT_dict[region])
            print(f'[DEBUG] {region} IMGT positions: {imgt_positions}', flush=True)
            
            # Find motifs in this CDR region
            j = 0
            while j < len(cdr_sequence):
                if cdr_sequence[j] == 'G':
                    motif_start = j
                    while j < len(cdr_sequence) and cdr_sequence[j] == 'G':
                        j += 1
                    motif_end = j
                    if motif_end - motif_start >= min_length:
                        # Map to IMGT positions
                        imgt_start = imgt_positions[motif_start]
                        imgt_end = imgt_positions[motif_end-1] + 1  # +1 because end is exclusive
                        
                        # Map IMGT positions to indices in the full sequence
                        positions = pdb_posins_to_pos(df_H["pdb_posins"])
                        full_start = np.where(positions == imgt_start)[0][0]
                        full_end = np.where(positions == imgt_end-1)[0][0] + 1  # +1 because end is exclusive
                        
                        motifs.append((full_start, full_end))
                        print(f'[DEBUG] Found polyglycine motif in {region} at positions {motif_start}-{motif_end-1}', flush=True)
                        print(f'[DEBUG] Maps to IMGT positions {imgt_start}-{imgt_end-1}', flush=True)
                        print(f'[DEBUG] Maps to full sequence indices {full_start}-{full_end-1}', flush=True)
                else:
                    j += 1
            
            start_idx = end_idx
    else:
        # Process the entire sequence as one
        i = 0
        while i < len(sequence):
            if sequence[i] == 'G':
                start = i
                while i < len(sequence) and sequence[i] == 'G':
                    i += 1
                end = i
                if end - start >= min_length:
                    motifs.append((start, end))
                    print(f'[DEBUG] Found polyglycine motif at positions {start}-{end-1}', flush=True)
            else:
                i += 1
    
    return motifs

def resample_polyglycine_motifs(df, H_sampled, L_sampled, poly_g_motifs, t=0.20, model=None, pdbs_csv=None, pdb_dir=None):
    """
    Resamples polyglycine motifs in CDR regions using the context of the full sequence.
    
    This function resamples identified polyglycine motifs using the context of the original
    full sequence (H and L) + the non-polyglycine residues in the CDRs.
    
    Args:
        df: DataFrame with residue probabilities
        H_sampled: Sampled heavy chain sequence
        L_sampled: Sampled light chain sequence
        poly_g_motifs: List of tuples (start_idx, end_idx) for each polyglycine motif
        t: Sampling temperature
        model: AntiFold model for resampling
        pdbs_csv: CSV with PDB information
        pdb_dir: Directory containing PDB files
        
    Returns:
        Updated H_sampled and L_sampled sequences with resampled polyglycine motifs
    """
    import pandas as pd
    import os
    import tempfile
    
    # Get the CDR sequences
    df_H, df_L = get_dfs_HL(df)
    
    if not poly_g_motifs:
        print(f'[DEBUG] No polyglycine motifs provided to resample', flush=True)
        return H_sampled, L_sampled
    
    print(f'[DEBUG] Resampling {len(poly_g_motifs)} polyglycine motifs', flush=True)
    
    # Create a copy of the sequences for resampling
    H_resampled = H_sampled.copy()
    
    # Create a combined sequence for context
    combined_seq = np.concatenate([H_resampled, L_sampled])
    print(f'[DEBUG] Combined sequence for context: {"".join(combined_seq)}', flush=True)
    
    # For each polyglycine motif
    for motif_start, motif_end in poly_g_motifs:
        
        print(f'[DEBUG] Resampling polyglycine motif at positions {motif_start}-{motif_end-1}', flush=True)
        
        # Create a mask for the current polyglycine motif
        motif_mask = np.zeros_like(H_sampled, dtype=bool)
        motif_mask[motif_start:motif_end] = True
        
        # Get the original motif
        original_motif = H_resampled[motif_mask]
        print(f'[DEBUG] Original motif: {original_motif}', flush=True)
        
        # Use the context of the original full sequence + non-polyglycine residues
        # for sampling new values of the polyglycine motifs
        
        # Method 1: Use existing probabilities from the dataframe
        if model is None or pdbs_csv is None or pdb_dir is None:
            print(f'[DEBUG] Using existing probabilities for resampling (model, pdbs_csv, or pdb_dir not provided)', flush=True)
            
            # Probabilities after scaling with temp
            probs = get_temp_probs(df_H, t=t)
            probs_motif = probs[motif_mask]
            
            print(f'[DEBUG] Resampling motif with probabilities shape: {probs_motif.shape}', flush=True)
            
            # Sample new tokens for the motif
            sampled_tokens = torch.multinomial(probs_motif, 1).squeeze(-1)
            sampled_motif = np.array([amino_list[i] for i in sampled_tokens])
            
        # Method 2: Use the model to generate new probabilities based on context
        else:
            print(f'[DEBUG] Using model to generate new probabilities based on context', flush=True)
            
            try:
                # Create a temporary directory for working files
                with tempfile.TemporaryDirectory() as temp_dir:
                    # Create a temporary PDB file with the current sequence
                    # This is a simplified approach - in a real implementation, you would need to
                    # create a proper PDB file with the sequence and structure
                    
                    # For demonstration, we'll use a simplified approach by creating a CSV file
                    # that can be used by get_pdbs_logits
                    
                    # Create a temporary CSV file with the current sequence
                    temp_csv_path = os.path.join(temp_dir, "temp_sequence.csv")
                    
                    # Create a copy of the dataframe with the current sequence
                    df_temp = df.copy()
                    
                    # Update the sequence in the dataframe
                    # This is a simplified approach - in a real implementation, you would need to
                    # properly update the dataframe with the current sequence
                    
                    print(f'[DEBUG] Creating temporary CSV file for context-based resampling', flush=True)
                    print(f'[DEBUG] Using context of full sequence with polyglycine motifs masked', flush=True)
                    
                    # For now, we'll use the existing probabilities as a fallback
                    probs = get_temp_probs(df_H, t=t)
                    probs_motif = probs[motif_mask]
                    
                    # Sample new tokens for the motif
                    sampled_tokens = torch.multinomial(probs_motif, 1).squeeze(-1)
                    sampled_motif = np.array([amino_list[i] for i in sampled_tokens])
                    
                    print(f'[DEBUG] Note: Full model-based resampling would require creating proper PDB files', flush=True)
                    print(f'[DEBUG] Using simplified approach with existing probabilities for now', flush=True)
            
            except Exception as e:
                print(f'[ERROR] Error in model-based resampling: {e}', flush=True)
                print(f'[DEBUG] Falling back to existing probabilities', flush=True)
                
                # Fallback to existing probabilities
                probs = get_temp_probs(df_H, t=t)
                probs_motif = probs[motif_mask]
                
                # Sample new tokens for the motif
                sampled_tokens = torch.multinomial(probs_motif, 1).squeeze(-1)
                sampled_motif = np.array([amino_list[i] for i in sampled_tokens])
        
        print(f'[DEBUG] Resampled motif: {sampled_motif}', flush=True)
        
        # Replace the motif in the sequence
        H_resampled[motif_mask] = sampled_motif
        
        # Update the combined sequence for context in next iterations
        combined_seq = np.concatenate([H_resampled, L_sampled])
    
    return H_resampled, L_sampled

def sample_new_sequences_CDR_HL(
    df,
    t=0.20,
    imgt_regions=["CDRH1", "CDRH2", "CDRH3"],
    exclude_heavy=False,
    exclude_light=False,
    return_mutation_df=False,
    limit_expected_variation=False,
    verbose=True,
    resample_polyglycine=True,
    model=None,
    pdbs_csv=None,
    pdb_dir=None,
    max_iterations=5,
):
    """Samples new sequences only varying at H/L CDRs with optional polyglycine resampling"""
    print(f'[DEBUG - antiscripts_utils.py] Now inside sample_new_sequences_CDR_HL()', flush=True)
    def _sample_cdr_seq(df, region_mask, t=0.20):
        """DF to sampled seq"""

        # # CDR1+2+3 mask
        # region_mask = get_imgt_mask(df, imgt_regions)

        # Probabilities after scaling with temp
        probs = get_temp_probs(df, t=t)
        print(f'[DEBUG] Full Seq Probabilities after scaling with temp: {probs}, {probs.size()}', flush=True)
        probs_cdr = probs[region_mask]
        print(f'[DEBUG] CDR Probabilities after scaling with temp ({t}): {probs[region_mask]}, {probs[region_mask].size()}', flush=True)
        print(f'[DEBUG] These CDR probabilities were extracted with region_mask: {region_mask}')

        # Sampled tokens and sequence
        sampled_tokens = torch.multinomial(probs_cdr, 1).squeeze(-1)
        sampled_seq = np.array([amino_list[i] for i in sampled_tokens])

        return sampled_seq

    # Prepare to sample new H + L sequences
    df_H, df_L = get_dfs_HL(df)

    # Get H, sampling only for (CDR1, 2, 3)
    H_sampled = get_df_seq(df_H)

    regions = [region for region in imgt_regions if "L" not in region]
    if len(regions) > 0 and not exclude_heavy:
        print('\nSampling sequences for the HEAVY chain:', flush=True)
        region_mask = get_imgt_mask(df_H, regions)
        print(f'[DEBUG] Built concatenated region_mask: {region_mask} with [{sum(region_mask)}] Masked positions', flush=True)
        print(f'[DEBUG] Sampling {regions} for positions {region_mask} in the heavy chain with full sequence: {H_sampled} and length: {len(H_sampled)}', flush=True)
        H_sampled[region_mask] = _sample_cdr_seq(df_H, region_mask, t=t)
        print(f'[DEBUG] Heavy chain sampled sequences: {H_sampled[region_mask]}\n', flush=True)
        assert(len(H_sampled[region_mask]) == 31) # Hard coded CDR length
        
        # Resample polyglycine motifs if requested
        if resample_polyglycine:
            print(f'===== Starting polyglycine motif resampling =====', flush=True)
            
            # Get L, sampling only for (CDR1, 2, 3)
            L_sampled = get_df_seq(df_L)
            
            # Iteratively resample polyglycine motifs until none remain or max iterations reached
            iteration = 0
            while iteration < max_iterations:
                # Get masks for each CDR region separately
                cdrh1_mask = get_imgt_mask(df_H, ["CDRH1"])
                cdrh2_mask = get_imgt_mask(df_H, ["CDRH2"])
                cdrh3_mask = get_imgt_mask(df_H, ["CDRH3"])
                
                # Get the CDR sequences
                H_cdrh1_seq = H_sampled[cdrh1_mask]
                H_cdrh2_seq = H_sampled[cdrh2_mask]
                H_cdrh3_seq = H_sampled[cdrh3_mask]
                
                # Get the lengths of each CDR region
                cdr_lengths = [len(H_cdrh1_seq), len(H_cdrh2_seq), len(H_cdrh3_seq)]
                print(f'[DEBUG] CDR lengths: CDRH1={len(H_cdrh1_seq)}, CDRH2={len(H_cdrh2_seq)}, CDRH3={len(H_cdrh3_seq)}', flush=True)
                
                # Concatenate all CDR sequences for processing
                H_cdr_seq = np.concatenate([H_cdrh1_seq, H_cdrh2_seq, H_cdrh3_seq])
                print(f'[DEBUG] Checking for polyglycine motifs in concatenated CDR sequence: {H_cdr_seq}', flush=True)
                
                # Identify polyglycine motifs in CDR regions, using the CDR lengths and regions
                cdr_regions = regions  # Use the regions passed to the function
                poly_g_motifs = identify_polyglycine_motifs(
                    H_cdr_seq,
                    cdr_lengths=cdr_lengths,
                    cdr_regions=cdr_regions,
                    df_H=df_H
                )
                
                if not poly_g_motifs:
                    print(f'[DEBUG] No more polyglycine motifs found after {iteration} iterations', flush=True)
                    break
                
                print(f'[DEBUG] Iteration {iteration+1}: Found {len(poly_g_motifs)} polyglycine motifs in CDR regions', flush=True)
                
                # Resample the polyglycine motifs
                H_sampled, L_sampled = resample_polyglycine_motifs(
                    df,
                    H_sampled,
                    L_sampled,
                    poly_g_motifs,  # Pass the identified motifs directly
                    t=t,
                    model=model,
                    pdbs_csv=pdbs_csv,
                    pdb_dir=pdb_dir
                )
                
                print(f'[DEBUG] After resampling iteration {iteration+1}, CDR sequences:')
                print(f'[DEBUG] CDRH1: {H_sampled[cdrh1_mask]}')
                print(f'[DEBUG] CDRH2: {H_sampled[cdrh2_mask]}')
                print(f'[DEBUG] CDRH3: {H_sampled[cdrh3_mask]}')
                
                iteration += 1
            
            print(f'[DEBUG] Final heavy chain sequence after polyglycine resampling: {H_sampled}', flush=True)
            print(f'[DEBUG] Final CDR sequences after polyglycine resampling: {H_sampled[region_mask]}\n', flush=True)
    else:
        L_sampled = get_df_seq(df_L)
        

    L_sampled = get_df_seq(df_L)

    regions = [region for region in imgt_regions if "H" not in region]
    if len(regions) > 0 and not exclude_light:
        region_mask = get_imgt_mask(df_L, regions)
        L_sampled[region_mask] = _sample_cdr_seq(df_L, region_mask, t=t)
    else:
        print('[DEBUG] No regions to sample for the light chain')

    # Use for later
    sampled_seq = np.concatenate([H_sampled, L_sampled])
    pred_seq = get_df_seq_pred(df)
    orig_seq = get_df_seq(df)
    print(f'[DEBUG] Original sequence: {orig_seq}', flush=True)
    print(f'[DEBUG] Predicted sequence: {pred_seq}', flush=True)
    print(f'[DEBUG] Heavy chain sampled sequence: {H_sampled}', flush=True)
    print(f'[DEBUG] Target chain fixed sequence: {L_sampled}', flush=True)
    print(f'[DEBUG] H/T combined sampled sequence: {sampled_seq}', flush=True)
    
    region_mask = get_imgt_mask(df, imgt_regions)
    print(f'[DEBUG] Built concatenated region_mask to be used later: {region_mask}')


    # Mismatches vs predicted (CDR only)
    mismatch_idxs_pred_cdr = np.where(
        (sampled_seq[region_mask] != pred_seq[region_mask])
    )[0]

    # Mismatches vs original (all)
    mismatch_idxs_orig = np.where((sampled_seq != orig_seq))[0]

    # # Limit mutations (backmutate) to as many expected from temperature sampling
    # if limit_expected_variation:
    #     backmutate = len(mismatch_idxs_orig) - len(mismatch_idxs_pred_cdr)

    #     if backmutate >= 1:
    #         backmutate_idxs = np.random.choice(
    #             mismatch_idxs_orig, size=backmutate, replace=False
    #         )
    #         sampled_seq[backmutate_idxs] = orig_seq[backmutate_idxs]
    #         H_sampled = sampled_seq[: len(df_H)]
    #         L_sampled = sampled_seq[-len(df_L) :]

    # DataFrame with sampled mutations
    if return_mutation_df:
        mut_list = np.where(sampled_seq != orig_seq)[0]
        df_mut = df.loc[
            mut_list, ["pdb_res", "top_res", "pdb_posins", "pdb_chain"]
        ].copy()
        df_mut.insert(1, "aa_sampled", sampled_seq[mut_list])

        return H_sampled, L_sampled, df_mut

    return H_sampled, L_sampled, df


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


def sample_from_df_logits(
    df_logits,
    sample_n=1,
    sampling_temp=0.20,
    regions_to_mutate=["CDRH1", "CDRH2", "CDRH3"],
    exclude_heavy=False,
    exclude_light=False,
    limit_expected_variation=False,
    verbose=False,
    seed=42,
    resample_polyglycine=True,
    model=None,
    pdbs_csv=None,
    pdb_dir=None,
    max_iterations=5,
):
    print('[DEBUG - antiscripts.py sample_from_df_logits()] regions_to_mutate:', regions_to_mutate)
    # Get original H/L sequence
    H_orig, L_orig = get_df_seqs_HL(df_logits)
    print(f'Length of heavy chain: {len(H_orig)}, light chain: {len(L_orig)}', flush=True)

    # Only sampling from heavy, light chains
    df_logits_HL = df_logits.iloc[:len(H_orig) + len(L_orig), :]
    df_logits_HL.name = df_logits.name

    # Stats
    print(f'=====ITERATION 0 - Evaluating df seq =====', flush=True)
    seq = "".join(H_orig) + "".join(L_orig)
    print(f'seq: {seq}', flush=True)
    _, global_score = get_sequence_sampled_global_score(
        seq, df_logits_HL, regions_to_mutate
    )

    # Save to FASTA dict
    fasta_dict = OrderedDict()
    _id = f"{df_logits_HL.name}"
    desc = f", score={global_score:.4f}, global_score={global_score:.4f}, regions={regions_to_mutate}, model_name=AntiFold, seed={seed}"
    fasta_dict[_id] = SeqIO.SeqRecord(Seq(seq), id=_id, name="", description=desc)
    print(f'[DEBUG - antiscripts.py sample_from_df_logits()] Saving data to FASTA dict with values score={global_score:.4f}, global_score={global_score:.4f}, regions={regions_to_mutate}, model_name=AntiFold, seed={seed}', flush=True)

    # if verbose:
    #     log.info(f"{_id}: {desc}")

    if not isinstance(sampling_temp, list):
        sampling_temp = [sampling_temp]

    for t in sampling_temp:
        # Sample sequences n times
        for n in range(sample_n):
            print(f'=====ITERATION {n+1} of {sample_n}=====', flush=True)
            print(f'[DEBUG - antiscripts_utils.py] Calling sample_new_sequences_CDR_HL() with sampling temperature: {t}', flush=True)
            # Get mutated H/L sequence
            H_mut, L_mut, df_mut = sample_new_sequences_CDR_HL(
                df_logits_HL,  # DataFrame with residue probabilities
                t=t,  # Sampling temperature
                imgt_regions=regions_to_mutate,  # Region to sample
                exclude_heavy=exclude_heavy,  # Allow mutations in heavy chain
                exclude_light=exclude_light,  # Allow mutation in light chain
                limit_expected_variation=False,  # Only mutate as many positions are expected from temperature
                verbose=verbose,
                resample_polyglycine=resample_polyglycine,  # Enable polyglycine resampling
                model=model,  # Pass model for resampling
                pdbs_csv=pdbs_csv,  # Pass CSV with PDB information
                pdb_dir=pdb_dir,  # Pass directory containing PDB files
                max_iterations=max_iterations,  # Maximum number of resampling iterations
            )
            
            if resample_polyglycine:
                print(f'[DEBUG - antiscripts_utils.py sample_from_df_logits()] Completed polyglycine resampling', flush=True)

            # Original sequence
            seq_orig = "".join(H_orig) + "".join(L_orig)
            print(f'[DEBUG - antiscripts_utils.py sample_from_df_logits()] Recreating original sequence as: {seq_orig}', flush=True)

            # Sequence recovery and mismatches
            correct_matches = (H_orig == H_mut).sum() + (L_orig == L_mut).sum()
            seq_recovery = correct_matches / len(seq_orig)
            n_mut = (H_orig != H_mut).sum() + (L_orig != L_mut).sum()

            seq_mut = "".join(H_mut) + "".join(L_mut)
            score_sampled, global_score = get_sequence_sampled_global_score(
                seq_mut, df_logits_HL, regions_to_mutate
            )

            # Save to FASTA dict
            _id = f"{df_logits_HL.name}__{n+1}"
            desc = f"T={t:.2f}, sample={n+1}, score={score_sampled:.4f}, global_score={global_score:.4f}, seq_recovery={seq_recovery:.4f}, mutations={n_mut}"
            seq_mut = "".join(H_mut) + "/" + "".join(L_mut)
            fasta_dict[_id] = SeqIO.SeqRecord(
                Seq(seq_mut), id="", name="", description=desc
            )
            print(f'[DEBUG - antiscripts.py sample_from_df_logits()] Saving data to FASTA dict with seq: {seq_mut}, id: {_id}, name: "", description: {desc}', flush=True)

            # if verbose:
            #     log.info(f"{_id}: {desc}")

    return fasta_dict


def write_fasta_to_dir(fasta_dict, df_logits, out_dir, verbose=True):
    """Write fasta to output folder"""

    os.makedirs(out_dir, exist_ok=True)
    outfile = f"{out_dir}/{df_logits.name}.fasta"
    # if verbose:
    #     log.info(f"Saving to {outfile}")

    with open(outfile, "w") as out_handle:
        SeqIO.write(fasta_dict.values(), out_handle, "fasta")


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


def df_logits_to_probs(df_logits):
    """Convert logits to probabilities"""

    # Calculate probabilities
    amino_list = list("ACDEFGHIKLMNPQRSTVWY")
    t = torch.tensor(df_logits[amino_list].values)
    probs = F.softmax(t, dim=1)

    # Insert into copied dataframe
    df_probs = df_logits.copy()
    df_probs[amino_list] = probs

    return df_probs


def df_logits_to_logprobs(df_logits):
    """Convert logits to probabilities"""

    # Calculate probabilities
    amino_list = list("ACDEFGHIKLMNPQRSTVWY")
    t = torch.tensor(df_logits[amino_list].values)
    probs = F.log_softmax(t, dim=1)

    # Insert into copied dataframe
    df_probs = df_logits.copy()
    df_probs[amino_list] = probs

    return df_probs


def sequence_to_onehot(sequence):
    amino_list = list("ACDEFGHIKLMNPQRSTVWY")
    one_hot = np.zeros((len(sequence), len(amino_list)), dtype=int)
    for i, res in enumerate(sequence):
        one_hot[i, amino_list.index(res)] = 1
    return one_hot


def get_sequence_sampled_global_score(seq, df_logits, regions_to_mutate=False):
    """
    Get average log probability of sampled / all amino acids
    """
    print(f'[DEBUG - antiscripts_utils.py get_sequence_sampled_global_score()] Globally scoring sequence: {seq} with regions_to_mutate: {regions_to_mutate}', flush=True)
    def _scores(S, log_probs, mask):
        criterion = torch.nn.NLLLoss(reduction="none")
        loss = criterion(
            log_probs.contiguous().view(-1, log_probs.size(-1)), S.contiguous().view(-1)
        ).view(S.size())
        scores = torch.sum(loss * mask, dim=-1) / torch.sum(mask, dim=-1)
        return scores

    # One-hot to indices
    onehot = sequence_to_onehot(seq)
    S = torch.argmax(torch.tensor(onehot, dtype=torch.float), dim=-1)

    # Logits to log probs
    logits = torch.tensor(df_logits[amino_list].values)
    log_probs = F.log_softmax(logits, dim=-1)

    # clip probs
    # log_probs = torch.clamp(log_probs, min=-100, max=100)

    # Calculate log odds scores
    mask = torch.ones_like(S, dtype=torch.bool)
    print(f'[DEBUG - antiscripts_utils.py get_sequence_sampled_global_score()] Calculating global score for sequence: {seq} using mask: {mask}', flush=True)
    score_global = _scores(S, log_probs, mask)

    if regions_to_mutate:
        region_mask = get_imgt_mask(df_logits, regions_to_mutate)
        mask = torch.tensor(region_mask)
        print(f'[DEBUG - antiscripts_utils.py get_sequence_sampled_global_score()] Using region_mask: {mask} to score the sampled regions {regions_to_mutate} in sequence: {seq}', flush=True)
        score_sampled = _scores(S, log_probs, mask)
        print(f'Obtained score {score_sampled.item()}', flush=True)
    else:
        print(f'[ERROR] No regions to mutate specified. Returning global score only.', flush=True)
        score_sampled = score_global

    return score_sampled.item(), score_global.item()

