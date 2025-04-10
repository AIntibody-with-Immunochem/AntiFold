import numpy as np
import torch
import pandas as pd
import os
import tempfile

from antifold.general_utils import (IMGT_dict, amino_list, pdb_posins_to_pos, get_dfs_HL,
                            get_temp_probs, get_df_logits)
from antifold.antiscripts import (get_dataset_dataloader, dataset_dataloader_to_predictions_list,
                           predictions_list_to_df_logits_list)

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

def get_pdbs_logits_for_polyg(
    model,
    original_pdb_csv,
    pdb_dir,
    H_sampled,
    L_sampled,
    poly_g_motifs,
    batch_size=1,
    custom_chain_mode=False,
    num_threads=0,
    seed=42,
    gaussian_noise=True,  # Enable Gaussian noise by default for data augmentation
    gaussian_scale=0.1,   # Default scale for Gaussian noise
):
    """
    Get logits for a sequence with polyglycine motifs.
    
    This function uses the AntiFold model to generate logits for the sequence
    while keeping the original PDB structure. It uses a custom sequence (combined_seq)
    instead of extracting the sequence from the PDB file.
    
    Args:
        model: AntiFold model for resampling
        original_pdb_csv: CSV with PDB information for the original structure
        pdb_dir: Directory containing PDB files
        H_sampled: Current sampled heavy chain sequence
        L_sampled: Current sampled light chain sequence
        poly_g_motifs: List of tuples (start_idx, end_idx) for each polyglycine motif
        batch_size: Batch size for model inference
        custom_chain_mode: Whether to use custom chain mode
        num_threads: Number of threads to use
        seed: Random seed
        Returns:
            DataFrame with logits for the sequence and a mask for the polyglycine motifs
            
        Note:
            When gaussian_noise is True, Gaussian noise will be added to the coordinates
            for data augmentation, which can help improve the robustness of the model.
        DataFrame with logits for the sequence and a mask for the polyglycine motifs
    """
    print(f'[DEBUG] Getting logits for sequence with polyglycine motifs', flush=True)
    
    # Create a mask for all polyglycine motifs in the heavy chain
    mask = np.zeros_like(H_sampled, dtype=bool)
    for motif_start, motif_end in poly_g_motifs:
        mask[motif_start:motif_end] = True
    
    print(f'[DEBUG] Created mask for polyglycine motifs: {mask}', flush=True)
    print(f'[DEBUG] Number of masked positions: {np.sum(mask)}', flush=True)
    
    # Get the combined sequence
    combined_seq = np.concatenate([H_sampled, L_sampled])
    combined_seq_str = ''.join(combined_seq)  # Convert to string for better compatibility
    print(f'[DEBUG] Combined sequence: {combined_seq_str}', flush=True)
    
    # Load PDBs
    print(f'[DEBUG] Loading dataset and dataloader for original PDB structure with custom sequence', flush=True)
    print(f'[DEBUG] Custom sequence: {combined_seq_str}', flush=True)
    # Add debug message for Gaussian noise
    if gaussian_noise:
        print(f'[DEBUG] Enabling Gaussian noise for data augmentation with scale {gaussian_scale}', flush=True)
    else:
        print(f'[DEBUG] Gaussian noise disabled for data augmentation', flush=True)
        
    dataset, dataloader = get_dataset_dataloader(
        original_pdb_csv,
        pdb_dir,
        batch_size=batch_size,
        custom_chain_mode=custom_chain_mode,
        num_threads=num_threads,
        custom_sequence=combined_seq_str,  # Pass the string version of the combined sequence
        gaussian_noise_flag=gaussian_noise,  # Pass the Gaussian noise flag
        gaussian_scale_A=gaussian_scale,    # Pass the Gaussian noise scale
    )
    
    # Get predictions from the model
    print(f'[DEBUG] Getting predictions from model', flush=True)
    predictions_list, _ = dataset_dataloader_to_predictions_list(
        model,
        dataset,
        dataloader,
        batch_size=batch_size,
        extract_embeddings=False,
    )
    
    # Convert predictions to dataframes
    print(f'[DEBUG] Converting predictions to dataframes', flush=True)
    df_logits_list = predictions_list_to_df_logits_list(
        predictions_list, dataset, dataloader
    )
    
    # We only need the first dataframe since we're only processing one PDB
    df_logits = df_logits_list[0]
    
    return df_logits, mask

def resample_polyglycine_motifs(df, H_sampled, L_sampled, poly_g_motifs, t=0.20, model=None, pdbs_csv=None, pdb_dir=None, gaussian_noise=True, gaussian_scale=0.1):
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
        model: AntiFold model for resampling (required)
        pdbs_csv: CSV with PDB information (required)
        pdb_dir: Directory containing PDB files (required)
        
    Returns:
        Updated H_sampled and L_sampled sequences with resampled polyglycine motifs
        
    Note:
        When gaussian_noise is True, Gaussian noise will be added to the coordinates
        for data augmentation, which can help improve the robustness of the model.
    """
    # Get the CDR sequences
    df_H, df_L = get_dfs_HL(df)
    
    if not poly_g_motifs:
        print(f'[DEBUG] No polyglycine motifs provided to resample', flush=True)
        return H_sampled, L_sampled
    
    # Check that required parameters are provided
    if model is None or pdbs_csv is None or pdb_dir is None:
        raise ValueError("Model, pdbs_csv, and pdb_dir are required for polyglycine resampling")
    
    print(f'[DEBUG] Resampling {len(poly_g_motifs)} polyglycine motifs', flush=True)
    
    # Create a copy of the sequences for resampling
    H_resampled = H_sampled.copy()
    
    # Create a combined sequence for context
    combined_seq = np.concatenate([H_resampled, L_sampled])
    print(f'[DEBUG] Combined sequence for context: {"".join(combined_seq)}', flush=True)
    
    print(f'[DEBUG] Using AntiFold model for context-based resampling', flush=True)
    
    # Get logits for the sequence
    print(f'[DEBUG] Using Gaussian noise: {gaussian_noise}, scale: {gaussian_scale}', flush=True)
    df_logits, mask = get_pdbs_logits_for_polyg(
        model=model,
        original_pdb_csv=pdbs_csv,
        pdb_dir=pdb_dir,
        H_sampled=H_sampled,
        L_sampled=L_sampled,
        poly_g_motifs=poly_g_motifs,
        batch_size=1,
        custom_chain_mode=False,
        num_threads=0,
        seed=42,
        gaussian_noise=gaussian_noise,
        gaussian_scale=gaussian_scale,
    )
    
    # Get the heavy chain logits
    df_H_logits, _ = get_dfs_HL(df_logits)
    
    # For each polyglycine motif
    for motif_start, motif_end in poly_g_motifs:
        print(f'[DEBUG] Resampling polyglycine motif at positions {motif_start}-{motif_end-1}', flush=True)
        
        # Create a mask for the current polyglycine motif
        motif_mask = np.zeros_like(H_sampled, dtype=bool)
        motif_mask[motif_start:motif_end] = True
        
        # Get the original motif
        original_motif = H_resampled[motif_mask]
        print(f'[DEBUG] Original motif: {original_motif}', flush=True)
        
        # Get probabilities for the motif from the model logits
        probs = get_temp_probs(df_H_logits, t=t)
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
