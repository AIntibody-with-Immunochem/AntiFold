import logging
import os
import sys
# import warnings
import urllib.request
from pathlib import Path

ROOT_PATH = Path(os.path.dirname(__file__)).parent
sys.path.insert(0, ROOT_PATH)

from argparse import ArgumentParser, RawTextHelpFormatter

import numpy as np
import pandas as pd

# from antifold.antiscripts import (df_logits_to_logprobs,
#                                   extract_chains_biotite, generate_pdbs_csv,
#                                   get_pdbs_logits, load_model,
#                                   sample_from_df_logits, write_fasta_to_dir,
#                                   visualize_mutations)

from antifold.antiscripts_utils import (
                                  write_fasta_to_dir,
                                   sample_from_df_logits)
from antifold.antiscripts import(df_logits_to_logprobs, generate_pdbs_csv, get_pdbs_logits)
    
from antifold.general_utils import (extract_chains_biotite, load_model)
log = logging.getLogger(__name__)


def cmdline_args():
    # Make parser object
    usage = f"""
# Run AntiFold on single PDB (or CIF) file
python antifold/main.py \
    --out_dir output/single_pdb \
    --pdb_file data/pdbs/6y1l_imgt.pdb \
    --heavy_chain H \
    --light_chain L

# Run AntiFold on an antibody-antigen complex (enables custom_chain_mode)
python antifold/main.py \
    --out_dir output/antibody_antigen \
    --pdb_file data/antibody_antigen/3hfm.pdb \
    --heavy_chain H \
    --light_chain L \
    --antigen_chain Y

# Run AntiFold on a folder of PDB/CIFs (specify chains to run in CSV file)
# and consider extra antigen chains
python antifold/main.py \
    --out_dir output/antibody_antigen \
    --pdbs_csv data/antibody_antigen.csv \
    --pdb_dir data/antibody_antigen \
    --custom_chain_mode
    """
    p = ArgumentParser(
        description="Predict antibody variable domain inverse folding probabilities and sample sequences with maintained fold.\nPDB structures should be IMGT-numbered, paired heavy and light chain variable domains (positions 1-128).\n\nFor IMGT numbering PDBs use SAbDab or https://opig.stats.ox.ac.uk/webapps/sabdab-sabpred/sabpred/anarci/",
        formatter_class=RawTextHelpFormatter,
        usage=usage,
    )

    def is_valid_path(parser, arg):
        if not os.path.exists(arg):
            parser.error(f"Path {arg} does not exist!")
        else:
            return arg

    def is_valid_dir(parser, arg):
        if not os.path.isdir(arg):
            parser.error(f"Directory {arg} does not exist!")
        else:
            return arg

    p.add_argument(
        "--pdb_file",
        help="Input PDB file (for single PDB predictions)",
        type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--nanobody_chain", help="Antibody nanobody chain (for single PDB predictions)",
    )

    p.add_argument(
        "--heavy_chain", help="Antibody heavy chain (for single PDB predictions)",
    )

    p.add_argument(
        "--light_chain", help="Antibody light chain (for single PDB predictions)",
    )

    p.add_argument(
        "--antigen_chain", help="Antigen chain (optional)",
    )

    p.add_argument(
        "--pdbs_csv",
        help="Input CSV file with PDB names and H/L chains (multi-PDB predictions)",
        type=lambda x: is_valid_path(p, x),
    )

    p.add_argument(
        "--pdb_dir",
        help="Directory with input PDB files (multi-PDB predictions)",
        type=lambda x: is_valid_dir(p, x),
    )

    p.add_argument(
        "--out_dir", default="antifold_output", help="Output directory",
    )

    p.add_argument(
        "--regions",
        default="CDR1 CDR2 CDR3",
        help="Space-separated regions to mutate. Default 'CDR1 CDR2 CDR3H'",
    )

    p.add_argument(
        "--num_seq_per_target",
        default=0,
        type=int,
        help="Number of sequences to sample from each antibody PDB (default 0)",
    )

    p.add_argument(
        "--sampling_temp",
        default="0.20",
        help="A string of temperatures e.g. '0.20 0.25 0.50' (default 0.20). Sampling temperature for amino acids. Suggested values 0.10, 0.15, 0.20, 0.25, 0.30. Higher values will lead to more diversity.",
    )

    p.add_argument(
        "--limit_variation",
        default=False,
        action="store_true",
        help="Limit variation to as many mutations as expected from temperature sampling",
    )

    p.add_argument(
        "--extract_embeddings",
        default=False,
        action="store_true",
        help="Extract per-residue embeddings from AntiFold / ESM-IF1",
    )

    p.add_argument(
        "--custom_chain_mode",
        default=False,
        action="store_true",
        help="Run all specified chains (for antibody-antigen complexes or any combination of chains)",
    )

    p.add_argument(
        "--exclude_heavy", action="store_true", help="Exclude heavy chain from sampling"
    )

    p.add_argument(
        "--exclude_light", action="store_true", help="Exclude light chain from sampling"
    )

    p.add_argument(
        "--batch_size", default=1, type=int, help="Batch-size to use",
    )

    p.add_argument(
        "--num_threads",
        default=0,
        type=int,
        help="Number of CPU threads to use for parallel processing (0 = all available)",
    )

    p.add_argument(
        "--seed", default=42, type=int, help="Seed for reproducibility",
    )
    
    p.add_argument(
        "--resample_polyglycine",
        default=True,
        action="store_true",
        help="Resample polyglycine motifs in CDR regions",
    )
    
    p.add_argument(
        "--max_polyglycine_iterations",
        default=5,
        type=int,
        help="Maximum number of iterations for polyglycine resampling",
    )

    p.add_argument(
        "--model_path",
        default="",
        help="Alternative model weights (default models/model.pt). See --use_esm_if1_weights flag to use ESM-IF1 weights instead of AntiFold",
    )

    p.add_argument(
        "--esm_if1_mode",
        default=False,
        action="store_true",
        help="Use ESM-IF1 weights instead of AntiFold",
    )

    p.add_argument(
        "--verbose", default=1, type=int, help="Verbose printing",
    )

    return p.parse_args()


def sample_pdbs(
    model,
    pdbs_csv_or_dataframe,
    regions_to_mutate,
    pdb_dir="data/pdbs",
    out_dir="output/sampled",
    sample_n=10,
    sampling_temp=0.50,
    limit_expected_variation=False,
    exclude_heavy=False,
    exclude_light=False,
    batch_size=1,
    extract_embeddings=False,
    custom_chain_mode=False,
    num_threads=0,
    seed=42,
    save_flag=False,
    resample_polyglycine=True,
    max_polyglycine_iterations=5,
):
    print(f'[DEBUG - main.py sample_pdbs()] regions to mutate: {regions_to_mutate}', flush=True)
    print(f'[DEBUG - main.py sample_pdbs()] pdb_dir: {pdb_dir}', flush=True)
    print(f'[DEBUG - main.py sample_pdbs()] sampling_temp: {sampling_temp}', flush=True)
    print(f'[DEBUG - main.py sample_pdbs()] pdbs_csv_or_dataframe: {pdbs_csv_or_dataframe}', flush=True)
    print(f'[DEBUG - main.py sample_pdbs()] batch_size: {batch_size}', flush=True)
    print(f'[DEBUG - main.py sample_pdbs()] custom_chain_mode: {custom_chain_mode}', flush=True)
    
    # Predict with CSV on folder of solved (SAbDab) structures
    try:
        print('[DEBUG - main.py sample_pdbs()] About to call get_pdbs_logits()', flush=True)
        df_logits_list = get_pdbs_logits(
            model=model,
            pdbs_csv_or_dataframe=pdbs_csv_or_dataframe,
            pdb_dir=pdb_dir,
            out_dir=out_dir,
            save_flag=save_flag,
            batch_size=batch_size,
            extract_embeddings=extract_embeddings,
            custom_chain_mode=custom_chain_mode,
            seed=seed,
            num_threads=num_threads,
        )
        print('[DEBUG - main.py sample_pdbs()] Successfully called get_pdbs_logits()', flush=True)
        print(f'Output logits: {df_logits_list[0]}', flush=True)
    except Exception as e:
        print(f'[ERROR - main.py sample_pdbs()] Exception in get_pdbs_logits(): {e}', flush=True)
        raise

    if sample_n >= 1:
        # Sample from output probabilities
        pdb_output_dict = {}
        for df_logits in df_logits_list:
            # Sample 10 sequences with a temperature of 0.50
            fasta_dict = sample_from_df_logits(
                df_logits,
                sample_n=sample_n,
                sampling_temp=sampling_temp,
                regions_to_mutate=regions_to_mutate,
                limit_expected_variation=False,
                verbose=True,
                seed=seed,
                resample_polyglycine=resample_polyglycine,  # Use parameter
                model=model,  # Pass model for resampling
                pdbs_csv=pdbs_csv_or_dataframe,  # Pass CSV with PDB information
                pdb_dir=pdb_dir,  # Pass directory containing PDB files
                max_iterations=max_polyglycine_iterations,  # Use parameter
            )
            
            print(f'[DEBUG - main.py sample_pdbs()] Completed polyglycine resampling', flush=True)
            
            print(f'[DEBUG] fasta_dict: {fasta_dict}')
            
            pdb_output_dict[df_logits.name] = {
                "sequences": fasta_dict,
                "logits": df_logits,
                "logprobs": df_logits_to_logprobs(df_logits),
            }

            # Write to file
            if save_flag:
                write_fasta_to_dir(fasta_dict, df_logits, out_dir=out_dir)

        return pdb_output_dict


def check_valid_input(args):
    """Checks for valid arguments"""

    # Check valid input files input arguments
    # Check either: PDB file, PDB dir or PDBs CSV inputted
    if not (args.pdb_file or args.pdb_dir):
        log.error(
            f"""Please choose one of:
        1) PDB file (--pdb_file). We heavily recommend specifying --heavy_chain [letter] and --light_chain [letter]
        2) PDB directory (--pdb_dir) and CSV file (--pdbs_csv) with columns for PDB names (pdb), H (Hchain) and L (Lchain) chains
        3) PDB directory (--pdb_dir). Warning: Will assume 1st chain is heavy, 2nd chain is light
        """
        )
        sys.exit(1)

    # # Check that AntiFold weights are downloaded
    # root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    # filename = "models/model.pt"
    # model_path = f"{root_dir}/{filename}"
    # if not os.path.exists(model_path):
    #     log.warning(
    #         f"Downloading AntiFold model weights from https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt to {model_path}"
    #     )
    #     url = "https://opig.stats.ox.ac.uk/data/downloads/AntiFold/models/model.pt"

    #     os.makedirs(f"{root_dir}/models", exist_ok=True)
    #     urllib.request.urlretrieve(url, filename)

    # Option 1: PDB file, check heavy and light chain
    if args.pdb_file:
        if not args.heavy_chain and not args.nanobody_chain:
            _pdb = os.path.splitext(os.path.basename(args.pdb_file))[0]
            log.warning(
                f"WARNING: Heavy/light chain(s) not specified for {_pdb}. Assuming 1st chain heavy, 2nd chain light."
            )
            log.warning(
                f"WARNING: Specify manually with e.g. --heavy_chain H --light_chain L"
            )

    # Option 2: Check PDBs in PDB dir and CSV formatted correctly
    elif args.pdb_dir and args.pdbs_csv:

        # Run all chains specified in the CSV file
        args.custom_chain_mode = True

        # Check CSV formatting
        df = pd.read_csv(args.pdbs_csv, comment="#")
        if (
            not df.columns.isin(["pdb", "Hchain", "Lchain"]).sum() >= 3
            and not args.custom_chain_mode
        ):
            log.error(
                f"Multi-PDB input: Please specify CSV  with columns ['pdb', 'Hchain', 'Lchain'] with PDB names (no extension), H and L chains"
            )
            log.error(f"CSV columns: {df.columns}")
            sys.exit(1)

        # Check PDBs exist
        missing = 0
        for i, _pdb in enumerate(df["pdb"].values):
            pdb_path = f"{args.pdb_dir}/{_pdb}.pdb"

            # Check for PDB/CIF file
            pdb_path = (
                pdb_path if os.path.exists(pdb_path) else f"{args.pdb_dir}/{_pdb}.cif"
            )

            if not os.path.exists(pdb_path):
                log.warning(
                    f"WARNING: Unable to find PDB/CIF file ({missing+1}): {pdb_path}"
                )
                missing += 1

        if missing >= 1:
            log.error(
                f"WARNING: Missing {missing} PDB/CIFs specified in {args.pdbs_csv} but not found in {args.pdb_dir}"
            )
            sys.exit(1)

    # Option 3: PDB directory only, infer chains
    elif args.pdb_dir:
        _dir = os.path.dirname(args.pdb_dir)
        log.warning(
            f"WARNING: Heavy/light chains not specified for PDB/CIF files in folder {_dir}. Assuming 1st chain heavy, 2nd chain light."
        )
        log.warning(f"WARNING: Specify manually with --pdbs_csv CSV file")

    # ESM-IF1 mode
    if args.esm_if1_mode:
        args.model_path = "ESM-IF1"
        args.custom_chain_mode = True

        if args.out_dir == "antifold_output":
            args.out_dir = "esmif1_output"

        #log.info(
        #    f"NOTE: ESM-IF1 mode enabled, will use ESM-IF1 weights and run all specified chains"
        #)


def main(args):
    """Predicts antibody heavy and light chain inverse folding probabilities"""

    # Create output directory
    os.makedirs(args.out_dir, exist_ok=True)

    # Try reading in regions
    regions_to_mutate = []
    for region in args.regions.split(" "):
        # Either interpret as positions (ints)
        try:
            regions_to_mutate.append(int(region))
        # Or as regions (strings)
        except ValueError:
            regions_to_mutate.append(region)

    # Try reading in sampling temperatures
    try:
        args.sampling_temp = [float(t) for t in args.sampling_temp.split(" ")]
    except ValueError:
        raise Exception(
            "Sampling temperature must be a float or space-separated floats, e.g. '0.20 0.25 0.50'"
        )

    # No chains provided: assume 1st chain is heavy, 2nd is light
    # Option 1: Single PDB
    if args.pdb_file:

        _pdb = os.path.splitext(os.path.basename(args.pdb_file))[0]

        # Nanobody requires custom_chain_mode
        if args.nanobody_chain:
            print('[DEBUG - main.py] Using a Nanobody PDB File. Setting custom_chain_mode to True')
            args.custom_chain_mode = True
            args.heavy_chain = args.nanobody_chain

        # No chains specified, assume 1st heavy, 2nd light (unless single-chain mode)
        elif not args.heavy_chain:
            args.heavy_chain, args.light_chain = extract_chains_biotite(args.pdb_file)[
                :2
            ]
            log.warning(
                f"{_pdb}: assuming heavy_chain {args.heavy_chain}, light_chain {args.light_chain}"
            )

        pdbs_csv = pd.DataFrame(
            {"pdb": _pdb, "Hchain": args.heavy_chain, "Lchain": args.light_chain,},
            index=[0],
        )

        # Single heavy chain requires custom_chain_mode
        if args.heavy_chain and not args.light_chain:
            args.custom_chain_mode = True

        # Antigen chain requires custom_chain_mode
        if args.antigen_chain:
            print('[DEBUG - main.py] Using an Antigen Chain.')
            pdbs_csv.loc[0, "Agchain"] = args.antigen_chain
            args.custom_chain_mode = True

        # PDB dir is the directory of the PDB file
        pdb_dir = os.path.dirname(args.pdb_file)
        print(f'[DEBUG - main.py] PDB dir: {pdb_dir}')

    # # Option 2: PDB dir and CSV file
    # elif args.pdb_dir and args.pdbs_csv:
    #     pdbs_csv = pd.read_csv(args.pdbs_csv, comment="#")
    #     pdb_dir = args.pdb_dir

    # # Option 3: PDB dir and no CSV file (infer chains)
    # else:
    #     pdb_dir = args.pdb_dir

    #     # custom_chain_mode consider all (10) chains in file
    #     if args.custom_chain_mode:
    #         pdbs_csv = generate_pdbs_csv(args.pdb_dir, max_chains=10)

    #     # Other only consider 1st (heavy) chain and 2nd (light) chain
    #     else:
    #         pdbs_csv = generate_pdbs_csv(args.pdb_dir, max_chains=2)

    print(f'[DEBUG - main.py main()] Predicting {args.num_seq_per_target} seqs per target', flush=True)
    print(f'[DEBUG - main.py main()] regions_to_mutate: {regions_to_mutate}', flush=True)
    print(f'[DEBUG - main.py main()] sampling_temp: {args.sampling_temp}', flush=True)
    print(f'[DEBUG - main.py main()] custom_chain_mode: {args.custom_chain_mode}', flush=True)
    # Extra: Sample sequences with num_seq_per_target >= 1
    if args.num_seq_per_target >= 1:
        log.info(
            f"Will sample {args.num_seq_per_target} sequences from {len(pdbs_csv.values)} PDBs at temperature(s) {args.sampling_temp} and regions: {regions_to_mutate}"
        )

    # Load AntiFold or ESM-IF1 model
    # Infer model from file path
    print(f'[DEBUG - main.py main()] Loading model with path: {args.model_path}', flush=True)
    try:
        model = load_model(args.model_path)
        print(f'[DEBUG - main.py main()] Successfully loaded model', flush=True)
    except Exception as e:
        print(f'[ERROR - main.py main()] Exception in load_model(): {e}', flush=True)
        raise

    # Get dict with PDBs, sampled sequences and logits / log_odds DataFrame
    print(f'\n[DEBUG - main.py main()] Calling sample_pdbs()', flush=True)
    # Log polyglycine resampling settings
    if args.resample_polyglycine:
        print(f'[DEBUG - main.py main()] Polyglycine resampling enabled with max {args.max_polyglycine_iterations} iterations', flush=True)
    else:
        print(f'[DEBUG - main.py main()] Polyglycine resampling disabled', flush=True)
        
    pdb_output_dict = sample_pdbs(
        model=model,
        pdbs_csv_or_dataframe=pdbs_csv,
        pdb_dir=pdb_dir,
        regions_to_mutate=regions_to_mutate,
        out_dir=args.out_dir,
        sample_n=args.num_seq_per_target,
        sampling_temp=args.sampling_temp,
        limit_expected_variation=args.limit_variation,
        exclude_heavy=args.exclude_heavy,
        exclude_light=args.exclude_light,
        batch_size=args.batch_size,
        extract_embeddings=args.extract_embeddings,
        custom_chain_mode=args.custom_chain_mode,
        num_threads=args.num_threads,
        seed=args.seed,
        save_flag=True,
        resample_polyglycine=args.resample_polyglycine,
        max_polyglycine_iterations=args.max_polyglycine_iterations,
    )


if __name__ == "__main__":
    args = cmdline_args()

    # Log to file and stdout
    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.abspath(f"{args.out_dir}/log.txt")

    logging.basicConfig(
        level=logging.INFO,
        format="[{asctime}] {message}",
        style="{",
        handlers=[
            logging.FileHandler(filename=log_path, mode="w"),
            logging.StreamHandler(stream=sys.stdout),
        ],
    )
    log = logging.getLogger(__name__)

    # INFO prints total summary and errors (default)
    if args.verbose == 1:
        logging.getLogger().setLevel(logging.INFO)

    # DEBUG prints every major step
    elif args.verbose >= 2:
        logging.getLogger().setLevel(logging.DEBUG)

    # Check valid input
    try:
        log.info(f"Running inverse folding on PDB/CIFs ...")
        check_valid_input(args)
        print(f'[DEBUG - main.py] Using model: {args.model_path}')
        main(args)

    except Exception as E:
        log.exception(f"Prediction encountered an unexpected error: {E}")
