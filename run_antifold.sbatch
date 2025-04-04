#!/bin/bash
#SBATCH --job-name=AntiFold         # short name for your job
#SBATCH --nodes=1                   # node count
#SBATCH --ntasks=1                  # total number of tasks across all nodes
#SBATCH --cpus-per-task=1           # cpu-cores per task (>1 if multi-threaded tasks)
#SBATCH --mem-per-cpu=56G           # memory per cpu-core 
#SBATCH --time=05:00:00             # total run time limit (HH:MM:SS)
#SBATCH --gres=gpu:A100:1           # number of GPUs per node
#SBATCH --partition=gpu             # requested partition
#SBATCH --account=p_2024001_hpccom_immunochem     # billing account

# ------------------------------------------------------------------------------
# Print some info
# ------------------------------------------------------------------------------
echo "job_id             : $SLURM_JOB_ID"
echo "jobname            : $SLURM_JOB_NAME"
echo "number of mpi tasks: $SLURM_NTASKS tasks"
echo "OMP_NUM_THREADS    : $OMP_NUM_THREADS"

# ------------------------------------------------------------------------------
# Activate Conda Environment
# ------------------------------------------------------------------------------
module load devel/python/Anaconda3-2022.05  # Load Anaconda if required
source $(conda info --base)/etc/profile.d/conda.sh  # Ensure `conda activate` works

# Activate your IgDesign Conda environment
conda activate /gpfs/projects/immunochem/p_2024001_hpccom_immunochem/.conda/envs/antifold

# **Force the Job to Use the Conda Python**
export PATH=/gpfs/projects/immunochem/p_2024001_hpccom_immunochem/.conda/envs/antifold/bin:$PATH
export CONDA_PREFIX=/gpfs/projects/immunochem/p_2024001_hpccom_immunochem/.conda/envs/antifold
export CONDA_DEFAULT_ENV=antifold
export PYTHONNOUSERSITE=1  # Prevent interference from user-wide packages

# Verify Python Path to ensure Conda's Python is used
echo "Using Python from: $(which python)"
python --version

# ------------------------------------------------------------------------------
# Run the AntiFold inference
# ------------------------------------------------------------------------------
# Go to the IgDesign directory
cd /gpfs/projects/immunochem/p_2024001_hpccom_immunochem/garmstro/AntiFold

UPLOADED_FILE="data/nanobody_antigen/anti_ClfA_design_16.pdb"
NANO_CHAIN="H"
ANTIGEN_CHAIN="T"
SAMPLE_NUMBER=10
SAMPLING_TEMP=0.25
REGIONS="CDRH1 CDRH2 CDRH3"
LIMIT_VARIATION_STRING=""

# Start the inference
pwd
date
echo "Starting AntiFold inference... "
python antifold/main.py \
    --pdb_file "$UPLOADED_FILE" \
    --nanobody_chain "$NANO_CHAIN" \
    --antigen_chain "$ANTIGEN_CHAIN" \
    --regions "$REGIONS" \
    --num_seq_per_target "$SAMPLE_NUMBER" \
    --sampling_temp "$SAMPLING_TEMP" \
    --model_path "models/model.pt" \
    --verbose 1 \
    --seed 12 \
    $LIMIT_VARIATION_STRING

wait
echo "Inference complete"
date
