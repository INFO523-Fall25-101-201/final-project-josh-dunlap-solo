#!/bin/bash
#SBATCH --job-name=xml-to-parquet-array
#SBATCH --mail-type=BEGIN,FAIL,END
#SBATCH --mail-user=joshdunlapc@email.arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16GB
#SBATCH --time=04:00:00
#SBATCH --output=logs/xml-to-parquet-%A-%a.log
#SBATCH --array=0-337

echo "========================================="
echo "Job Information"
echo "========================================="
echo "Job started: $(date)"
echo "Hostname: $(hostname -s)"
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Working directory: $(pwd)"
echo ""

# REPRODUCIBILITY: Load and initialize micromamba properly
echo "Loading environment..."
module load micromamba

# Initialize micromamba for this shell session
eval "$(micromamba shell hook --shell bash)"

# Now activate the environment
micromamba activate xml-processing

# Verify environment
echo "Python location: $(which python)"
echo "Python version: $(python --version)"
echo "Pandas version: $(python -c 'import pandas; print(pandas.__version__)')"
echo "lxml version: $(python -c 'import lxml; print(lxml.__version__)')"
echo "pyarrow version: $(python -c 'import pyarrow; print(pyarrow.__version__)')"
echo ""

# Project directory
PROJECT_DIR="/xdisk/cjgomez/joshdunlapc/xml-to-parquet-project"

# Read the folder path for this array task
FOLDERS_FILE="${PROJECT_DIR}/folders.txt"
INPUT_FOLDER=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" $FOLDERS_FILE)

# Define output directory and filename
OUTPUT_DIR="/xdisk/cjgomez/joshdunlapc/parquet_output"
mkdir -p $OUTPUT_DIR

# Extract folder name for output file
FOLDER_NAME=$(basename $INPUT_FOLDER)
OUTPUT_FILE="${OUTPUT_DIR}/${FOLDER_NAME}.parquet"

echo "========================================="
echo "Processing Details"
echo "========================================="
echo "Processing folder: $INPUT_FOLDER"
echo "Output file: $OUTPUT_FILE"
echo ""

# Run the Python script
python ${PROJECT_DIR}/process_xml_to_parquet.py "$INPUT_FOLDER" "$OUTPUT_FILE"

EXIT_CODE=$?
echo ""
echo "========================================="
echo "Job Summary"
echo "========================================="
echo "Job ended: $(date)"
echo "Exit code: $EXIT_CODE"
echo "========================================="

exit $EXIT_CODE