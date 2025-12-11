#!/bin/bash
#SBATCH --job-name=merge-decades
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=joshdunlapc@email.arizona.edu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32GB
#SBATCH --time=02:00:00
#SBATCH --output=logs/merge-decades-%j.log

echo "Job started: $(date)"

# Load environment
module load micromamba
eval "$(micromamba shell hook --shell bash)"
micromamba activate xml-processing

# Run merge script
cd /xdisk/cjgomez/joshdunlapc/xml-to-parquet-project
python merge_by_decade.py

echo "Job ended: $(date)"