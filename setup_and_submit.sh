#!/bin/bash

PROJECT_DIR="/xdisk/cjgomez/joshdunlapc/xml-to-parquet-project"
XML_BASE_DIR="/xdisk/cjgomez/joshdunlapc/datasets/xml_only/nyt_xml"
OUTPUT_DIR="/xdisk/cjgomez/joshdunlapc/parquet_output"

# Create necessary directories
mkdir -p ${PROJECT_DIR}/logs
mkdir -p $OUTPUT_DIR

echo "Creating folder list..."
cd $XML_BASE_DIR

# List all NYT_* directories (folders, not files)
ls -d $PWD/NYT_*/ | sed 's|/$||' | sort > ${PROJECT_DIR}/folders.txt

# Count folders
NUM_FOLDERS=$(wc -l < ${PROJECT_DIR}/folders.txt)
echo "Found $NUM_FOLDERS folders to process"

# Show first few entries
echo ""
echo "First 5 folders:"
head -5 ${PROJECT_DIR}/folders.txt

echo ""
echo "Last 5 folders:"
tail -5 ${PROJECT_DIR}/folders.txt

echo ""
echo "======================================"
echo "Total folders: $NUM_FOLDERS"
echo "Full array range: --array=0-$((NUM_FOLDERS - 1))"
echo "======================================"
echo ""

# Ask what to do
echo "What would you like to do?"
echo "1) Test run (first 10 folders only)"
echo "2) Run remaining folders (after test)"
echo "3) Run all folders"
echo "4) Exit (no submission)"
echo ""
read -p "Enter choice (1-4): " choice

case $choice in
    1)
        echo "Submitting TEST run with first 10 folders (array 0-9)..."
        cd $PROJECT_DIR
        sbatch --account=cjgomez --partition=standard --array=0-9 process_xml_array.sbatch
        echo ""
        echo "Test jobs submitted!"
        echo "Check status with: squeue -u $USER"
        echo "Check logs with: ls -lh ${PROJECT_DIR}/logs/"
        echo ""
        echo "After test completes, re-run this script and choose option 2"
        ;;
    2)
        echo "Submitting REMAINING folders (array 10-$((NUM_FOLDERS - 1)))..."
        cd $PROJECT_DIR
        sbatch --account=cjgomez --partition=standard --array=10-$((NUM_FOLDERS - 1)) process_xml_array.sbatch
        echo ""
        echo "Remaining jobs submitted!"
        echo "Check status with: squeue -u $USER"
        ;;
    3)
        echo "Submitting ALL folders (array 0-$((NUM_FOLDERS - 1)))..."
        cd $PROJECT_DIR
        sbatch --account=cjgomez --partition=standard --array=0-$((NUM_FOLDERS - 1)) process_xml_array.sbatch
        echo ""
        echo "All jobs submitted!"
        echo "Check status with: squeue -u $USER"
        ;;
    4)
        echo "Submission cancelled."
        echo ""
        echo "When ready to submit manually, run:"
        echo "  cd $PROJECT_DIR"
        echo "  # For test (10 folders):"
        echo "  sbatch --account=cjgomez --partition=standard --array=0-9 process_xml_array.sbatch"
        echo "  # For remaining:"
        echo "  sbatch --account=cjgomez --partition=standard --array=10-336 process_xml_array.sbatch"
        ;;
    *)
        echo "Invalid choice. Exiting."
        ;;
esac