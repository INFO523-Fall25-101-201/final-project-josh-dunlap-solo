#!/bin/bash
# setup_xml_env.sh 

module load micromamba

ENV_NAME="xml-processing"
ENV_LOC=/xdisk/cjgomez/joshdunlapc/conda_envs

# Ensure dirs exist
mkdir -p $ENV_LOC/envs
mkdir -p $ENV_LOC/pkgs

# Configure micromamba locations 
micromamba config append envs_dirs $ENV_LOC/envs
micromamba config append pkgs_dirs $ENV_LOC/pkgs

# Create environment
micromamba create -y -n $ENV_NAME python=3.10

# Activate
micromamba activate $ENV_NAME

# Install packages
pip install pandas==2.1.4
pip install lxml==5.1.0
pip install pyarrow==14.0.1
pip install numpy==1.26.4
pip install scipy==1.10.1   
pip install gensim==4.3.2 --no-build-isolation --no-use-pep517

# Optional
micromamba install -y ipython
