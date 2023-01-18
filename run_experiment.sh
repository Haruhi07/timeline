#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition gpu
#SBATCH --job-name=clustering
#SBATCH --time=12:0:0
#SBATCH --mem=8192M

module load lang/perl/5.30.0-bioperl-gcc
module load lang/python/anaconda/3.8-2020.07

cd "${SLURM_SUBMIT_DIR}"
source venv/bin/activate

time python -u news-tls/experiments/evaluate.py \
	--dataset dataset/crisis \
	--method clust \
	--resources news-tls/resources/datewise \
	--output multinews_results/clust/crisis
