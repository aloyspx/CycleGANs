#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --time=168:00:00
#SBATCH --output=%N-%j.out

module load python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install monai batchgenerators numpy matplotlib scikit-learn scikit-image transformers tensorboard torch lightning --no-index

srun python main.py --source_modality $1 --target_modality $2
