#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32000M
#SBATCH --time=168:00:00
#SBATCH --output=%N-%j.out

cp translation_mbrats_cyclegan.h5 $SLURM_TMPDIR/
module load python
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install monai batchgenerators numpy matplotlib scikit-learn scikit-image tensorboard torch lightning h5py --no-index

srun python main.py --source_modality $1 --target_modality $2 --data_source $SLURM_TMPDIR/translation_mbrats_cyclegan.h5
