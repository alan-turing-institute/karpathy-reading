echo "This should be sourced from a compute node"
#srun --qos turing --account usjs9456-ati-test --time 4:00:00 --nodes 1 --gpus 1 --cpus-per-gpu 36 --mem 16384 --pty /bin/bash

echo "Loading moduels"
module -q purge
module -q load baskerville
module -q load bask-apps/live
module -q load PyTorch/2.1.2-foss-2023a-CUDA-12.1.1
module -q load torchvision/0.16.0-foss-2023a-CUDA-12.1.1
#module load ImageMagick/7.1.0-37-GCCcore-11.3.0

echo "Deploying virtual environment"
python3.11 -m venv venv
. ./venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

echo "Setup complete. You can now run the pixelcnn code"

