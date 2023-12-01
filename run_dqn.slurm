#!/bin/bash
#SBATCH --job-name=dqn_agent_run
#SBATCH --output=dqn_output_%j.txt
#SBATCH --nodes=1
#SBATCH --time=10:00:00

PYTHON_VERSION=3.11.5
INSTALL_DIR=$HOME/.local/python-$PYTHON_VERSION
wget https://www.python.org/ftp/python/$PYTHON_VERSION/Python-$PYTHON_VERSION.tar.xz
tar -xf Python-$PYTHON_VERSION.tar.xz
cd Python-$PYTHON_VERSION
./configure --prefix=$INSTALL_DIR
make
make install
cd ..
$INSTALL_DIR/bin/python -m venv venv

source venv/bin/activate

# Install requirements (assuming requirements.txt is in the current directory)
pip install -r requirements.txt

# Run the script
python dqn_agent.py

deactivate