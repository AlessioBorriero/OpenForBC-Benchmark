#!/bin/bash 

currPath=$(dirname "$0")
## PYTHON MODULES INSTALLATION ##
pip install -r $currPath/requirements.txt

## CHECK THE PRESENCE OF A CUDA VERSION ##
CUDA_VER=$(nvidia-smi | grep -oP  '(?<=CUDA Version: )[0-9]*')
re='^[0-9]+$'   

if ! [[ $CUDA_VER =~ $re ]]
then
    echo 'CUDA not installed, please manually install it.'
fi
