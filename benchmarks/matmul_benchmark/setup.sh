#!/bin/bash 

currPath=$(dirname "$0")
# python modules installation
if ! pip install -r $currPath/requirements.txt
then
    echo "Required modules installed"
fi