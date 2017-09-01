#! /bin/bash

# create virtual environment
virtualenv --python=python3 venv
source venv/bin/activate

# install dependencies
pip install -r requirements.txt

cd src/
# partially download repo
svn export https://github.com/tensorflow/models.git/trunk/object_detection
svn export https://github.com/tensorflow/models.git/trunk/slim

# build protobufs
sudo apt-get install protobuf-compiler
protoc object_detection/protos/*.proto --python_out=.

echo 'export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim' >> ~/.bashrc

mkdir graphs

cd ..
