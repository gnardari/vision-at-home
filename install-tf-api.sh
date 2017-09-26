#! /bin/bash


# install dependencies
pip install -r requirements.txt

cd src/
# partially download repo
svn export https://github.com/tensorflow/models.git/trunk/research/object_detection
svn export https://github.com/tensorflow/models.git/trunk/research/slim

# build protobufs
sudo apt-get install protobuf-compiler
protoc object_detection/protos/*.proto --python_out=.

echo 'export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim' >> ~/.bashrc

mkdir graphs

cd ..
