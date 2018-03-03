#!/bin/bash

SOURCE_URL='http://yann.lecun.com/exdb/mnist/'

mkdir data
rm -fr data/*

function download_file () {
    curl -o data/$1 ${SOURCE_URL}/$1
    gunzip data/$1
}

download_file train-images-idx3-ubyte.gz
download_file train-labels-idx1-ubyte.gz
download_file t10k-images-idx3-ubyte.gz
download_file t10k-labels-idx1-ubyte.gz
