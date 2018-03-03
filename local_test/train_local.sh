#!/bin/sh

image=$1

mkdir -p test_dir/model
mkdir -p test_dir/output

rm test_dir/model/*
rm test_dir/output/*

rm -fr test_dir/input/data/training/*
rm -fr test_dir/input/data/eval/*

cp ../data/train-* test_dir/input/data/training
cp ../data/t10k-* test_dir/input/data/eval

docker run -v $(pwd)/test_dir:/opt/ml --rm ${image} train
