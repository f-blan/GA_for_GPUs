#!/bin/bash

rm a.out

nvcc -isystem ./utils -isystem ./device_utils -isystem ./kernels ./main/$1.cu -lcurand  -o a.out 


./a.out

mkdir logs/$2/$1_logs

sudo /usr/local/cuda-10/bin/nvprof --log-file logs/$2/$1_logs/n.log  ./a.out 
sudo /usr/local/cuda-10/bin/nvprof --print-gpu-trace --log-file logs/$2/$1_logs/gpu.log ./a.out 
sudo /usr/local/cuda-10/bin/nvprof --print-api-trace --log-file logs/$2/$1_logs/api.log ./a.out
sudo /usr/local/cuda-10/bin/nvprof --events all --log-file logs/$2/$1_logs/all.log ./a.out
exit
