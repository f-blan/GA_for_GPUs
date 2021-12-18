#!/bin/bash

rm a.out

nvcc -isystem ./utils -isystem ./device_utils -isystem ./kernels -isystem ./main ./main/$1.cu -lcurand  -o a.out 


mkdir logs/$1_logs

sudo /usr/local/cuda-10/bin/nvprof --log-file logs/$1_logs/n.log  ./a.out 
