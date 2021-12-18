#!/bin/bash

rm a.out

nvcc -isystem ./utils -isystem ./device_utils -isystem ./kernels -isystem ./main ./main/$1.cu -lcurand  -o a.out 

mkdir logs/$1_logs

./a.out > logs/$1_logs/run.log

cat logs/$1_logs/run.log
