#!/bin/bash

rm a.out

nvcc -isystem ./utils -isystem ./device_utils -isystem ./kernels ./main/$1 -lcurand -o a.out 

./a.out 
