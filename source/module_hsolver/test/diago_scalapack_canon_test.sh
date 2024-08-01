#!/bin/bash

npmax=`cat /proc/cpuinfo | grep "cpu cores" | uniq | awk '{print $NF}'`

for i in {1..8};
do
    if [[ $i -gt $npmax ]];then
        break
    fi
    mpirun -np $i ./diago_scalapack_canon_test
done
