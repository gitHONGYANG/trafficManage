#! /bin/bash

count=0
cat /home/highway/Highway/bottomlayer/core/process.txt | while read line
do
    pid=${line}
    echo $pid
    kill -9 $pid
done
