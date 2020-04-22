#!/bin/bash

find ./dataset-resized | grep ._ | xargs rm

for i in $(ls dataset-resized)
do
    mkdir -p "train/$i"
    mkdir -p "val/$i"
    mkdir -p "test/$i"
    n=$(ls -v "dataset-resized/$i" | wc -l)

    # train data
    split=$(($n-$n*1/10))
    for j in $(find "dataset-resized/$i" | grep -v "./$i$" | head -n $split )
    do
        mv $j "train/$i"
    done

    # validation data
    remain=$(($n-$split))
    for j in $(find "dataset-resized/$i" | grep -v "./$i$" | head -n $(($remain)) )
    do
        mv $j "val/$i"
    done
done
rm -rf dataset-resized
