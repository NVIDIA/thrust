#!/usr/bin/env bash

for file in *; do
    if [ -x "$file" ]; then
        if [ "${file/*./}" != "sh" ]; then
            echo "Running $file"
            ./"$file";
            echo
        fi;
    fi; 
done


