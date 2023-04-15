#!/bin/bash

folder_name="Clusters"
subfolders=("KMeans" "GMM" "Ignore")
values=("1" "3" "5" "7" "9" "11")

if [ ! -d "$folder_name" ]; then
  mkdir "$folder_name"
  echo "New folder '$folder_name' created!"
fi

for subfolder in "${subfolders[@]}"; do
  subfolder_path="$folder_name/$subfolder"
  if [ ! -d "$subfolder_path" ]; then
    mkdir "$subfolder_path"
    echo "New subfolder '$subfolder_path' created!"
  fi
  
  for value in "${values[@]}"; do
    value_path="$subfolder_path/$value"
    if [ ! -d "$value_path" ]; then
      mkdir "$value_path"
      echo "New subfolder '$value_path' created!"
    fi
  done
done

