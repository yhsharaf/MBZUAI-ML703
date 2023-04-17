#!/bin/bash

folder_name="Clusters"
subfolders=("KMeans" "GMM" "Ignore")
subsubfolders=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10" "11" "12" "13" "14" "15" "16" "17" "18" "19" "20" "21" "22" "23" "24")
values=("1" "3" "5" "7" "9" "11")

if [ ! -d "$folder_name" ]; then
  mkdir "$folder_name"
  echo "New folder '$folder_name' created!"
fi

for subfolder in "${subfolders[@]}"; do
  subfolder_path="$folder_name/$subfolder"
  if [ ! -d "$subfolder_path" ]; then
    mkdir "$subfolder_path"
    # echo "New subfolder '$subfolder_path' created!"
  fi

  for subsubfolder in "${subsubfolders[@]}"; do
    subsubfolder_path="$subfolder_path/$subsubfolder"
    if [ ! -d "$subsubfolder_path" ]; then
      mkdir "$subsubfolder_path"
      # echo "New subsubfolder '$subsubfolder_path' created!"
    fi
  
    for value in "${values[@]}"; do
      value_path="$subsubfolder_path/$value"
      if [ ! -d "$value_path" ]; then
        mkdir "$value_path"
        # echo "New value folder '$value_path' created!"
      fi
    done
  done
done
