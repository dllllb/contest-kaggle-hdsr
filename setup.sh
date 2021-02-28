#!/bin/sh

# use the following command to automatically setup the remote storage: `rclone config create gcs "google cloud storage"`
rclone copy -P gcs:di-datasets/home-depot-search-relevance.zip data
cd data
unzip home-depot-search-relevance.zip
