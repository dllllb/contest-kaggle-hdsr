#!/usr/bin/env bash

# use the following command to automatically setup the remote storage: rclone config create gcs "google cloud storage"
rclone copy -P gcs:di-datasets/home-depot-search-relevance data
