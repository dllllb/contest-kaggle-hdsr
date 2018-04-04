#!/usr/bin/env bash

if [[ ! -e attributes.csv.gz ]]
then
    gsutil cp gs://di-datasets/home-depot-search-relevance/attributes.csv.gz .
fi

if [[ ! -e product_descriptions.csv.gz ]]
then
    gsutil cp gs://di-datasets/home-depot-search-relevance/product_descriptions.csv.gz .
fi
