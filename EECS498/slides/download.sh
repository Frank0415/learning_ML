#!/bin/bash

base_url="https://web.eecs.umich.edu/~justincj/slides/eecs498/WI2022/598_WI2022_lecture"

for i in {14..16}; do
    url="${base_url}${i}.pdf"
    output_file="498_FA2019_lecture${i}.pdf"
    curl -o "$output_file" "$url"
done
