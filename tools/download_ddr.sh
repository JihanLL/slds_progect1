#!/bin/bash
curl -L -o data/ddrdataset.zip https://www.kaggle.com/api/v1/datasets/download/mariaherrerot/ddrdataset
echo "Download complete"
unzip -o data/ddrdataset.zip -d ../data
echo "Unzipped data"
rm data/ddrdataset.zip