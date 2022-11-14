#!/bin/bash
id=1hK5ILpMSSxpeQ_2MTw8q95hMDnGyqu2o
name=book-dataset.zip
confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id='$id -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')
echo $confirm
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$confirm&id=$id" -O $name && rm -rf /tmp/cookies.txt
unzip book-dataset.zip
rm -rf book-dataset.zip
