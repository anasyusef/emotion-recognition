#!/usr/bin/env bash

FNAME=yolov3-wider_16000.weights.zip

wget --load-cookies /tmp/cookies.txt -r "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=13gFDLFhhBqwMw6gf8jVUvNDH2UrgCCrX' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=13gFDLFhhBqwMw6gf8jVUvNDH2UrgCCrX" -O $FNAME && rm -rf /tmp/cookies.txt

# Unzip
unzip -q $FNAME

# Delete .zip files
rm -rf $FNAME