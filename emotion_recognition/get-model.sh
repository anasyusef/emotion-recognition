#!/usr/bin/env bash

FNAME=emotion_recognition_model.zip

wget --load-cookies /tmp/cookies.txt -r "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=10DMG2OM9QoCjGtrP-UsUXPqX0kmACUKq' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=10DMG2OM9QoCjGtrP-UsUXPqX0kmACUKq" -O $FNAME && rm -rf /tmp/cookies.txt

# Unzip
unzip -q $FNAME

# # Delete .zip files
rm -rf $FNAME