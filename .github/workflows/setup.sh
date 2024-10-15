#!/bin/bash

cd ../..

# 必要なデータのダウンロード
mkdir -p ./step2/data
curl -o ./step2/data/height.txt https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-5/refs/heads/main/step02/height.txt

mkdir -p ./step3/data
curl -o ./step3/data/height_weight.txt https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-5/refs/heads/main/step03/height_weight.txt

mkdir -p ./step4/data
curl -o ./step4/data/old_faithful.txt https://raw.githubusercontent.com/oreilly-japan/deep-learning-from-scratch-5/refs/heads/main/step04/old_faithful.txt
