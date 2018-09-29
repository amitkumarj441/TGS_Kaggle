#!/usr/bin/env bash

set -e

apt-get update >/dev/null
apt-get -y install python3-dev libsm-dev libxrender1 libxext6 zip git >/dev/null
rm -rf /var/lib/apt/lists/*

pip -q install virtualenv
virtualenv env --python=python3
. env/bin/activate

pip -q install -r requirements.txt

printf "commit: $(git rev-parse HEAD)\n\n" | tee -a /artifacts/out.log

if [ -z "$2" ]
then
  python -m cProfile -o /artifacts/train.prof train.py 2>/artifacts/err.log | tee -a /artifacts/out.log
else
  cp /kaggle/models/tgs/$2/*.pth /artifacts
  python -m cProfile -o /artifacts/train.prof train.py /kaggle/models/tgs/$2 2>/artifacts/err.log | tee -a /artifacts/out.log
fi

( cd /artifacts && zip -r logs.zip logs )
rm -rf /artifacts/logs

rm -rf /kaggle/models/tgs/$1
mkdir -p /kaggle/models/tgs/$1
cp -r /artifacts/* /kaggle/models/tgs/$1
