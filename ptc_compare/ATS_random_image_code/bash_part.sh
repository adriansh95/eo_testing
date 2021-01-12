#!/bin/bash

REPO_DIR=/home/adriansh/lsst_devel/analysis/ptc_comparison/simulated_pedestal
DATA_DIR=$REPO_DIR/raw
rm -rf $REPO_DIR
mkdir $REPO_DIR
mkdir $DATA_DIR
cd $REPO_DIR
python /home/adriansh/lsst_devel/software/ptc_compare/ATS_random_image_code/dataCopy.py
echo "lsst.obs.lsst.auxTel.AuxTelMapper" > $REPO_DIR/_mapper
chmod -R 777 ./*
ingestImages.py $REPO_DIR $DATA_DIR/*.fits --mode=move
chmod 777 registry.sqlite3
ln -s /project/shared/auxTel/CALIB CALIB
