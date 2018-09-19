#!/bin/sh
DEST=/home/xinyuc/hddwork/liucheng/paper_experiments/fpga_only_graph_framework
SUBDIRECTORY=src
SUBDIRECTORY1=host
for a in $DEST/*; do
  #echo $a
  BASENAME=`basename $a`
  echo $BASENAME
  if [ -d "$a/$SUBDIRECTORY" ]; then
    mkdir -p ./$BASENAME
    cp -r $a/$SUBDIRECTORY ./$BASENAME
    #cp -r $a/$SUBDIRECTORY1 ./$BASENAME
  fi

done
