#!/bin/sh
DEST=/home/xinyuc/hddwork/hash_join_harp/radix_partition
SUBDIRECTORY=device
SUBDIRECTORY1=host
for a in $DEST/*; do
  #echo $a
  BASENAME=`basename $a`
  echo $BASENAME
  if [ -d "$a/$SUBDIRECTORY" ]; then
    mkdir -p ./$BASENAME
    cp -r $a/$SUBDIRECTORY ./$BASENAME
    cp -r $a/$SUBDIRECTORY1 ./$BASENAME
  fi

done
