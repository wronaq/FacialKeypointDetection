#! /bin/bash

#############################################################
####### move half of test data to make validation set #######
#############################################################

# 1. count how much data to move
COUNT=$(wc -l data/test_frames_keypoints.csv | awk '{print $1}')
HALF=$(($COUNT/2))

# 2. clean & make dirs
mv data/test data/test.old
mkdir -p data/validation
mkdir -p data/test

# 3. get list of files to move
FIRST_LINE=$(cat data/test_frames_keypoints.csv | head -n 1)
LIST=$(cat data/test_frames_keypoints.csv | sed 1d | sort -R)
VAL_LIST=$(echo $LIST | cut -d ' ' --output-delimiter=$'\n' -f1- | head -n $HALF | cut -d ',' -f 1)
TEST_LIST=$(echo $LIST | cut -d ' ' --output-delimiter=$'\n' -f1- | tail -n $(($COUNT-$HALF)) | cut -d ',' -f 1)

# 4. move it
echo $VAL_LIST | cut -d ' ' --output-delimiter=$'\n' -f1- | xargs -i cp data/test.old/{} data/validation
echo $TEST_LIST | cut -d ' ' --output-delimiter=$'\n' -f1- | xargs -i cp data/test.old/{} data/test
echo 'Data moved to ./data/validation and ./data/test'

# 5. prepare new keypoints csv
mv data/test_frames_keypoints.csv data/test_frames_keypoints.old
echo $FIRST_LINE > data/validation_frames_keypoints.csv && echo $FIRST_LINE > data/test_frames_keypoints.csv
for i in $VAL_LIST; do
  grep $i data/test_frames_keypoints.old >> data/validation_frames_keypoints.csv
done
for i in $TEST_LIST; do
  grep $i data/test_frames_keypoints.old >> data/test_frames_keypoints.csv
done

echo 'Files with keypoints created.' && echo 'All done.'
