#! /bin/bash

# clean first
DATA_FOLDER=$1
rm -f all_kps unique rest_of_file test_frames_keypoints.csv training_frames_keypoints.csv\
     $DATA_FOLDER/test_frames_keypoints.csv $DATA_FOLDER/training_frames_keypoints.csv\
     training_persons test_persons persons
rm -rf tmp $DATA_FOLDER/test $DATA_FOLDER/training

# unzip and manage files
unzip -o $DATA_FOLDER/train-test-data.zip -d $DATA_FOLDER
mv $DATA_FOLDER/test_frames_keypoints.csv $DATA_FOLDER/test_frames_keypoints.old
mv $DATA_FOLDER/training_frames_keypoints.csv $DATA_FOLDER/training_frames_keypoints.old
mv $DATA_FOLDER/test $DATA_FOLDER/test.old
mv $DATA_FOLDER/training $DATA_FOLDER/training.old



#### KEYPOINTS ####

# concatenate all files, drop duplicates
cat $DATA_FOLDER/test_frames_keypoints.old $DATA_FOLDER/training_frames_keypoints.old > all_kps
sort -u -t, -k1,1 all_kps > unique
head -n 1 unique > training_frames_keypoints.csv
head -n 1 unique > test_frames_keypoints.csv
sed 1d unique | shuf > rest_of_file
awk -F, '{print substr($1, 1, length($1)-7)}' rest_of_file | sort | uniq | shuf > persons

# set train and validation size
TRAIN_SIZE=$(( $2 / 10 ))
TEST_SIZE=$(( $(wc -l persons | awk '{print $1}') - $TRAIN_SIZE))



# prepare proper keypoints files
head -n $TRAIN_SIZE persons > training_persons
tail -n $TEST_SIZE persons > test_persons
cat training_persons | xargs -I {} grep {} unique >> training_frames_keypoints.csv
cat test_persons | xargs -I {} grep {} unique >> test_frames_keypoints.csv

mv test_frames_keypoints.csv $DATA_FOLDER/test_frames_keypoints.csv
mv training_frames_keypoints.csv $DATA_FOLDER/training_frames_keypoints.csv


#### IMAGES ####

mkdir -p tmp
mkdir -p $DATA_FOLDER/{test,training}
cp -r $DATA_FOLDER/test.old/* tmp
cp -r $DATA_FOLDER/training.old/* tmp

awk -F, '{print $1}' $DATA_FOLDER/test_frames_keypoints.csv | sed 1d | xargs -I {} mv tmp/{} $DATA_FOLDER/test
awk -F, '{print $1}' $DATA_FOLDER/training_frames_keypoints.csv | sed 1d | xargs -I {} mv tmp/{} $DATA_FOLDER/training


# output
echo '--------------------------------------------------------------------------------------'
echo "No. of observations in train: $(( $(wc -l  $DATA_FOLDER/training_frames_keypoints.csv | awk '{print $1}') - 1))"
echo "No. of observations in test: $(( $(wc -l  $DATA_FOLDER/test_frames_keypoints.csv | awk '{print $1}') - 1))"


# clean on exit
rm -f all_kps unique rest_of_file test_frames_keypoints.csv training_frames_keypoints.csv\
     $DATA_FOLDER/test_frames_keypoints.old $DATA_FOLDER/training_frames_keypoints.old\
     training_persons test_persons persons
rm -rf $DATA_FOLDER/test.old $DATA_FOLDER/training.old tmp

echo 'DONE'
