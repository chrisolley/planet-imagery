7za x data/train-jpg.tar.7z
tar xf data/train-jpg.tar
rm data/train-jpg.tar*
7za x data/test-jpg.tar.7z
tar xf data/test-jpg.tar
rm data/test-jpg.tar*
7za x data/test-jpg-additional.tar.7z
tar xf data/test-jpg-additional.tar
rm data/test-jpg-additional.tar*
unzip data/sample_submission_v2.csv.zip
unzip data/train_v2.csv.zip
rm data/sample_submission_v2.csv.zip
rm data/train_v2.csv.zip
