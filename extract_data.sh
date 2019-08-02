kaggle competitions download -p ./data -c planet-understanding-the-amazon-from-space
sudo apt-get install p7zip-full
sudo apt-get install unzip
7za x data/train-jpg.tar.7z -o./data
tar xf data/train-jpg.tar -C ./data
rm data/train-jpg.tar*
7za x data/test-jpg.tar.7z -o./data
tar xf data/test-jpg.tar -C ./data
rm data/test-jpg.tar*
7za x data/test-jpg-additional.tar.7z -o./data
tar xf data/test-jpg-additional.tar -C ./data
rm data/test-jpg-additional.tar*
unzip data/sample_submission_v2.csv.zip -d ./data
unzip data/train_v2.csv.zip -d ./data
rm data/sample_submission_v2.csv.zip
rm data/train_v2.csv.zip
