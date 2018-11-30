FILE=$1

if [[ $FILE != "mini" ]]
then
    echo "Available datasets are: mini"
    exit 1
fi

URL=http://zefirus.org/datasets/$FILE.zip
ZIP_FILE=./datasets/$FILE.zip
TARGET_DIR=./datasets/$FILE/
wget -N $URL -O $ZIP_FILE
mkdir $TARGET_DIR
unzip $ZIP_FILE -d ./datasets/
rm $ZIP_FILE
