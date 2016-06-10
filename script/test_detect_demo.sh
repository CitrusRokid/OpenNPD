SH_FILE=$(readlink -f $0)
SH_DIR=$(dirname $SH_FILE)
cd $SH_DIR 
../bin/test_detect ../models/frontal_face_detector.bin ../data/sample_images/facedetect_5.jpg
