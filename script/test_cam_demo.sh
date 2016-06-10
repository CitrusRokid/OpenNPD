SH_FILE=$(readlink -f $0)
SH_DIR=$(dirname $SH_FILE)
cd $SH_DIR 
../bin/test_cam ../models/frontal_face_detector.bin
