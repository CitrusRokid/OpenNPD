SH_FILE=$(readlink -f $0)
SH_DIR=$(dirname $SH_FILE)
cd $SH_DIR && \
cd ../models/ && \
mkdir npd_train_demo/ 
cd npd_train_demo/ && \
find ../../data/face_rect_img/images/ -name "*.jpg" > pos.list && \
find ../../data/neg_img/images/ -name "*.jpg" > neg.list && \
../../bin/trainNPD -faceDB pos.list -negDB neg.list -outModel result -objSize 20 -numPos 200 -negRatio 1 -maxTreeLevel 4 -minDR 1 -maxFAR 1e-3 -maxNumStages 200 -ifFlip 1

