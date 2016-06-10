# OpenNPD

Project of object detection.

 Usage:
  1. Add needed librarys to 3rdpart folder(opencv).
  2. cd src; make clean; make RELEASE=1; cd ..;
  3. Detect example:
  ```
    sh ./script/test_detect_demo.sh
  ```
  4. Training example:
    Please refer to ./script/npd_train_demo.sh


References:

  [Project page](http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/index.html)
  @article{
      Author = {Shengcai Liao, Member, IEEE, Anil K. Jain, Fellow, IEEE, and Stan Z. Li, Fellow, IEEE},
      Title = {A Fast and Accurate Unconstrained Face Detector},
      Year = {2014}
  }

