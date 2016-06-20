# OpenNPD

Project of object detection.

 Usage:

  1. Add needed librarys to 3rdpart folder(opencv).

  2. Make: `cd src; make clean; make RELEASE=1; cd ..`;

  3. Detect example: `sh ./script/test_detect_demo.sh`

  4. Training example: Please refer to `./script/npd_train_demo.sh`

Note:

1. Thereis a `npddetect::prescandetect` function for faster detection with some lose on recall. Additional parameter `stepR` refers to pre-scan step size compared to the original scan step size ( float more than 1 ) . The `thresR` refers to the threshold to reject the window( float in [0-1] ).

Result:

1. ROC:

![图片1.png-123.8kB][1]

2. Speed:

| image size | window size | cores | time (ms) |
| --- | --- | --- | --- | --- |
| 640x480 | 20x20 | 1 | ~50 |

References:

  [Project page](http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/index.html)

> @article{

>      Author = {Shengcai Liao, Member, IEEE, Anil K. Jain, Fellow, IEEE, and Stan Z. Li, Fellow, IEEE},

>      Title = {A Fast and Accurate Unconstrained Face Detector},

>      Year = {2014}

>  }


  [1]: http://static.zybuluo.com/No-Winter/4n2oqec9hqj80qic2ujebxmn/%E5%9B%BE%E7%89%871.png