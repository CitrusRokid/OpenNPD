# OpenNPD

Project of object detection.

## Usage:

  1. Add needed librarys to 3rdpart folder(opencv).

  2. Make: `cd src; make clean; make RELEASE=1; cd ..`;

  3. Detect example: `sh ./script/test_detect_demo.sh`

  4. Training example: Please refer to `./script/npd_train_demo.sh`

## Note:

1. Thereis a `npddetect::prescandetect` function for faster detection with some lose on recall. Additional parameter `stepR` refers to pre-scan step size compared to the original scan step size ( float more than 1 ) . The `thresR` refers to the threshold to reject the window( float in [0-1] ).

## Result:

+ ROC:

  ![图片1.png-123.8kB][1]

+ Speed:

| image size | window size | cores | time (ms) |
| :---: | :---: | :---: | :---: | :---: |
| 640x480 | 20x20 | 1 | ~50 |


+ ROC for `npddetect::prescandetect`:

![图片2.png-109.5kB][2]

+ Speed for `npddetect::prescandetect`:

| params | image size | window size | time(ms) |
| :---: | :---: | :---: | :---: |
| none | 1920x1080 | 20x20 | 532.400239 |
| stepR = 2, thresR = 0.2 | 1920x1080 | 20x20 | 344.154205 |	
| stepR = 2, thresR = 0.3 | 1920x1080 | 20x20 | 282.128798 |		
| stepR = 3, thresR = 0.2 | 1920x1080 | 20x20 | 286.230415 |	
| stepR = 3, thresR = 0.3 | 1920x1080 | 20x20 | 226.091203 |	
| stepR = 4, thresR = 0.3 | 1920x1080 | 20x20 | 202.147923 |	


#License and Citation

This software is free for noncommercial use. This software is provided "as is", without any warranty of upgradation or customized development. It is your own risk of using this software. The authors are not responsible for any damage caused by using this software. 

## References:

This software is based on the MATLAB edition. Thanks for the work of Liao et al. [Project page](http://www.cbsr.ia.ac.cn/users/scliao/projects/npdface/index.html).

> @article{

>      Author = {Shengcai Liao, Member, IEEE, Anil K. Jain, Fellow, IEEE, and Stan Z. Li, Fellow, IEEE},

>      Title = {A Fast and Accurate Unconstrained Face Detector},

>      Year = {2014}

>  }


  [1]: http://static.zybuluo.com/No-Winter/4n2oqec9hqj80qic2ujebxmn/%E5%9B%BE%E7%89%871.png
  [2]: http://static.zybuluo.com/No-Winter/bzf61zqjs4c1qfilwzds4f7s/%E5%9B%BE%E7%89%872.png