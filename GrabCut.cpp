#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <time.h>
#include <sstream>
#include <cstdlib>

#include "opencv2/core/core.hpp"   // Mat class와 각종 data structure 및 산술 루틴을 포함하는 헤더
#include "opencv2/highgui/highgui.hpp" // GUI와 관련된 요소를 포함하는 헤더(imshow 등)
#include "opencv2/imgproc/imgproc.hpp" // 각종 이미지 처리 함수를 포함하는 헤더
#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/stitching.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/photo.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/video/tracking.hpp>
#define OPENCV_TRAITS_ENABLE_DEPRECATED
using namespace cv;
using namespace std;
using namespace cv::xfeatures2d;

void grabcut(){
    //Mat image = imread("hamster.jpg");
    Mat image = imread("white2.jpeg");
    namedWindow("original image");
    imshow("original image",image);
    imwrite("origin img.jpg",image);
    //Rect rectangle(400,13,500,300);
    Rect rectangle(10,20,20,30);
    
    Mat result;
    Mat bgModel,fgModel;
    grabCut(image,result,rectangle,bgModel,fgModel,5,GC_INIT_WITH_RECT);
    compare(result,GC_PR_FGD,result,CMP_EQ);
    Mat foreground(image.size(),CV_8UC3,Scalar(255,255,255));
    image.copyTo(foreground,result);
    namedWindow("Result");
    imshow("Result",result);
    imwrite("result image.jpg",result);
    namedWindow("foreground");
    imshow("foreground",foreground);
    imwrite("foreground image.jpg",foreground);
    waitKey(0);
    destroyAllWindows();
}
