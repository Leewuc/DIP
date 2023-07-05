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

Mat yellow(Mat src_img){
    Scalar lowerYellow = Scalar(20,20,100);
    Scalar upperYellow = Scalar(32,255,255);
    Mat hsv_img = myBgr2Hsv(src_img);
    Mat result_img;
    Mat mask = myInRange(hsv_img, lowerYellow, upperYellow);
    cout << "yellow" << endl;
    return mask;
}
Mat red(Mat src_img){
    Scalar lowerred = Scalar(0,100,100);
    Scalar upperred = Scalar(5,255,255);
    Mat hsv_img = myBgr2Hsv(src_img);
    Mat result_img;
    Mat mask = myInRange(hsv_img, lowerred, upperred);
    cout << "red" << endl;
    return mask;
}
Mat orange(Mat src_img){
    Scalar lowerorange = Scalar(1,190,200);
    Scalar upperorange = Scalar(18,255,255);
    Mat hsv_img = myBgr2Hsv(src_img);
    Mat result_img;
    Mat mask = myInRange(hsv_img, lowerorange, upperorange);
    cout << "orange" << endl;
    return mask;
}
void fruit(){
    cout << "banana.jpeg: ";
    Mat b_img = imread("banana.jpeg",1);
    Mat b_hsv = myBgr2Hsv(b_img);
    Mat b_result;
    Mat b_mask = yellow(b_img);
    cvtColor(b_mask, b_mask, COLOR_BGR2GRAY);
    bitwise_and(b_hsv,b_hsv,b_result,b_mask);
    cvtColor(b_result, b_result, COLOR_HSV2BGR_FULL);
    
    Mat result_img;
    hconcat(b_img,b_result,result_img);
    imshow("test window",result_img);
    waitKey(0);
    destroyWindow("test window");
    
    cout << "strawberry.jpeg: ";
    Mat s_img = imread("strawberry.jpeg",1);
    Mat s_hsv = myBgr2Hsv(s_img);
    Mat s_result;
    Mat s_mask = red(s_img);
    cvtColor(s_mask, s_mask, COLOR_BGR2GRAY);
    bitwise_and(s_hsv,s_hsv,s_result,s_mask);
    cvtColor(s_result,s_result,COLOR_HSV2BGR_FULL);
    
    hconcat(s_img,s_result,result_img);
    imshow("test window",result_img);
    waitKey(0);
    destroyWindow("test window");
    
    cout << "orange.jpeg: ";
    Mat o_img = imread("orange.jpeg",1);
    Mat o_hsv = myBgr2Hsv(o_img);
    Mat o_result;
    Mat o_mask = orange(o_img);
    
    cvtColor(o_mask,o_mask,COLOR_BGR2GRAY);
    bitwise_and(o_hsv,o_hsv,o_result,o_mask);
    cvtColor(o_result,o_result,COLOR_HSV2BGR_FULL);
    hconcat(o_img,o_result,result_img);
    imshow("test window",result_img);
    waitKey(0);
    destroyWindow("test window");
}
void kseg(){
    Mat src_img = imread("banana.jpeg",1);
    //Mat src_img = imread("beach.jpg",1);
    Mat result = MyKmeans(src_img, 5,0);
    Mat randomResult = MyKmeans(src_img, 5,1);
    
    Mat temp;
    hconcat(src_img,result,temp);
    
    Mat result_img;
    hconcat(temp,randomResult,result_img);
    imshow("test window",result_img);
    waitKey(0);
    destroyWindow("test window");
}
void exCVMeanShift() {
    Mat img = imread("fruits.png");
    if (img.empty()) exit(-1);
    cout << " exCvMeanShift() " << endl;
    
    resize(img, img, Size(256, 256), 0, 0, cv::INTER_AREA);
    imshow("src", img);
    imwrite("exCVMeanShift.jpeg", img);
    
    pyrMeanShiftFiltering(img, img, 8, 16);
    
    imshow("Dst", img);
    waitKey();
    destroyAllWindows();
    imwrite("exCVMeanShift_dst.jpeg", img);
}
