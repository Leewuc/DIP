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

Mat warpPers(Mat src){
    Mat dst;
    Point2f src_p[4],dst_p[4];
    //int row = src.rows;
    //int col = src.cols;
    
    src_p[0] = Point2f(0,0);
    src_p[1] = Point2f(512,0);
    src_p[2] = Point2f(0,512);
    src_p[3] = Point2f(512,512);
    
    dst_p[0] = Point2f(0,0);
    dst_p[1] = Point2f(512,0);
    dst_p[2] = Point2f(0,512);
    dst_p[3] = Point2f(412,412);
    
    Mat pers_mat = getPerspectiveTransform(src_p, dst_p);
    warpPerspective(src, dst, pers_mat, Size(512,512));
    return dst;
}
Mat changeBright(Mat src){
    Mat dst = src + Scalar(30,30,30);
    return dst;
}
void warpFeatureSIFT(){
    Mat img = imread("church.png",1);
    Mat gray;
    cvtColor(img,gray,COLOR_BGR2GRAY);
    Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    vector<KeyPoint> keypoints;
    detector->detect(gray, keypoints);
    Mat result;
    drawKeypoints(img, keypoints, result);
    Mat warp_img = changeBright(img);
    warp_img = warpPers(warp_img);
    Mat warp_gray;
    cvtColor(warp_img, warp_gray, COLOR_BGR2GRAY);
    
    Ptr<SiftFeatureDetector> warp_detector = SiftFeatureDetector::create();
    vector<KeyPoint>warp_keypoints;
    detector->detect(warp_gray,warp_keypoints);
    Mat warp_result;
    drawKeypoints(warp_img, warp_keypoints, warp_result);
    
    Mat final;
    hconcat(result,warp_result,final);
    imshow("compare img",final);
    waitKey(0);
    destroyAllWindows();
}
