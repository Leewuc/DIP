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

void myBlobDetection(){
    Mat src,src_gray,dst;
    src = cv::imread("butt.jpg",1);
    cv::cvtColor(src,src_gray,COLOR_RGB2GRAY);
    
    int gau_ksize = 11;
    int lap_ksize = 3;
    int lap_scale = 1;
    int lap_delta = 1;
    
    GaussianBlur(src_gray, src_gray, Size(gau_ksize,gau_ksize), 3,3,BORDER_DEFAULT);
    Laplacian(src_gray, dst, CV_64F,lap_ksize,lap_scale,lap_delta,BORDER_DEFAULT);
    // Gaussian + Laplacian -> LoG
    
    normalize(-dst,dst,0,255,NORM_MINMAX,CV_8U,Mat());
    
    imwrite("my_log_dst.png",dst);
    imshow("Original Image",src);
    imshow("Laplacian Image",dst);
    waitKey(0);
    destroyAllWindows();
}
void cvFeaturesSIFT(){
    const Mat img = imread("church.png",1);
    Mat gray;
    cvtColor(img,gray,COLOR_BGR2GRAY);
    
    cv::Ptr<cv::SiftFeatureDetector> detector = cv::SiftFeatureDetector::create();
    std::vector<cv::KeyPoint> keypoints;
    detector->detect(gray,keypoints);
    Mat result;
    drawKeypoints(img,keypoints,result,Scalar::all(-1),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    imwrite("sift_result_jpg",result);
    imshow("Sift result",result);
    waitKey(0);
    destroyAllWindows();
}
void cvBlobDetection(){
    Mat img = imread("coin.png",IMREAD_COLOR);
    // <Set params>
    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 300;
    params.filterByArea = true;
    params.minArea = 10;
    params.maxArea = 9000;
    params.filterByCircularity = true;
    params.minCircularity = 0.8064;
    params.filterByConvexity = true;
    params.minConvexity = 0.9;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;
    
    // <Set blob detecotr>
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    // <detect blobs>
    std::vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);
    // <draw blobs>
    Mat result;
    drawKeypoints(img,keypoints,result,Scalar(255,0,255),DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    cout << "coin count : " << keypoints.size() << '\n';
    imshow("keypoints",result);
    waitKey(0);
    destroyAllWindows();
}
