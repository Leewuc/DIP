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

vector<Point2f> countCircle2(Mat img) {
    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 300;
    params.filterByArea = true;
    params.minArea = 10;
    params.filterByCircularity = true;
    params.minCircularity = 0.895;
    params.filterByConvexity = true;
    params.minConvexity = 0.4;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;
    
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    
    vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);
    
    Mat dst;
    drawKeypoints(img, keypoints, dst, Scalar(0, 255, 0), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    vector<Point2f> centers;
    for(auto keypoint : keypoints){
        centers.push_back(keypoint.pt);
    }
    return centers;
}
Mat myTransMat(){
    Mat matrix1 = (Mat_<double>(3,3) <<
                   1,tan(45*CV_PI/180),0,
                   0,1,0,
                   0,0,1);
    Mat matrix2 = (Mat_<double>(3,3) <<
                   1,0,-256,
                   0,1,0,
                   0,0,1);
    Mat matrix3 = (Mat_<double>(3,3) <<
                   0.5,0,0,
                   0,0.5,0,
                   0,0,1);
    return matrix3 * matrix2 * matrix1;
}
void cvPerspective1(){
    Mat src = imread("Lenna.png",1);
    Mat dst,matrix;
    
    matrix = myTransMat();
    warpPerspective(src,dst,matrix,src.size());
    
    imwrite("nonper.jpg",src);
    imwrite("per.jpg",dst);
    
    imshow("nonper",src);
    imshow("per",dst);
    waitKey(0);
    
    destroyAllWindows();
}
void MyPerspective(){
    Mat src = imread("card_per.png",1);
    Mat dst;
    Mat matrix;
    Mat img = detectAngle2(src);
    vector<Point2f> centers = countCircle2(img);
    Point2f srcQuad[4];
    srcQuad[0] = centers[3];
    srcQuad[1] = centers[2];
    srcQuad[2] = centers[0];
    srcQuad[3] = centers[1];
    
    Point2f dstQuad[4];
    dstQuad[0] = Point2f(0.f,0.f);
    dstQuad[1] = Point2f(src.cols-1.f,0.f);
    dstQuad[2] = Point2f(0.f,src.rows - 1.f);
    dstQuad[3] = Point2f(src.cols - 1.f,src.rows-1.f);
    
    matrix = getPerspectiveTransform(srcQuad, dstQuad);
    warpPerspective(src,dst,matrix,src.size());
    resize(dst,dst,Size(1000,1000));
    
    imshow("nonrot",src);
    imshow("rot",dst);
    waitKey(0);
    destroyAllWindows();
}
