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

Mat detectAngle(Mat img) {
    if(img.empty()) {
        cout << "Empty image!\n";
        exit(-1);
    }
    
    resize(img, img, Size(900, 900), 0, 0, INTER_CUBIC);
    
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    Mat harr;
    cornerHarris(gray, harr, 5, 3, 0.05, BORDER_DEFAULT);
    normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    
    Mat harr_abs;
    convertScaleAbs(harr, harr_abs);
    
    int thresh = 120; // threshold
    Mat result = img.clone();
    
    for (int y=0; y<harr.rows; y++) {
        for (int x=0; x<harr.cols; x++) {
            if((int)harr.at<float>(y, x) > thresh) {
                circle(result, Point(x, y), 7, Scalar(0, 0, 255), -1, 4, 0);
            }
        }
    }
    
    imshow("Target image", result);
    
    for (int y=0; y<result.rows; y++) {
        for (int x=0; x<result.cols; x++) { // B G R 순서
            if((int)result.at<Vec3b>(y, x)[2] != 255) {
                result.at<Vec3b>(y, x)[0] = 255;
                result.at<Vec3b>(y, x)[1] = 255;
                result.at<Vec3b>(y, x)[2] = 255;
            }
        }
    }
    
    
    return result;
}

// blob detection을 사용하여 원의 개수를 세는 함수
int countCircle(Mat img) {
    SimpleBlobDetector::Params params;
    params.minThreshold = 10;
    params.maxThreshold = 300;
    params.filterByArea = true;
    params.minArea = 1;
    params.filterByCircularity = true;
    params.minCircularity = 0.6;
    params.filterByConvexity = true;
    params.minConvexity = 0.9;
    params.filterByInertia = true;
    params.minInertiaRatio = 0.01;
    
    Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
    
    vector<KeyPoint> keypoints;
    detector->detect(img, keypoints);
    
    Mat dst;
    drawKeypoints(img, keypoints, dst, Scalar(255, 255, 255), DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
    
    cout << "꼭짓점의 개수: " << keypoints.size() << endl;
    
    return (int)keypoints.size();
}

// 몇 각형인지 출력
void detectPolygon(int angle) {
    switch (angle) {
        case 3:
            cout << "삼각형입니다." << endl;
            break;
        case 4:
            cout << "사각형입니다." << endl;
            break;
        case 5:
            cout << "오각형입니다." << endl;
            break;
        case 6:
            cout << "육각형입니다." << endl;
            break;
        default:
            cout << "삼각형, 사각형, 오각형, 육각형 중 하나가 아닙니다." << endl;
            break;
    }
}
void printPolygon() {
    // 삼각형
    Mat img = imread("triangle.png", IMREAD_COLOR);
    Mat temp = detectAngle(img);
    int angle = countCircle(temp);
    detectPolygon(angle);
    
    imshow("source image", img);
    imshow("detect Angle", temp);
    waitKey(0);
    
    // 사각형
    img = imread("square.png", IMREAD_COLOR);
    temp = detectAngle(img);
    angle = countCircle(temp);
    detectPolygon(angle);
    
    imshow("source image", img);
    imshow("detect Angle", temp);
    waitKey(0);
    
    // 오각형
    img = imread("pentagon.png", IMREAD_COLOR);
    temp = detectAngle(img);
    angle = countCircle(temp);
    detectPolygon(angle);
    
    imshow("source image", img);
    imshow("detect Angle", temp);
    waitKey(0);
    
    // 육각형
    img = imread("hexagon.png", IMREAD_COLOR);
    temp = detectAngle(img);
    angle = countCircle(temp);
    detectPolygon(angle);
    
    imshow("source image", img);
    imshow("detect Angle", temp);
    waitKey(0);
    
}
void cvFlip(){
    Mat src = imread("Lenna.png",1);
    Mat dst_x,dst_y,dst_xy;
    
    flip(src,dst_x,0);
    flip(src,dst_y,1);
    flip(src,dst_xy,-1);
    
    imwrite("nonflip.jpg",src);
    imwrite("xflip.jpg",dst_x);
    imwrite("yflip.jpg",dst_y);
    imwrite("xyflip.jpg",dst_xy);
    
    imshow("nonflip",src);
    imshow("xflip",dst_x);
    imshow("yflip",dst_y);
    imshow("xyflip",dst_xy);
    waitKey(0);
    destroyAllWindows();
}
void cvRotation(){
    Mat src = imread("Lenna.png",1);
    Mat dst,matrix;
    
    Point center = Point(src.cols / 2, src.rows / 2);
    matrix = getRotationMatrix2D(center, 45.0, 1.0);
    warpAffine(src, dst, matrix, src.size());
    
    imwrite("nonrot.jpg",src);
    imwrite("rot.jpg",dst);
    
    imshow("nonrot",src);
    imshow("rot",dst);
    waitKey(0);
    destroyAllWindows();
}
Mat GetMyRotation(Point center,double angle){
    Mat matrix1,matrix2,matrix3;
    matrix1 = (Mat_<double>(3,3) <<
               1,0,-center.x-100,
               0,1,-center.y+200,
               0,0,1);
    matrix2 = (Mat_<double>(3,3) <<
               cos(angle*CV_PI/180),sin(angle*CV_PI/180),0,
               -sin(angle*CV_PI/180),cos(angle*CV_PI/180),0,
               0,0,1);
    matrix3 = (Mat_<double>(3,3) <<
               1,0,center.x,
               0,1,center.y,
               0,0,1);
    Mat matrix = matrix1 * matrix2 * matrix3;
    return matrix;
}
void MyRotation(){
    Mat src = imread("Lenna.png",1);
    Mat dst;
    Mat matrix;
    
    Point center = Point(src.cols/1000, src.rows/1000);
    matrix = GetMyRotation(center, 45.0);
    warpPerspective(src, dst, matrix, src.size());
    
    imshow("nonrot",src);
    imshow("rot - mine",dst);
    imwrite("rot - mine.png",dst);
    waitKey(0);
    destroyAllWindows();
}
Mat detectAngle2(Mat img) {
    if(img.empty()) {
        cout << "Empty image!\n";
        exit(-1);
    }
    
    resize(img, img, Size(500, 500), 0, 0, INTER_CUBIC);
    
    // 꼭짓점을 제외한 점이 corner로 검출되어 내부 색을 채워주었음
    for (int y=0; y<img.rows; y++) {
        for (int x=0; x<img.cols; x++) {
            bool black = (int)img.at<Vec3b>(y, x)[0] == 0 && (int)img.at<Vec3b>(y, x)[1] == 0 && (int)img.at<Vec3b>(y, x)[2] == 0;
            if(!black) {
                img.at<Vec3b>(y, x)[0] = 196;
                img.at<Vec3b>(y, x)[1] = 114;
                img.at<Vec3b>(y, x)[2] = 68;
            }
        }
    }
    
    Mat gray;
    cvtColor(img, gray, COLOR_BGR2GRAY);
    
    Mat harr;
    cornerHarris(gray, harr, 2, 3, 0.05, BORDER_DEFAULT);
    normalize(harr, harr, 0, 255, NORM_MINMAX, CV_32FC1, Mat());
    
    Mat harr_abs;
    convertScaleAbs(harr, harr_abs);
    
    int thresh = 120; // threshold
    Mat result = img.clone();
    
    for (int y=0; y<harr.rows; y++) {
        for (int x=0; x<harr.cols; x++) {
            if((int)harr.at<float>(y, x) > thresh) {
                circle(result, Point(x, y), 3, Scalar(255, 0, 255), -1, 4, 0);
            }
        }
    }
    
    for (int y=0; y<result.rows; y++) {
        for (int x=0; x<result.cols; x++) {
            bool pink = (int)result.at<Vec3b>(y, x)[0] == 255 && (int)result.at<Vec3b>(y, x)[1] == 0 && (int)result.at<Vec3b>(y, x)[2] == 255;
            if(!pink) {
                result.at<Vec3b>(y, x)[0] = 255;
                result.at<Vec3b>(y, x)[1] = 255;
                result.at<Vec3b>(y, x)[2] = 255;
            }
        }
    }
    return result;
}
void cvAffine(){
    Mat src = imread("Lenna.png",1);
    Mat dst,matrix;
    
    Point2f srcTri[3];
    srcTri[0] = Point2f(0.f,0.f);
    srcTri[1] = Point2f(src.cols - 1.f,0.f);
    srcTri[2] = Point2f(0.f,src.rows - 1.f);
    
    Point2f dstTri[3];
    dstTri[0] = Point2f(0.f,src.rows * 0.33f);
    dstTri[1] = Point2f(src.cols * 0.85f,src.rows * 0.25f);
    dstTri[2] = Point2f(src.cols * 0.15f,src.rows * 0.7f);
    
    matrix = getAffineTransform(srcTri, dstTri);
    warpAffine(src, dst, matrix, src.size());
    
    imwrite("nonaff.jpg",src);
    imwrite("aff.jpg",dst);
    
    imshow("nonaff",src);
    imshow("aff",dst);
    waitKey(0);
    
    destroyAllWindows();
}
void cvPerspective(){
    Mat src = imread("card_per.png",1);
    Mat dst,matrix;
    
    Point2f srcQuad[4];
    srcQuad[0] = Point2f(125.f,133.f);
    srcQuad[1] = Point2f(376.f,178.f);
    srcQuad[2] = Point2f(73.f,356.f);
    srcQuad[3] = Point2f(428.f,321.f);
    
    Point2f dstQuad[4];
    dstQuad[0] = Point2f(0.f,0.f);
    dstQuad[1] = Point2f(src.cols -1.f,0.f);
    dstQuad[2] = Point2f(0.f,src.rows-1.f);
    dstQuad[3] = Point2f(src.cols -1.f,src.rows-1.f);
    
    matrix = getPerspectiveTransform(srcQuad, dstQuad);
    warpPerspective(src, dst, matrix, src.size());
    
    resize(dst,dst,Size(500,300));
    imshow("nonrot",src);
    imshow("row",dst);
    waitKey(0);
    
    destroyAllWindows();
}
