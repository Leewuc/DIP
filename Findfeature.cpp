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
void findBook(Mat img) {
    Mat scene = imread("scene.png",1);
    
    // 특징점 추출
    Mat img_gray;
    Mat scene_gray;
    
    cvtColor(img, img_gray, COLOR_BGR2GRAY);
    cvtColor(scene, scene_gray, COLOR_BGR2GRAY);
    
    Ptr<SiftFeatureDetector> detector = SiftFeatureDetector::create();
    vector<KeyPoint> img_keypoints, scene_keypoints;
    detector->detect(img_gray, img_keypoints);

    detector = SiftFeatureDetector::create();
    detector->detect(scene_gray, scene_keypoints);
    
    Mat img_result;
    drawKeypoints(img, img_keypoints, img_result);
    
    Mat scene_result;
    drawKeypoints(scene, scene_keypoints, scene_result);
    
    imshow("sift_img_result", img_result);
    waitKey(0);
    
    imshow("sift_img_result", scene_result);
    waitKey(0);
    
    // Brute Force 매칭
    Ptr<SiftDescriptorExtractor> extractor = SiftDescriptorExtractor::create();
    Mat img_des_obj, img_des_scene;
    
    extractor->compute(img_gray, img_keypoints, img_des_obj);
    extractor->compute(scene_gray, scene_keypoints, img_des_scene);
    
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(img_des_obj, img_des_scene, matches);
    
    Mat img_matches;
    drawMatches(img_gray, img_keypoints, scene_gray, scene_keypoints, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("img_matches", img_matches);
    waitKey(0);
    
    // 매칭 결과 정제
    double thresh_dist = 3;
    int min_matches = 60;
    
    double dist_max = matches[0].distance;
    double dist_min = matches[0].distance;
    double dist;
    
    for (int i=0; i<img_des_obj.rows; i++) {
        dist = matches[i].distance;
        if (dist < dist_min) dist_min = dist;
        if (dist > dist_max) dist_max = dist;
    }
    
    cout << "max dist: " << dist_max << endl;
    cout << "min dist: " << dist_min << endl;
    
    vector<DMatch> matches_good;
    do {
        vector<DMatch> good_matches2;
        for (int i=0; i<img_des_obj.rows; i++) {
            if (matches[i].distance < thresh_dist * dist_min) {
                good_matches2.push_back(matches[i]);
            }
        }
        matches_good = good_matches2;
        thresh_dist -= 1;
    } while (thresh_dist != 2 && matches_good.size() > min_matches);
    
    Mat img_matches_good;
    drawMatches(img_gray, img_keypoints, scene_gray, scene_keypoints, matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("img_matches_good.png", img_matches_good);
    waitKey(0);
    
    
    vector<Point2f> imgVector, sceneVector;
    if (matches_good.size()>=4) {
        cout << "Object found" << endl;
        for (int i=0; i<matches_good.size(); i++) {
            imgVector.push_back(img_keypoints[matches_good[i].queryIdx].pt);
            sceneVector.push_back(scene_keypoints[matches_good[i].trainIdx].pt);
        }
    }
    
    Mat H = findHomography(imgVector, sceneVector, RANSAC);
    
    // 책의 윤곽 그리기
    vector<Point2f> imgCorners, sceneCorners;
    imgCorners.push_back(Point2f(0.f, 0.f));
    imgCorners.push_back(Point2f(img.cols - 1.f, 0.f));
    imgCorners.push_back(Point2f(img.cols - 1.f, img.rows - 1.f));
    imgCorners.push_back(Point2f(0.f, img.rows - 1.f));
    perspectiveTransform(imgCorners, sceneCorners, H);
    
    vector<Point> dstCorners;
    for (Point2f pt : sceneCorners) {
        dstCorners.push_back(Point(cvRound(pt.x + img.cols), cvRound(pt.y)));
    }
    
    polylines(img_matches_good, dstCorners, true, Scalar(0, 255, 0), 3, LINE_AA);
    
    imshow("result", img_matches_good);
    waitKey(0);
    destroyAllWindows();
    
}
void readImagesAndTimes(vector<Mat>& images,vector<float>& times){
    int numImages = 4;
    //static const float timesArray[] = {1 / 30.0f, 0.25f, 2.5f, 15.0f};
    static const float timesArray[] = {2.0f, 1.0f, 0.25f, 1/30.0f};
    times.assign(timesArray,timesArray + numImages);
    //static const char* filenames[] = {"img_0.033.jpg","img_0.25.jpg","img_2.5.jpg","img_15.jpg"};
    static const char* filenames[] = {"exp2.jpeg","exp1.jpeg","exp0.jpeg","exp-1.jpeg"};
    for(int i=0;i<numImages;i++){
        Mat im = imread(filenames[i]);
        images.push_back(im);
    }
}
