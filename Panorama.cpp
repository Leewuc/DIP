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
Mat makePanorama(Mat img_l, Mat img_r, int thresh_dist, int min_matches) {
    
    Mat img_gray_l, img_gray_r;
    cvtColor(img_l, img_gray_l, COLOR_BGR2GRAY);
    cvtColor(img_r, img_gray_r, COLOR_BGR2GRAY);
    
    Ptr<SurfFeatureDetector> Detector = SURF::create(300);
    vector<KeyPoint> kpts_obj, kpts_scene;
    Detector->detect(img_gray_l, kpts_obj);
    Detector->detect(img_gray_r, kpts_scene);
    
    Mat img_kpts_l, img_kpts_r;
    drawKeypoints(img_gray_l, kpts_obj, img_kpts_l, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    drawKeypoints(img_gray_r, kpts_scene, img_kpts_r, Scalar::all(-1), DrawMatchesFlags::DEFAULT);
    
    imshow("img_kpts_l.png", img_kpts_l);
    imshow("img_kpts_r.png", img_kpts_r);
    waitKey(0);
    
    Ptr<SurfDescriptorExtractor> Extractor = SURF::create(100, 4, 3, false, true);
    
    Mat img_des_obj, img_des_scene;
    Extractor->compute(img_gray_l, kpts_obj, img_des_obj);
    Extractor->compute(img_gray_r, kpts_scene, img_des_scene);
    
    BFMatcher matcher(NORM_L2);
    vector<DMatch> matches;
    matcher.match(img_des_obj, img_des_scene, matches);
    
    Mat img_matches;
    drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches, img_matches, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("img_matches.png", img_matches);
    waitKey(0);
    
    // 대각선으로 매칭 된 것은 잘못된 매칭이므로 정제가 필요
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
    drawMatches(img_gray_l, kpts_obj, img_gray_r, kpts_scene, matches_good, img_matches_good, Scalar::all(-1), Scalar::all(-1), vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    
    imshow("img_matches_good.png", img_matches_good);
    waitKey(0);
    
    vector<Point2f> obj, scene;
    for (int i=0; i<matches_good.size(); i++) {
        obj.push_back(kpts_obj[matches_good[i].queryIdx].pt);
        scene.push_back(kpts_scene[matches_good[i].trainIdx].pt);
    }
    
    Mat mat_homo = findHomography(scene, obj, RANSAC);
    
    Mat img_result;
    warpPerspective(img_r, img_result, mat_homo, Size(img_l.cols * 2, img_l.rows * 1.2), INTER_CUBIC);

    Mat img_pano;
    img_pano = img_result.clone();
    imshow("connect image - img_pano", img_pano);
    waitKey(0);
    imshow("connect image - img_l", img_l);
    waitKey(0);
    
    Mat roi(img_pano, Rect(0, 0, img_l.cols, img_l.rows));
    
    Rect rect = Rect(img_l.cols-30, 0, 60, img_l.rows);

    
    img_l.copyTo(roi);
    medianBlur(img_pano(rect), img_pano(rect), 19);
    
    // 검은 부분 제거
    int cut_x = 0, cut_y = 0;
    for (int y=0; y<img_pano.rows; y++) {
        for (int x=0; x<img_pano.cols; x++) {
            if (img_pano.at<Vec3b>(y, x)[0] == 0 &&
                img_pano.at<Vec3b>(y, x)[1] == 0 &&
                img_pano.at<Vec3b>(y, x)[2] == 0) {
                continue;
            }
            if (cut_x < x) cut_x = x;
            if (cut_y < y) cut_y = y;
        }
    }
    
    Mat img_pano_cut;
    
    img_pano_cut = img_pano(Range(0, cut_y), Range(0, cut_x));
    imshow("img_pano_cut.png", img_pano_cut);
    waitKey(0);
    destroyAllWindows();
    
    return img_pano_cut;
    
}
// Stitcher를 이용한 파노라마 영상 생성
void ex_panorama_simple() {
    Mat img;
    vector<Mat> imgs;
    img = imread("one.jpeg", IMREAD_COLOR);
    imgs.push_back(img);
    img = imread("two.jpeg", IMREAD_COLOR);
    imgs.push_back(img);
    img = imread("three.jpeg", IMREAD_COLOR);
    imgs.push_back(img);
    Mat result;
    Ptr<Stitcher> stitcher = Stitcher::create(Stitcher::PANORAMA);
    Stitcher::Status status = stitcher->stitch(imgs, result);
    if (status != Stitcher::OK) {
        cout << "Can't stitch images, error code = " << int(status) << endl;
        exit(-1);
    }
    imshow("ex_panorama_simple_result", result);
    waitKey(0);
}
void ex_panorama() {
    Mat matImage1 = imread("two.jpeg", IMREAD_COLOR);
    Mat matImage2 = imread("one.jpeg", IMREAD_COLOR);
    Mat matImage3 = imread("three.jpeg", IMREAD_COLOR);
    
    if (matImage1.empty() || matImage2.empty() || matImage3.empty()) exit(-1);
    
    Mat result;
    flip(matImage1, matImage1, 1);
    flip(matImage2, matImage2, 1);
    
    result = makePanorama(matImage1, matImage2, 3, 60);
    flip(result, result, 1);
    result = makePanorama(result, matImage3, 3, 60);
    
    imshow("ex_panorama_result", result);
    waitKey(0);
}
