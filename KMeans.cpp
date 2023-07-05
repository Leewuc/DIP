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

Mat CvKmeans(Mat src_img, int k) {
    // 2차원 영상 -> 1차원 벡터
    Mat samples(src_img.rows * src_img.cols, src_img.channels(), CV_32F);
    for (int y = 0; y < src_img.rows; y++) {
        for (int x = 0; x < src_img.cols; x++) {
            if (src_img.channels() == 3) {
                for (int z = 0; z < src_img.channels(); z++) {
                    samples.at<float>(y + x * src_img.rows, z) = (float)src_img.at<Vec3b>(y, x)[z];
                }
            }
            else {
                samples.at<float>(y + x + src_img.rows) = (float)src_img.at<uchar>(y, x);
            }
        }
    }
    //opencv k-means 수행
    Mat labels; // 군집판별 결과가 담길 1차원 벡터
    Mat centers; // 각 군집의 중앙값(대표값)
    int attempts = 5;
    kmeans(samples, k, labels,
        TermCriteria(TermCriteria::MAX_ITER|TermCriteria::EPS, 10000, 0.0001),
        attempts, KMEANS_PP_CENTERS, centers);

    //1차원 벡터 -> 2차원 영상
    Mat dst_img(src_img.size(), src_img.type());
    for (int y = 0; y < src_img.rows; y++) {
        for (int x = 0; x < src_img.cols; x++) {
            int cluster_idx = labels.at<int>(y + x * src_img.rows, 0);
            if (src_img.channels() == 3) {
                for (int z = 0; z < src_img.channels(); z++) {
                    dst_img.at<Vec3b>(y, x)[z] = (uchar)centers.at<float>(cluster_idx, z);
                    // 군집 판별 결과에 따라 각 군집의 중앙값으로 결과 생성
                }
            }
            else {
                dst_img.at<uchar>(y, x) = (uchar)centers.at<float>(cluster_idx, 0);
            }
        }
    }
    imshow("results - CVkMeans",dst_img);
    return dst_img;
}
Mat MyKmeans(Mat src_img, int n_cluster,int flag = 0) {
    vector<Scalar>clustersCenters;
    vector<vector<Point>>ptInClusters;
    double threshold = 0.001;
    double oldCenter = INFINITY;
    double newCenter = 0;
    double diffChange = oldCenter - newCenter;

    createClustersInfo(src_img, n_cluster, clustersCenters, ptInClusters);

    while (diffChange > threshold) {

        newCenter = 0;
        for (int k = 0; k < n_cluster; k++) { ptInClusters[k].clear(); }

        findAssociatedCluster(src_img, n_cluster, clustersCenters, ptInClusters);

        diffChange = adjustClusterCenters(src_img, n_cluster, clustersCenters, ptInClusters, oldCenter, newCenter);


    }
    Mat dst_img;
    if (flag == 0) dst_img = applyFinalClusterToImage(src_img, n_cluster, ptInClusters, clustersCenters);
    else if(flag == 1) dst_img = applyFinalClusterToRandomImage(src_img, n_cluster, ptInClusters, clustersCenters);
    //waitKey(0);


    return dst_img;
}
