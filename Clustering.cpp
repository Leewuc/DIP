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

Mat myBgr2Hsv(Mat src_img){
    double b,g,r,h =0.0,s,v;
    Mat dst_img(src_img.size(),src_img.type());
    for(int y=0;y<src_img.rows;y++){
        for(int x=0;x<src_img.cols;x++){
            b = (double)src_img.at<Vec3b>(y,x)[0];
            g = (double)src_img.at<Vec3b>(y,x)[1];
            r = (double)src_img.at<Vec3b>(y,x)[2];
            b /= 255.0;
            g /= 255.0;
            r /= 255.0;
            double cMax = max({b,g,r});
            double cMin = min({b,g,r});
            double delta = cMax - cMin;
            v = cMax;
            if(v==0) s = 0;
            else s = (delta / cMax);
            
            if(delta ==0) h = 0;
            else if(cMax == r) h = 60 * (g-b)/(v - cMin);
            else if(cMax == g) h = 120 + 60 * (b-r) / (v - cMin);
            else if(cMax == b) h = 240 + 60 * (r-g) / (v-cMin);
            
            if (h<0) h+= 360;
            
            v *= 255;
            s *= 255;
            h /= 2;
            
            h = h > 255.0 ? 255.0 : h < 0 ? 0 : h;
            s = s > 255.0 ? 255.0 : s < 0 ? 0 :s;
            v = v > 255.0 ? 255.0 : v < 0 ? 0 : v;
            
            dst_img.at<Vec3b>(y,x)[0] = (uchar) h;
            dst_img.at<Vec3b>(y,x)[1] = (uchar) s;
            dst_img.at<Vec3b>(y,x)[2] = (uchar) v;
        }
    }
    return dst_img;
}
Mat myInRange(Mat src_img, Scalar lowerb, Scalar upperb){
    Mat dst_img(src_img.size(),src_img.type());
    for(int y =0; y < src_img.rows;y++){
        for(int x=0;x<src_img.cols;x++){
            double b,g,r;
            
            b = (double)src_img.at<Vec3b>(y,x)[0];
            g = (double)src_img.at<Vec3b>(y,x)[1];
            r = (double)src_img.at<Vec3b>(y,x)[2];
            Scalar bgrScalar(b,g,r);
            bool range = (lowerb[0] <= bgrScalar[0]) && (bgrScalar[0] <= upperb[0]);
            bool range1 = (lowerb[1] <= bgrScalar[1]) && (bgrScalar[1] <= upperb[1]);
            bool range2 = (lowerb[2] <= bgrScalar[2]) && (bgrScalar[2] <= upperb[2]);
            
            if(range && range1 && range2){
                dst_img.at<Vec3b>(y,x)[0] = 255;
                dst_img.at<Vec3b>(y,x)[1] = 255;
                dst_img.at<Vec3b>(y,x)[2] = 255;
            }
            else{
                dst_img.at<Vec3b>(y,x)[0] = 0;
                dst_img.at<Vec3b>(y,x)[1] = 0;
                dst_img.at<Vec3b>(y,x)[2] = 0;
            }
        }
    }
    return dst_img;
}
void CVColorModels(Mat bgr_img){
    Mat gray_img,rgb_img,hsv_img,yuv_img,xyz_img;
    
    cvtColor(bgr_img, gray_img, cv::COLOR_BGR2GRAY);
    cvtColor(bgr_img, rgb_img, cv::COLOR_BGR2RGB);
    cvtColor(bgr_img, hsv_img, cv::COLOR_BGR2HSV);
    cvtColor(bgr_img, yuv_img, cv::COLOR_BGR2YCrCb);
    cvtColor(bgr_img, xyz_img, cv::COLOR_BGR2XYZ);
    
    Mat print_img;
    bgr_img.copyTo(print_img);
    cvtColor(gray_img, gray_img, cv::COLOR_GRAY2BGR);
    hconcat(print_img,gray_img,print_img);
    hconcat(print_img,rgb_img,print_img);
    hconcat(print_img,hsv_img,print_img);
    hconcat(print_img,yuv_img,print_img);
    hconcat(print_img,xyz_img,print_img);
    
    imshow("results",print_img);
    imwrite("fruits.png",print_img);
    waitKey(0);
}
Mat GetCbCr(Mat src_img){
    double b,g,r,y,cb,cr;
    Mat dst_img;
    src_img.copyTo(dst_img);
    
    // <화소 인덱싱>
    for(int row =0;row<dst_img.rows;row++){
        for(int col = 0;col<dst_img.cols;col++){
            // <BGR 획득>
            // OPENCV의 Mat은 BGR순서를 가짐의 유의
            b = (double)dst_img.at<Vec3b>(row,col)[0];
            g = (double)dst_img.at<Vec3b>(row,col)[1];
            r = (double)dst_img.at<Vec3b>(row,col)[2];
            
            // <색상 변환 계산>
            // 정확한 계산을 위해 double 자료형 사용
            y = 0.267 * r + 0.678 * g + 0.0593 * b;
            cb = -0.13963 * r + -0.36037 * g + 0.5 * b;
            cr = 0.5 * r - 0.45979 * g - -0.04021 * b;
            
            // <오버플로우 방지>
            y = y > 255.0 ? 255.0 : y < 0 ? 0 : y;
            cb = cb > 255.0 ? 255.0 : cb < 0 ? 0 : cb;
            cr = cr > 255.0 ? 255.0 : cr < 0 ? 0 : cr;
            
            // <변환된 색상 대입>
            // double 자료형의 값을 본래 자료형으로 변환
            dst_img.at<Vec3b>(row,col)[0] = (uchar)y;
            dst_img.at<Vec3b>(row,col)[1] = (uchar)cb;
            dst_img.at<Vec3b>(row,col)[2] = (uchar)cr;
        }
    }
    return dst_img;
}
void createClustersInfo(Mat imgInput, int n_cluster, vector<Scalar>& clustersCenters,
    vector<vector<Point>>& ptInClusters) {

    RNG random(cv::getTickCount());

    for (int k = 0; k < n_cluster; k++) {
        Point centerKPoint;
        centerKPoint.x = random.uniform(0, imgInput.cols);
        centerKPoint.y = random.uniform(0, imgInput.rows);
        Scalar centerPixel = imgInput.at<Vec3b>(centerKPoint.y, centerKPoint.x);

        Scalar centerK(centerPixel.val[0], centerPixel.val[1], centerPixel.val[2]);
        clustersCenters.push_back(centerK);

        vector<Point>ptInClustersK;
        ptInClusters.push_back(ptInClustersK);
    }
}
double computeColorDistance(Scalar pixel,Scalar clusterPixel){
    double diffBlue = pixel.val[0] - clusterPixel[0];
    double diffGreen = pixel.val[1] - clusterPixel[1];
    double diffRed = pixel.val[2] - clusterPixel[2];
    
    double distance = sqrt(pow(diffBlue,2)+pow(diffGreen,2)+pow(diffRed,2));
    // Euclidian distance
    
    return distance;
}
double adjustClusterCenters(Mat src_img,int n_cluster,vector<Scalar>& clustersCenters, vector<vector<Point>>ptInClusters,double &oldCenter, double newCenter){
    double diffChange;
    for(int k=0;k < n_cluster; k++){
        vector<Point>ptInCluster = ptInClusters[k];
        double newBlue = 0;
        double newGreen = 0;
        double newRed = 0;
        
        for(int i=0; i < ptInCluster.size();i++){
            Scalar pixel = src_img.at<Vec3b>(ptInCluster[i].y,ptInCluster[i].x);
            newBlue += pixel.val[0];
            newGreen += pixel.val[1];
            newRed += pixel.val[2];
        }
        newBlue /= ptInCluster.size();
        newGreen /= ptInCluster.size();
        newRed /= ptInCluster.size();
        
        Scalar newPixel(newBlue,newGreen,newRed);
        newCenter += computeColorDistance(newPixel, clustersCenters[k]);
        clustersCenters[k] = newPixel;
    }
    newCenter /= n_cluster;
    diffChange = abs(oldCenter - newCenter);
    oldCenter = newCenter;
    return diffChange;
}
void findAssociatedCluster(Mat imgInput,int n_cluster,vector<Scalar>clustersCenters,vector<vector<Point>>& ptlnClusters){
    for(int r = 0;r< imgInput.rows;r++){
        for(int c = 0 ;c<imgInput.cols;c++){
            double minDistance = INFINITY;
            int closestClusterIndex = 0;
            Scalar pixel = imgInput.at<Vec3b>(r,c);
            for(int k = 0;k<n_cluster;k++){ // 군집별 계산
                // <각 군집 중앙값과의 차이를 계산>
                Scalar clusterPixel = clustersCenters[k];
                double distance = computeColorDistance(pixel, clusterPixel);
                // <차이가 가장 적은 군집으로 좌표의 군집을 판별>
                if(distance < minDistance){
                    minDistance = distance;
                    closestClusterIndex = k;
                }
            }
            // <좌표 저장>
            ptlnClusters[closestClusterIndex].push_back(Point(c,r));
        }
    }
}
Mat applyFinalClusterToImage(Mat src_img,int n_cluster,vector<vector<Point>> ptInClusters,vector<Scalar> clusterCenters){
    Mat dst_img(src_img.size(),src_img.type());
    for(int k=0; k<n_cluster;k++){
        vector<Point> ptInCluster = ptInClusters[k];
        for(int j=0;j<ptInCluster.size();j++){
            dst_img.at<Vec3b>(ptInCluster[j])[0] = clusterCenters[k].val[0];
            dst_img.at<Vec3b>(ptInCluster[j])[1] = clusterCenters[k].val[1];
            dst_img.at<Vec3b>(ptInCluster[j])[2] = clusterCenters[k].val[2];
        }
    }
    return dst_img;
}
Mat applyFinalClusterToRandomImage(Mat src_img,int n_cluster,vector<vector<Point>>ptInClusters,vector<Scalar> clustersCenters){
    Mat dst_img(src_img.size(),src_img.type());
    
    for(int k=0;k<n_cluster;k++){
        vector<Point> ptInCluster = ptInClusters[k];
        clustersCenters[k].val[0] = rand() % 255;
        clustersCenters[k].val[1] = rand() % 255;
        clustersCenters[k].val[2] = rand() % 255;
        for(int j=0;j<ptInCluster.size();j++){
            dst_img.at<Vec3b>(ptInCluster[j])[0] = clustersCenters[k].val[0];
            dst_img.at<Vec3b>(ptInCluster[j])[1] = clustersCenters[k].val[1];
            dst_img.at<Vec3b>(ptInCluster[j])[2] = clustersCenters[k].val[2];
        }
    }
    return dst_img;
}
