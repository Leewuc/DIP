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

void myMedian(const Mat& src_img,Mat& dst_img,const Size& kn_size){
    dst_img = Mat::zeros(src_img.size(),CV_8UC1);
    int wd = src_img.cols;
    int hg = src_img.rows;
    int kwd = kn_size.width;
    int khg = kn_size.height;
    int rad_w = kwd/2;
    int rad_h = khg/2;
    uchar* src_data = (uchar*)src_img.data;
    uchar* dst_data = (uchar*)dst_img.data;
    float* table = new float[kwd*khg](); // 커널 테이블 동적할당
    float tmp;
    // <픽셀 인덱싱(가장자리 제외)>
     for(int c= rad_w+1; c<wd - rad_w;c++){
            for(int r=rad_h+1;r<hg - rad_h;r++){
                for(int y=0;y<5;y++){
                    for(int x=0;x<5;x++){
                            table[x+y*5] = src_data[c - 1 + x + (r-1+y) * 256];
                    }
                }
                sort(table,table+25);
                tmp = table[13];
                dst_data[c+r*256] = tmp;
                }
        }
    delete[] table; //동적할당 해제
}
void doMedianEx(){
    cout << "--- doMedianEx() --- \n" << endl;
    // <input>
    Mat src_img = imread("salt_pepper2.png",0);
    if(!src_img.data) printf("No image data \n");
    // <Median 필터링 수행>
    Mat dst_img;
    myMedian(src_img, dst_img, Size(5,5));
    // <output>
    Mat result_img;
    hconcat(src_img,dst_img,result_img);
    imshow("doMedianEx()",result_img);
    waitKey(0);
    destroyWindow("doMedianEx()");
}

void followEdges(int x,int y,Mat &magnitude,int tUpper,int tLower,Mat &edges){
    edges.at<float>(y,x) = 255;
    // <이웃 픽셀 인덱싱>
    for(int i=-1;i<2;i++){
        for(int j=-1;j<2;j++){
            if(( i!= 0)&&(j!=0)&&(x+i>=0)&&(y+j>=0)&&(x+i< magnitude.cols)&&(y+j<magnitude.rows)){
                if((magnitude.at<float>(y+j,x+i)>tLower)&&(edges.at<float>(y+j,x+i)!=255)){
                    followEdges(x+i, y+j, magnitude, tUpper, tLower, edges);
                    //재귀적 방법으로 이웃 픽셀에서 불확실한 edge를 찾아 edge로 규정
                }
            }
        }
    }
}
void edgeDetect(Mat &magnitude,int tUpper,int tLower,Mat &edges){
    int rows = magnitude.rows;
    int cols = magnitude.cols;
    edges = Mat(magnitude.size(),CV_32F,0.0);
    // <픽셀 인덱싱>
    for(int x=0;x<cols;x++){
        for(int y=0;y<rows;y++){
            if(magnitude.at<float>(y,x) >= tUpper){
                followEdges(x, y, magnitude, tUpper, tLower, edges);
                //edgerk 확실하면 이와 연결된 불확실한 edge를 탐색
            }
        }
    }
}
void nonMaxSuppression(Mat &magnitudeImage,Mat &directImage){
    Mat checkImage = Mat(magnitudeImage.rows,magnitudeImage.cols,CV_8U);
    MatIterator_<float>itMag = magnitudeImage.begin<float>();
    MatIterator_<float>itDirection = directImage.begin<float>();
    MatIterator_<unsigned char>itRet = checkImage.begin<unsigned char>();
    MatIterator_<float>itEnd = magnitudeImage.end<float>();
    
    for(; itMag != itEnd; ++itDirection,++itRet, ++itMag){
        const Point pos = itRet.pos();
        float currentDirection = atan(*itDirection)* (180/3.142);
        while(currentDirection < 0) currentDirection += 180;
        *itDirection = currentDirection;
        if(currentDirection > 22.5 && currentDirection <= 67.5){
            if(pos.y > 0 && pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y-1,pos.x-1)){
                magnitudeImage.at<float>(pos.y,pos.x) = 0;
            }
            if(pos.y<magnitudeImage.rows -1 && pos.x < magnitudeImage.cols - 1 && *itMag <= magnitudeImage.at<float>(pos.y+1,pos.x+1)){
                magnitudeImage.at<float>(pos.y,pos.x) = 0;
            }
        }
        else if(currentDirection > 67.5 && currentDirection <= 112.5){
            if(pos.y > 0 && *itMag <= magnitudeImage.at<float>(pos.y-1,pos.x)){
                magnitudeImage.at<float>(pos.y,pos.x) = 0;
            }
            if(pos.y < magnitudeImage.rows-1 && *itMag <= magnitudeImage.at<float>(pos.y+1,pos.x)){
                magnitudeImage.at<float>(pos.y,pos.x) = 0;
            }
        }
        else if(currentDirection > 112.5 && currentDirection <= 157.5){
            if(pos.y > 0 && pos.x < magnitudeImage.cols-1&&*itMag <= magnitudeImage.at<float>(pos.y-1,pos.x+1)){
                magnitudeImage.at<float>(pos.y,pos.x) = 0;
            }
            if(pos.y < magnitudeImage.rows-1&&pos.x>0&&*itMag <= magnitudeImage.at<float>(pos.y+1,pos.x-1)){
                magnitudeImage.at<float>(pos.y,pos.x) = 0;
            }
        }
        else {
            if(pos.x > 0 && *itMag <= magnitudeImage.at<float>(pos.y,pos.x-1)){
                magnitudeImage.at<float>(pos.y,pos.x) = 0;
            }
            if(pos.x < magnitudeImage.cols-1 && *itMag <= magnitudeImage.at<float>(pos.y,pos.x+1)){
                magnitudeImage.at<float>(pos.y,pos.x) = 0;
            }
        }
    }
}
void doEdgeEx(){
    cout << "--- doEdgeEx() --- \n" << endl;
    // <입력>
    Mat src_img = imread("edge_test.jpg",0);
    if(!src_img.data) printf("No image data \n");
    // <잡음 제거>
    Mat blur_img;
    myGaussian(src_img, blur_img, Size(5,5));
    // <커널 생성>
    float kn_data[] = { 1.f,0.f,-1.f,
                        1.f,0.f,-1.f,
                        1.f,0.f,-1.f};
    Mat kn(Size(3,3),CV_32FC1,kn_data);
    cout << "Edge kernel : \n" << kn << endl;
    // <커널 컨볼루젼 수행>
    Mat dst_img;
    myKernelConv(blur_img, dst_img, kn);
    // <출력>
    Mat result_img;
    hconcat(src_img,dst_img,result_img);
    imshow("doEdgeEx()",result_img);
    waitKey(0);
}
