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

Mat saltAndPepper(Mat img,int num){
    for(int n=0;n<num;n++){
        int x = rand() % img.cols;
        int y = rand() % img.rows;
        if(img.channels()==1){
            if(n%2 == 0){
                img.at<char>(y,x) = 255;
            }
            else{
                img.at<char>(y,x) = 0;
            }
        }
        else{
            if(n%2 == 0){
                img.at<Vec3b>(y,x)[0] = 255;
                img.at<Vec3b>(y,x)[1] = 255;
                img.at<Vec3b>(y,x)[2] = 255;
            }
            else{
                img.at<Vec3b>(y,x)[0] = 255;
                img.at<Vec3b>(y,x)[1] = 255;
                img.at<Vec3b>(y,x)[2] = 255;
            }
        }
    }
    return img;
}
Mat mySobelFilter(Mat srcImg){
#if USE_OPENCV
    Mat dstImg(srcImg.size(),CV_8UC1);
    Mat sobelX,sobelY;
    Sobel(srcImg,sobleX,CV_8UC1,1,0); //가로방향 sobel
    Sobel(srcImg,sobelY,CV_8UC1,0,1); //세로방향 sobel
    dstImg = (abs(sobelX)+abs(sobelY))/2; // 두 에지 결과에 절대값 합 형태로 최종결과 도출
    return dstImg;
#else
    int kernelX[3][3] = {   -2,-1,0,
                            -1,0,1,
                            0,1,2}; //가로방향 Sobel 마스크
    int kernelY[3][3] = {  -0,1,2,
                            -1,0,1,
                            -2,-1,0}; //세로 방향 Sobel 마스크
    //마스크 합이 0이 되므로 1로 정규화하는 과정은 필요 없음
    Mat dstImg(srcImg.size(),CV_8UC1);
    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;
    int width = srcImg.cols;
    int height = srcImg.rows;
    for(int y = 0;y<height;y++){
        for(int x = 0;x<width;x++){
            dstData[y*width+x] = (abs(myKernelConv3x3(srcData,kernelX,x,y,width,height))+
                                abs(myKernelConv3x3(srcData,kernelY,x,y,width,height)))/2;
            //두 에지 결과의 절대값 형태로 값 도출
        }
    }
    return dstImg;
#endif
}
