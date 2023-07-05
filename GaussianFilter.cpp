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

Mat myGaussianFilter(Mat srcImg){
#if USE_OPENCV //OPENCV로 구현한것
    Mat dstImg(srcImg.size(),CV_8UC1);
    cv::GaussianBlur(srcImg,dstImg,Size(3,3),0);
    //마스크의 크기를 지정하면 자체적으로 마스크 생성 후 연산
    return dstImg;
#else //직접 구현한것(매크로에 의해 컴파일시 선택됨
    int width = srcImg.cols;
    int height = srcImg.rows;
    int kernel[9][9] = { 1,2,1,3,4,5,6,3,1,
                         2,4,2,3,4,5,2,1,1,
                         1,3,5,20,20,2,1,1,
                         1,2,2,3,4,5,2,1,1,
                         1,3,1,5,8,9,10,15,1,
                         1,4,3,25,13,21,3,1,
                         1,4,5,3,2,3,4,2,1,
                         1,2,1,4,5,3,2,1,2,
                         1,2,1,-10,-20,-15,1,2,3}; // 3x3 형태의 Gaussian 마스크 배열
    Mat dstImg(srcImg.size(),CV_8UC1);
    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;
    for(int y=0;y<height;y++){
        for(int x = 0; x < width;x++){
            dstData[y*width+x] = myKernelConv9x9(srcData,kernel,x,y,width,height);
        }
    }
    return dstImg;
#endif
}
Mat myColorGaussianFilter(Mat srcImg){
#if USE_OPENCV //OPENCV로 구현한것
    Mat dstImg(srcImg.size(),CV_8UC1);
    cv::GaussianBlur(srcImg,dstImg,Size(3,3),0);
    //마스크의 크기를 지정하면 자체적으로 마스크 생성 후 연산
    return dstImg;
#else //직접 구현한것(매크로에 의해 컴파일시 선택됨
    int width = srcImg.cols;
    int height = srcImg.rows;
    int kernel[3][3] = { 1,2,1,
                         2,5,1,
                         1,2,1}; // 3x3 형태의 Gaussian 마스크 배열
    Mat dstImg(srcImg.size(),CV_8UC3);
    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;
    for(int y=0;y<height;y++){
        for(int x=0;x<width;x++){
            int index = y *width *3 + x*3;
            int index2 = (y*2) *(width*2) *3 + (x*2)*3;
            for(int k = 0; k < 3;k++){
                dstData[index+k] = myColorKernelConv3x3(srcData,kernel,x,y,k,width,height);
            }
        }
    }
    return dstImg;
#endif
}
