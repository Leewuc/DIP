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

Mat myCopy(Mat srcImg){
    int width = srcImg.cols;
    int height = srcImg.rows;
    Mat dstImg(srcImg.size(),CV_8UC1); //입력영상과 동일한 크기의 Mat 생성
    uchar* srcData = srcImg.data; //Mat 객체의 data를 가리키는 포인터
    uchar* dstData = dstImg.data;
    for(int y = 0;y < height;y++){
        for(int x = 0; x < width;x++){
            dstData[y*width+x] = srcData[y*width+x];
            //화소 값을 일일이 읽어와 다른 배열에 저장
        }
    }
    return dstImg;
}
int myKernelConv3x3(uchar* arr,int kernel[][3],int x,int y,int width,int height){
    int sum = 0;
    int sumKernel = 0;
    //특정화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
    for(int j = -1; j <= 1; j++){
        for(int i = -1;i<=1;i++){
            if((y+j)>=0 && (y+j)<height && (x+i) >= 0 && (x+i) < width){
                //영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
                sum += arr[(y+j)*width + (x+i)] * kernel[i+1][j+1];
                sumKernel += kernel[i+1][j+1];
            }
        }
    }
    if(sumKernel != 0){
        return sum / sumKernel;
        //합이 정규화되도록 해 영상의 밝기 변화 방지
    }
    else{
        return sum;
    }
}
int myColorKernelConv3x3(uchar* arr,int kernel[][3],int col,int row,int k,int width,int height){
    int sum = 0;
    int sumKernel = 0;
    //특정화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
    for(int j = -1; j <= 1; j++){
        for(int i = -1;i<=1;i++){
            if((row+j) >=0 && (row+j) < height && (col+i) >= 0 && (col+i) < width){
                //영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
                int color = arr[(row+j) *3 *width+(col+i)*3+k];
                sum += color * kernel[i+1][j+1];
                sumKernel += kernel[i+1][j+1];
            }
        }
    }
    return sum/sumKernel;
}
int myKernelConv9x9(uchar* arr,int kernel[][9],int x,int y,int width,int height){
    int sum = 0;
    int sumKernel = 0;
    //특정화소의 모든 이웃화소에 대해 계산하도록 반복문 구성
    for(int j = -1; j <= 1; j++){
        for(int i = -1;i<=1;i++){
            if((y+j)>=0 && (y+j)<height && (x+i) >= 0 && (x+i) < width){
                //영상 가장자리에서 영상 밖의 화소를 읽지 않도록 하는 조건문
                sum += arr[(y+j)*width + (x+i)] * kernel[i+1][j+1];
                sumKernel += kernel[i+1][j+1];
            }
        }
    }
    if(sumKernel != 0){
        return sum / sumKernel;
        //합이 정규화되도록 해 영상의 밝기 변화 방지
    }
    else{
        return sum;
    }
}
