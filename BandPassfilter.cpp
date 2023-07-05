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

Mat doDft(Mat srcImg){
    Mat floatImg;
    srcImg.convertTo(floatImg, CV_32F);
    Mat complexImg;
    dft(floatImg,complexImg,DFT_COMPLEX_OUTPUT);
    return complexImg;
}
Mat getMagnitude(Mat complexImg){
    Mat planes[2];
    split(complexImg,planes);
    //실수부, 허수부 분리
    Mat magImg;
    magnitude(planes[0], planes[1], magImg);
    magImg += Scalar::all(1);
    log(magImg,magImg);
    //magnitude 취득
    // log(1+sqrt(Re(DFT(I))^2 + Im(DFT(I))^2))
    return magImg;
}
Mat myNormalize(Mat src){
    Mat dst;
    src.copyTo(dst);
    normalize(dst,dst,0,255,NORM_MINMAX);
    dst.convertTo(dst, CV_8UC1);
    return dst;
}
Mat getPhase(Mat complexImg){
    Mat planes[2];
    split(complexImg,planes);
    //실수부, 허수부 분리
    Mat phaImg;
    phase(planes[0], planes[1], phaImg);
    //phase 취득
    return phaImg;
}
Mat centralize(Mat complex){
    Mat planes[2];
    split(complex,planes);
    int cx = planes[0].cols / 2;
    int cy = planes[1].rows / 2;
    
    Mat q0Re(planes[0],Rect(0,0,cx,cy));
    Mat q1Re(planes[0],Rect(cx,0,cx,cy));
    Mat q2Re(planes[0],Rect(0,cy,cx,cy));
    Mat q3Re(planes[0],Rect(cx,cy,cx,cy));
    
    Mat tmp;
    q0Re.copyTo(tmp);
    q3Re.copyTo(q0Re);
    tmp.copyTo(q3Re);
    q1Re.copyTo(tmp);
    q2Re.copyTo(q1Re);
    tmp.copyTo(q2Re);
    
    Mat q0Im(planes[1],Rect(0,0,cx,cy));
    Mat q1Im(planes[1],Rect(cx,0,cx,cy));
    Mat q2Im(planes[1],Rect(0,cy,cx,cy));
    Mat q3Im(planes[1],Rect(cx,cy,cx,cy));
    
    q0Im.copyTo(tmp);
    q3Im.copyTo(q0Im);
    tmp.copyTo(q3Im);
    q1Im.copyTo(tmp);
    q2Im.copyTo(q1Im);
    tmp.copyTo(q2Im);
    
    Mat centerComplex;
    merge(planes,2,centerComplex);
    return centerComplex;
}
/*
void centralize(Mat magImg){
    magImg = magImg(Rect(0,0,magImg.cols&-2,magImg.rows&-2));
    int cx = magImg.cols/2;
    int cy = magImg.rows/2;
    Mat q0(magImg,Rect(0,0,cx,cy));
    Mat q1(magImg,Rect(cx,0,cx,cy));
    Mat q2(magImg,Rect(0,cy,cx,cy));
    Mat q3(magImg,Rect(cx,cy,cx,cy));
    Mat tmp;
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);
    q1.copyTo(tmp);
    q2.copyTo(q1);
    tmp.copyTo(q2);
}
 */
Mat multiplyDFT(Mat complexImg,Mat kernel){
    centralize(complexImg);
    mulSpectrums(complexImg, kernel, complexImg, DFT_ROWS);
    return complexImg;
}
Mat setComplex(Mat magImg,Mat phaImg){
    exp(magImg,magImg);
    magImg -= Scalar::all(1);
    //magnitude 계산을 반대로 수행
    
    Mat planes[2];
    polarToCart(magImg,phaImg,planes[0],planes[1]);
    //극 좌표계 -> 직교 좌표계(각도와 크기로부터 2차원 좌표)
    
    Mat complexImg;
    merge(planes,2,complexImg);
    //실수부, 허수부 합체
    return complexImg;
}
Mat doIdft(Mat complexImg){
    centralize(complexImg);
    Mat idftcvt;
    idft(complexImg,idftcvt);
    //IDFT를 위한 원본 영상 취득
    
    Mat planes[2];
    split(idftcvt,planes);
    
    Mat dstImg;
    magnitude(planes[0],planes[1],dstImg);
    normalize(dstImg,dstImg,255,0,NORM_MINMAX);
    dstImg.convertTo(dstImg, CV_8UC1);
    //일반영상의 type과 표현범위로 변환
    return dstImg;
}
Mat getFilterKerenl(Mat Img){
    Mat planes[] = {Mat_<float>(Img),Mat::zeros(Img.size(),CV_32F)};
    Mat kernel;
    merge(planes,2,kernel);
    return kernel;
}
Mat padding(Mat Img){
    int dftRows = getOptimalDFTSize(Img.rows);
    int dftCols = getOptimalDFTSize(Img.cols);
    
    Mat padded;
    copyMakeBorder(Img, padded, 0, dftRows - Img.rows, 0, dftCols -Img.cols, BORDER_CONSTANT,Scalar::all(0));
    return padded;
}
Mat doLPF(Mat srcImg){
    // <DFT>
    Mat padImg = padding(srcImg);
    Mat complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    Mat magImg = getMagnitude(centerComplexImg);
    Mat phaImg = getPhase(centerComplexImg);
    
    // <LPF>
    double minVal,maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg, &minVal,&maxVal,&minLoc,&maxLoc);
    normalize(magImg,magImg,0,1,NORM_MINMAX);
    
    Mat maskImg = Mat::zeros(magImg.size(),CV_32F);
    circle(maskImg,Point(maskImg.cols / 2,maskImg.rows / 2),20,Scalar::all(1),-1,-1,0);
    
    Mat magImg2;
    multiply(magImg, maskImg, magImg2);
    
    // <IDFT>
    normalize(magImg2,magImg2,(float)minVal,(float)maxVal,NORM_MINMAX);
    Mat complexImg2 = setComplex(magImg2, phaImg);
    Mat dstImg = doIdft(complexImg2);
    
    return myNormalize(dstImg);
}
Mat doLPF1(Mat complexImg,int r){
    Mat Img = Mat::zeros(complexImg.rows, complexImg.cols, CV_32F);
    circle(Img,Point(Img.cols/2,Img.rows/2),r,Scalar(1),-1,8);
    return Img;
}
Mat doBPF1(Mat complexImg,int r,int r1){
    Mat BandPassFilter = doLPF1(complexImg, r1) - doLPF1(complexImg, r);
    imshow("BandPassFilter",BandPassFilter);
    return BandPassFilter;
}
void Filter2DFreq(const Mat& inputImg,Mat& outputImg,const Mat& H){
    Mat planes[2] = {Mat_<float>(inputImg.clone()),Mat::zeros(inputImg.size(),CV_32F)};
    Mat complexImg;
    merge(planes,2,complexImg);
    dft(complexImg,complexImg,DFT_SCALE);
    Mat planes1[2] = {Mat_<float>(H.clone()),Mat::zeros(H.size(),CV_32F)};
    Mat complexh;
    merge(planes1,2,complexh);
    Mat complexImgh;
    mulSpectrums(complexImg,complexh,complexImgh,0);
    idft(complexImgh,complexImgh);
    split(complexImgh,planes);
    outputImg = planes[0];
}
void synthesize(Mat &inputoutput,Point center,int r){
    Point c = center,c1 = center,c2 = center;
    c.y = inputoutput.rows - center.y;
    c1.x = inputoutput.cols - center.x;
    c2 = Point(c1.x,c.y);
    circle(inputoutput,center,r,0,-1,8);
    circle(inputoutput,c,r,0,-1,8);
    circle(inputoutput,c1,r,0,-1,8);
    circle(inputoutput,c2,r,0,-1,8);
}
void calcpsd(const Mat &inputI,Mat& outputI){
    Mat planes[2] = {Mat_<float>(inputI.clone()),Mat::zeros(inputI.size(),CV_32F)};
    Mat complexImg;
    merge(planes,2,complexImg);
    dft(complexImg,complexImg);
    split(complexImg,planes);
    planes[0].at<float>(0) = 0;
    planes[1].at<float>(0) = 0;
    Mat img;
    magnitude(planes[0],planes[1],img);
    pow(img,2,img);
    outputI = img;
}
Mat FreqDomain(const Mat&img,const Mat&mask){
    Mat result(img.rows,img.cols,CV_32FC2,Scalar::all(0));
    for(int y=0;y<img.rows;y++){
        for(int x=0;x<img.cols;x++){
            result.at<Vec2f>(y,x)[0] = img.at<Vec2f>(y,x)[0]*mask.at<Vec2f>(y,x)[0] - img.at<Vec2f>(y,x)[1]*mask.at<Vec2f>(y,x)[1];
            result.at<Vec2f>(y,x)[1] = img.at<Vec2f>(y,x)[0]*mask.at<Vec2f>(y,x)[1] - img.at<Vec2f>(y,x)[1]*mask.at<Vec2f>(y,x)[0];
        }
    }
    return result;
}
Mat doHPF(Mat srcImg){
    // <DFT>
    Mat padImg = padding(srcImg);
    Mat complexImg = doDft(padImg);
    Mat centerComplexImg = centralize(complexImg);
    Mat magImg = getMagnitude(centerComplexImg);
    Mat phaImg = getPhase(centerComplexImg);
    
    // <LPF>
    double minVal, maxVal;
    Point minLoc, maxLoc;
    minMaxLoc(magImg,&minVal,&maxVal,&minLoc,&maxLoc);
    normalize(magImg,magImg,0,1,NORM_MINMAX);
    
    Mat maskImg = Mat::ones(magImg.size(),CV_32F);
    circle(maskImg,Point(maskImg.cols / 2,maskImg.rows / 2),50,Scalar::all(0),-1,-1,0);
    
    Mat magImg2;
    multiply(magImg, maskImg, magImg2);
    
    // <IDFT>
    normalize(magImg2,magImg2,(float)minVal,(float)maxVal,NORM_MINMAX);
    Mat complexImg2 = setComplex(magImg2, phaImg);
    Mat dstImg = doIdft(complexImg2);
    
    return myNormalize(dstImg);
}
Mat doBPF(Mat complexImg,int r,int radius){
    Mat BandPassFilter = doLPF1(complexImg,radius) - doLPF1(complexImg,r);
    imshow("BandPassFilter", BandPassFilter);
    return BandPassFilter;
}
Mat padding1(Mat img){
    int dftrows = getOptimalDFTSize(img.rows);
    int dftcols = getOptimalDFTSize(img.cols);
    Mat padded;
    copyMakeBorder(img, padded, 0, dftrows-img.rows, 0, dftcols-img.cols, BORDER_CONSTANT,Scalar::all(0));
    return padded;
}
