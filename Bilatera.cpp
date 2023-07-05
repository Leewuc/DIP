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

double gaussian(float x,double sigma){
    return exp(-(pow(x,2))/(2*pow(sigma,2))) / (2 * CV_PI * pow(sigma,2));
}
float distance(int x, int y, int i, int j){
    return float(sqrt(pow(x-i,2)+pow(y-j,2)));
}
void myKernelConv(const Mat& src_img,Mat& dst_img,const Mat& kn){
    dst_img = Mat::zeros(src_img.size(),CV_8UC1);
    int wd = src_img.cols;
    int hg = src_img.rows;
    int kwd = kn.cols;
    int khg = kn.rows;
    int rad_w = kwd / 2;
    int rad_h = khg / 2;
    float* kn_data = (float*)kn.data;
    uchar* src_data = (uchar*)src_img.data;
    uchar* dst_data = (uchar*)dst_img.data;
    float wei,tmp,sum;
    // <픽셀 인덱싱 (가장자리 제외)>
    for(int c=rad_w+1;c<wd-rad_w;c++){
        for(int r=rad_h+1;r<hg-rad_h;r++){
            tmp = 0.f;
            sum = 0.f;
            // <커널 인덱싱>
            for(int kc=-rad_w;kc<=rad_w;kc++){
                for(int kr = -rad_h;kr<=rad_h;kr++){
                    wei = (float)kn_data[(kr+rad_h)*kwd+(kc+rad_w)];
                    tmp += wei*(float)src_data[(r+kr)*wd+(c+kc)];
                    sum += wei;
                }
            }
            if(sum != 0.f) tmp = abs(tmp) / sum; //정규화 및 overflow 방지
            else tmp = abs(tmp);
            if(tmp > 255.f) tmp = 255.f; //overflow 방지
            dst_data[r*wd+c] = (uchar)tmp;
        }
    }
}
void myGaussian(const Mat& src_img,Mat& dst_img,Size size){
    // <커널 생성>
    Mat kn = Mat::zeros(size,CV_32FC1);
    double sigma = 1.0;
    float *kn_data = (float*)kn.data;
    for(int c=0;c<kn.cols;c++){
        for(int r=0;r<kn.rows;r++){
            kn_data[r*kn.cols+c] = (float)gaussian2D((float)(c-kn.cols/2),(float)(r-kn.rows/2),sigma);
        }
    }
    // <커널 컨볼루젼 수행>
    myKernelConv(src_img,dst_img,kn);
}
void bilateral(const Mat&src_img, Mat& dst_img,int c, int r, int diameter, double sig_r, double sig_s){
    int radius = diameter /2 ;
    double gr,gs,wei = 0.0;
    double tmp = 0;
    double sum = 0;
    // kernel indexing
    for(int kc= -radius;kc <= radius;kc++){
        for(int kr = -radius; kr <= radius; kr++){
            // range calc
            gr = gaussian((float)src_img.at<uchar>(c+kc,r+kr)-(float)src_img.at<uchar>(c,r),sig_r);
            //spatial calc
            gs = gaussian(distance(c,r,c+kr,r+kr),sig_s);
            wei += gr * gs;
            tmp += src_img.at<uchar>(c+kr,r+kr) * wei;
            sum += wei;
        }
    }
    dst_img.at<double>(c,r) = tmp / sum; //정규화
}

void myBilateral(const Mat&src_img,Mat& dst_img,int diameter,double sig_r,double sig_s){
    dst_img = Mat::zeros(src_img.size(),CV_8UC1);
    Mat guide_img = Mat::zeros(src_img.size(),CV_64F);
    int wh = src_img.cols;
    int hg = src_img.rows;
    int radius = diameter / 2;
    // <픽셀 인덱싱(가장자리제외)>
    for(int c = radius+1;c<hg-radius;c++){
        for(int r=radius+1;r<wh-radius;r++){
            bilateral(src_img,guide_img,c,r,diameter,sig_r,sig_s);
            //화소별 Bilateral 계산 수행
        }
    }
    guide_img.convertTo(dst_img,CV_8UC1); //Mat type 변환
}
void doBilateralEx(){
    cout << "--- doBilateralEx() --- \n" << endl;
    // <input>
    Mat src_img = imread("rock.png",0);
    Mat dst_img;
    if(!src_img.data) printf("No image data \n");
    // <Bilateral 필터링 수행>
    myBilateral(src_img, dst_img, 5, 25.0, 50.0);
    // <output>
    Mat result_img;
    hconcat(src_img,dst_img,result_img);
    imshow("doBilateralEx()",result_img);
    waitKey(0);
}
