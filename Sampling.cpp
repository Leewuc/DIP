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

Mat mySampling(Mat srcImg){
    int width = srcImg.cols/2;
    int height = srcImg.rows/2;
    Mat dstImg(height, width, CV_8UC3);
    //가로 세로가 입력 영상의 절반인 영상을 먼저 생성
    uchar* srcData = srcImg.data;
    uchar* dstData = dstImg.data;
    for(int y = 0 ;y<height;y++){
        for(int x = 0; x<width;x++){
            int index = y * width*3 + x*3;
            int index2 = (y*2) * (width*2) *3 + (x*2)*3;
            dstData[index+0] = srcData[index2+0];
            dstData[index+1] = srcData[index2+1];
            dstData[index+2] = srcData[index2+2];
            //2배 간격으로 인덱싱 해 큰 영상을 작은 영상에 대입할 수 있음
        }
    }
    return dstImg;
}
