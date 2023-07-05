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

Mat getHistogram(Mat &src){
    Mat histogram;
    const int* channel_numbers = {0};
    float channel_range[] = {0.0,255.0};
    const float* channel_ranges = channel_range;
    int number_bins = 255;
    calcHist(&src,1,channel_numbers,Mat(),histogram,1,&number_bins,&channel_ranges);
    int hist_w = 512;
    int hist_h = 400;
    int bin_w = cvRound((double)hist_w/number_bins);
    Mat histImage(hist_h,hist_w,CV_8UC1,Scalar(0,0,0));
    normalize(histogram,histogram,0,histImage.rows,NORM_MINMAX,-1,Mat());
    for(int i=1;i<number_bins;i++){
        line(histImage,Point(bin_w *(i-1),hist_h - cvRound(histogram.at<float>(i-1))),Point(bin_w *(i),hist_h - cvRound(histogram.at<float>(i))),Scalar(255,0,0),2,8,0);
    }
    return histImage;
}
