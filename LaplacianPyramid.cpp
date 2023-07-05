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

vector<Mat> myLaplacianPyramid(Mat srcImg){
    vector<Mat> Vec;
    for(int i =0;i<4;i++){
        if(i != 3){
            Mat highImg = srcImg; //수행하기 이전 영상을 백업
//#if USE_OPENCV
//            pyrDown(srcImg,srcImg,Size(srcImg.cols/2,srcImg.rows/2));
//#else
            srcImg = mySampling(srcImg);
            srcImg = myColorGaussianFilter(srcImg);
//#endif
            Mat lowImg = srcImg;
            resize(lowImg,lowImg,highImg.size());
            //작아진 영상을 백업한 영상의 크기로 확대
            Vec.push_back(highImg - lowImg + 128);
            //차 영상을 컨테이너에 삽입
            //128을 더해준 것은 차 영상에서 오버플로우를 방지하기 위함
        }
        else{
            Vec.push_back(srcImg);
        }
    }
    return Vec;
}
