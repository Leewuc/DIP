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

vector<Mat> myGaussianPyramid(Mat srcImg){
    vector<Mat> Vec; //여러 영상을 모아서 저장하기 위해 STL의 vector컨테이너 사용
    Vec.push_back(srcImg);
    for(int i=0;i<4;i++){
//#if USE_OPENCV
//        pyrDown(srcImg,srcImg,Size(srcImg.cols/2,srcImg.rows/2));
        //Down sampling과 GaussianFilter가 포함된 OPENCV 함수
        //영상의 크기가 가로,세로 절반으로 줄어들도록 출력사이즈 지정
//#else
        srcImg = myColorGaussianFilter(srcImg);//앞서 구현한 Gaussian filtering
        srcImg = mySampling(srcImg); //앞서 구현한 down sampling
//#endif
        Vec.push_back(srcImg); //vector 컨테이너에 하나씩 처리결과를 삽입
    }
    return Vec;
}
