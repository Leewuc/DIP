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

void doCannyEx(){
    cout << "--- doCannyEx() --- \n" << endl;
        Mat src_img = imread("edge_test.jpg", 0);

        if (!src_img.data) cout << "No image data\n";

        // Case 1
        Mat dst_img;
    std::chrono::system_clock::time_point start = std::chrono::system_clock::now();
        Canny(src_img, dst_img, 180, 240);
    std::chrono::system_clock::time_point end = std::chrono::system_clock::now();
    cout << "180, 240: " << duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;

        Mat result_img;
        hconcat(src_img, dst_img, result_img);
        imshow("doCannyEx()", result_img);
        waitKey(0);

        // Case 2
    start = std::chrono::system_clock::now();
        Canny(src_img, dst_img, 180, 480);
    end = std::chrono::system_clock::now();
    cout << "180, 480: " << duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;

        hconcat(src_img, dst_img, result_img);
        imshow("doCannyEx()", result_img);
        waitKey(0);

        // Case 3
    start = std::chrono::system_clock::now();
        Canny(src_img, dst_img, 90, 120);
    end = std::chrono::system_clock::now();
    cout << "90, 120: " << duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;

        hconcat(src_img, dst_img, result_img);
        imshow("doCannyEx()", result_img);
        waitKey(0);

        // Case 4
    start = std::chrono::system_clock::now();
        Canny(src_img, dst_img, 90, 240);
    end = std::chrono::system_clock::now();
    cout << "90, 240: " << duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;

        hconcat(src_img, dst_img, result_img);
        imshow("doCannyEx()", result_img);
        waitKey(0);

        // Case 5
    start = std::chrono::system_clock::now();
        Canny(src_img, dst_img, 360, 480);
    end = std::chrono::system_clock::now();
    cout << "360, 480: " << duration_cast<std::chrono::milliseconds>(end - start).count() << " ms" << endl;

        hconcat(src_img, dst_img, result_img);
        imshow("doCannyEx()", result_img);
        waitKey(0);

        hconcat(src_img, dst_img, result_img);
        imshow("doCannyEx()", result_img);
        waitKey(0);
    }
