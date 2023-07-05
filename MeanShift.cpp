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

class Point5D {
public:
    float x, y, l, u, v;
    
    // Point5D();
    // ~Point5D();
    
    void accumPt(Point5D);
    void copyPt(Point5D);
    float getColorDist(Point5D);
    float getSpatialDist(Point5D);
    void scalePt(float);
    void setPt(float, float, float, float, float);
    void printPt();
};

void Point5D::accumPt(Point5D Pt) {
    x += Pt.x;
    y += Pt.y;
    l += Pt.l;
    u += Pt.u;
    v += Pt.v;
}

void Point5D::copyPt(Point5D Pt) {
    x = Pt.x;
    y = Pt.y;
    l = Pt.l;
    u = Pt.u;
    v = Pt.v;
}

float Point5D::getColorDist(Point5D Pt) {
    return sqrt(pow(l - Pt.l, 2) + pow(u - Pt.u, 2) + pow(v - Pt.v, 2));
}

float Point5D::getSpatialDist(Point5D Pt) {
    return sqrt(pow(x - Pt.x, 2) + pow(y - Pt.y, 2));
}

void Point5D::scalePt(float scale) {
    x *= scale;
    y *= scale;
    l *= scale;
    u *= scale;
    v *= scale;
}

void Point5D::setPt(float px, float py, float pl, float pa, float pb) {
    x = px;
    y = py;
    l = pl;
    u = pa;
    v = pb;
}

void Point5D::printPt() {
    cout << x << " " << y << " " << l << " " << u << " " << v << endl;
}

class MeanShift {
public:
    float bw_spatial = 8;
    float bw_color = 16;
    float min_shift_color = 0.1;
    float min_shift_spatial = 0.1;
    int max_steps = 10;
    vector<Mat> img_split;
    MeanShift(float, float, float, float, int);
    void doFiltering(Mat&);
};

MeanShift::MeanShift(float bs, float bc, float msc, float mss, int ms) {
    bw_spatial = bs;
    bw_color = bc;
    max_steps = ms;
    min_shift_color = msc;
    min_shift_spatial = mss;
}

void MeanShift::doFiltering(Mat& img) {
    int height = img.rows;
    int width = img.cols;
    split(img, img_split);
    
    Point5D pt, pt_prev, pt_cur, pt_sum;
    
    int pad_left, pad_right, pad_top, pad_bottom;
    size_t n_pt, step;
    
    for (int row=0; row < height; row++) {
        for (int col=0; col < width; col++) {
            pad_left = (col - bw_spatial) > 0 ? (col - bw_spatial) : 0;
            pad_right = (col + bw_spatial) < width ? (col + bw_spatial) : width;
            pad_top = (row - bw_spatial) > 0 ? (row - bw_spatial) : 0;
            pad_bottom = (row + bw_spatial) < height ? (row + bw_spatial) : height;
            
            pt_cur.setPt(row, col, (float)img_split[0].at<uchar>(row, col), (float)img_split[1].at<uchar>(row, col), (float)img_split[2].at<uchar>(row, col));
            step = 0;
            do {
                pt_prev.copyPt(pt_cur);
                pt_sum.setPt(0, 0, 0, 0, 0);
                n_pt = 0;
                for (int hx=pad_top; hx < pad_bottom; hx++) {
                    for (int hy=pad_left; hy<pad_right; hy++) {
                        pt.setPt(hx, hy, (float)img_split[0].at<uchar>(hx, hy), (float)img_split[1].at<uchar>(hx, hy), (float)img_split[2].at<uchar>(hx, hy));
                        if (pt.getColorDist(pt_cur) < bw_color) {
                            pt_sum.accumPt(pt);
                            n_pt++;
                        }
                    }
                }
                
                pt_sum.scalePt(1.0 / n_pt);
                pt_cur.copyPt(pt_sum);
                step++;
            }
            while ((pt_cur.getColorDist(pt_prev) > min_shift_color) &&
                   (pt_cur.getSpatialDist(pt_prev) > min_shift_spatial) &&
                   (step < max_steps));
            img.at<Vec3b>(row, col) = Vec3b(pt_cur.l, pt_cur.u, pt_cur.v);
        }
    }
}
void exMyMeanShift() {
    Mat img = imread("fruits.png");
    if (img.empty()) exit(-1);
    cout << " exMyMeanShift() " << endl;
    
    resize(img, img, Size(256, 256), 0, 0, cv::INTER_AREA);
    imshow("src", img);
    imwrite("exMyMeanShift.jpeg", img);
    
    cvtColor(img, img, cv::COLOR_RGB2Luv);
    
    MeanShift MSProc(8, 16, 0.1, 0.1, 10);
    MSProc.doFiltering(img);
    
    cvtColor(img, img, cv::COLOR_Luv2RGB);
    
    imshow("Dst", img);
    waitKey();
    destroyAllWindows();
    imwrite("exMyMeanShift_dst.jpeg", img);
}
