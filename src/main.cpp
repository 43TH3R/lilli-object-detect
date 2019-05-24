#include <stdio.h>
#include <string.h>
#include <sl/Camera.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace sl;

cv::Mat slMat2cvMat(Mat& input);
string type2str(int type);
float compare(uchar L, uchar a, uchar b, uchar d, uchar L2, uchar a2, uchar b2, uchar d2);

int main(int argc, char **argv) {
    
    Camera zed;
    
    InitParameters init;
    init.sdk_verbose = false;
    init.camera_resolution = RESOLUTION_VGA;
    init.camera_fps = 30;
    init.depth_mode = DEPTH_MODE_MEDIUM;
    init.coordinate_units = UNIT_MILLIMETER;
    
    RuntimeParameters runtime;
    runtime.sensing_mode = SENSING_MODE_FILL;
    
    ERROR_CODE err = zed.open(init);
    if (err != SUCCESS) exit(-1);

    sl::Mat zed_img(zed.getResolution(), MAT_TYPE_8U_C4);
    sl::Mat zed_dep(zed.getResolution(), MAT_TYPE_8U_C4);//MAT_TYPE_32F_C1);
    cv::Mat img = slMat2cvMat(zed_img);
    cv::Mat dep = slMat2cvMat(zed_dep);
    
    //cv::Mat xEdge, yEdge, edge;

    cv::Mat lab;
    cv::Mat comp;
    
    // values pixel for comparison
    uchar L, a, b, d, L2, a2, b2, d2, pr, pc;
    
    vector<int> compMap;
    
    int bright = 1;
    
    char key = ' ';
    while (key != 'q') {
        if (zed.grab(runtime) == SUCCESS) {
            // retrieve image and depth maps
            zed.retrieveImage(zed_img, VIEW_LEFT);
            zed.retrieveImage(zed_dep, VIEW_DEPTH);
            cv::imshow("img", img);
            cv::imshow("dep", dep);
            
            /*cv::Scharr(dep, xEdge, CV_32F, 1, 0, 1, 0, cv::BORDER_DEFAULT);
            cv::convertScaleAbs(xEdge, xEdge);
            cv::Scharr(dep, yEdge, CV_32F, 0, 1, 1, 0, cv::BORDER_DEFAULT);
            cv::convertScaleAbs(yEdge, yEdge);
            cv::addWeighted(xEdge, 0.5, yEdge, 0.5, 0, edge);
            cv::imshow("edge", edge);
            cv::imshow("overlay", img + edge);*/

            // convert to L*a*b for better color comparison
            cv::cvtColor(img, lab, CV_BGR2Lab);
            cv::imshow("lab", lab);
            
            // init component matrix
            int rows = lab.rows;
            int cols = lab.cols;
            comp = cv::Mat::zeros(rows, cols, CV_8UC1);
            
            int newComp = 1;
            
            // our matrices are continuous - 1 long row - only 1 pointer needed
            // row pointers
            uchar *lPtr = lab.ptr<uchar>(0), *dPtr = dep.ptr<uchar>(0), *cPtr = comp.ptr<uchar>(0);
            // channel counts
            int lChan = lab.channels(), dChan = dep.channels(), cChan = comp.channels();
            // iterate
            for (int r = 0; r < rows; r++) {
                for (int c = 0; c < cols; c++) {
                    // get data for this pixel
                    L = lPtr[(r * cols + c) * lChan];
                    a = lPtr[(r * cols + c) * lChan + 1];
                    b = lPtr[(r * cols + c) * lChan + 2];
                    d = dPtr[(r * cols + c) * dChan];
                    if (r > 0) { // compare with neighbor in previous row
                        L2 = lPtr[((r-1) * cols + c) * lChan];
                        a2 = lPtr[((r-1) * cols + c) * lChan + 1];
                        b2 = lPtr[((r-1) * cols + c) * lChan + 2];
                        d2 = dPtr[((r-1) * cols + c) * dChan];
                        pr = compare(L, a, b, d, L2, a2, b2, d2);
                    }
                    if (c > 0) { // compare with neighbor in previous column
                        L2 = lPtr[(r * cols + c-1) * lChan];
                        a2 = lPtr[(r * cols + c-1) * lChan + 1];
                        b2 = lPtr[(r * cols + c-1) * lChan + 2];
                        d2 = dPtr[(r * cols + c-1) * dChan];
                        pc = compare(L, a, b, d, L2, a2, b2, d2);
                    }
                    cPtr[(r * cols + c) * cChan] = (uchar) ((pr < pc? pr : pc) * bright);
                }
            }
            
            imshow("test", comp);
            
            key = cv::waitKey(1);
            if (key == ',') bright--;
            if (key == '.') bright++;
        }
    }
}

float wL = 1;
float wa = 1;
float wb = 1;
float wd = 2;
float w = 10;
float compare(uchar L, uchar a, uchar b, uchar d, uchar L2, uchar a2, uchar b2, uchar d2) {
    // 4D euclidean distance
    return w*sqrt(wL*(L-L2)*(L-L2) + wa*(a-a2)*(a-a2) + wb*(b-b2)*(b-b2) + wd*(d-d2)*(d-d2));
}

/**
* Conversion function between sl::Mat and cv::Mat
**/
cv::Mat slMat2cvMat(Mat& input) {
    // Mapping between MAT_TYPE and CV_TYPE
    int cv_type = -1;
    switch (input.getDataType()) {
        case MAT_TYPE_32F_C1: cv_type = CV_32FC1; break;
        case MAT_TYPE_32F_C2: cv_type = CV_32FC2; break;
        case MAT_TYPE_32F_C3: cv_type = CV_32FC3; break;
        case MAT_TYPE_32F_C4: cv_type = CV_32FC4; break;
        case MAT_TYPE_8U_C1: cv_type = CV_8UC1; break;
        case MAT_TYPE_8U_C2: cv_type = CV_8UC2; break;
        case MAT_TYPE_8U_C3: cv_type = CV_8UC3; break;
        case MAT_TYPE_8U_C4: cv_type = CV_8UC4; break;
        default: break;
    }

    // Since cv::Mat data requires a uchar* pointer, we get the uchar1 pointer from sl::Mat (getPtr<T>())
    // cv::Mat and sl::Mat will share a single memory structure
    return cv::Mat(input.getHeight(), input.getWidth(), cv_type, input.getPtr<sl::uchar1>(MEM_CPU));
}

// openCV mat type debug
string type2str(int type) {
  string r;

  uchar depth = type & CV_MAT_DEPTH_MASK;
  uchar chans = 1 + (type >> CV_CN_SHIFT);

  switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
  }

  r += "C";
  r += (chans+'0');

  return r;
}

