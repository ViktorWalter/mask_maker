#ifndef SPARSEOPTFLOW_OCL_H
#define SPARSEOPTFLOW_OCL_H

#include <cv_bridge/cv_bridge.h>
#include <opencv2/highgui/highgui.hpp>
#include <image_transport/image_transport.h>
#include <opencv2/core/core.hpp>
#include <opencv2/core/ocl.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <CL/cl.h>
#include <CL/cl_ext.h>


class MaskMaker
{
public:
    MaskMaker();

    int processImage(
        cv::Mat imCurr_t,
        cv::Mat imView_t,
        bool gui=true,
        bool debug=true);

    bool initialized;

private:
    bool first;
    bool done;
    clock_t startTime;

    cv::Mat imPrev_l;
    cv::Mat imCurr_l;
    cv::Mat imAccum_l;
    cv::Mat imMask_l;

    void makeMask();

};


#endif // SPARSEOPTFLOW_OCL_H
