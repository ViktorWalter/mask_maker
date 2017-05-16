#include "../include/mask_maker/MaskMaker.h"
#include <ostream>
#include "ros/package.h"

#define minDelay 1.0
#define exposureTime 4.0
#define thresh 50

MaskMaker::MaskMaker()
{
  initialized = true;
  first = true;
  done = false;

}

int MaskMaker::processImage(
    cv::Mat imCurr_t,
    cv::Mat imView_t,
    bool gui,
    bool debug)
{
  if (done)
    return 0;
  
  if (!initialized)
  {
    std::cout << "Structure was not initialized; Returning.";
    return 1;
  }

  if (first) {
    startTime = clock();
    imCurr_t.convertTo(imPrev_l,CV_16SC3);
    imAccum_l = cv::Mat(imCurr_t.size(),CV_16SC1);
  }


  cv::Mat imDiff;



  imCurr_t.convertTo(imCurr_l,CV_16SC3);
//  cv::cvtColor(imCurr_g, imCurr_bw_g, CV_RGB2GRAY);
//
//
  clock_t currTime = clock();
  float dt = double(currTime - startTime) / CLOCKS_PER_SEC;
  if (dt < minDelay){
    /*   ROS_INFO("Not yet. dt = %f",dt); */
    imCurr_l.copyTo(imPrev_l);
    first = false;
    return 1;
  }
  else if (dt > minDelay+exposureTime){
    makeMask();
    cv::imshow("mask",imMask_l);
    done = true;
    return 2;
  }


  imDiff = imCurr_l-imPrev_l;
  /* imDiff.convertTo(imDiff,CV_32FC3); */
  pow(imDiff,2,imDiff);
  cv::transform(imDiff, imDiff, cv::Matx13f(1,1,1));
  cv::Mat brighterCompare = imDiff - imAccum_l ;
  cv::threshold(brighterCompare,brighterCompare,0,1,CV_THRESH_TOZERO);
  imAccum_l = imAccum_l + brighterCompare;
  cv::Mat imAccum_v, imDiff_v, imCurr_v;
  imAccum_l.convertTo(imAccum_v, CV_8UC3,0.01);

  cv::imshow("accum",imAccum_v);

  
  
  
  if (!first){
  }
  if (debug)
  {
    // ROS_INFO("out: %dx%d",outX_l.cols,outX_l.rows);
  }
  imCurr_l.copyTo(imPrev_l);

    first = false;
    return 0;
}


void MaskMaker::makeMask(){
  imAccum_l.convertTo(imAccum_l,CV_8UC1,0.01);
  cv::threshold(imAccum_l,imMask_l,thresh,255,CV_THRESH_BINARY);
  cv::Mat element = cv::getStructuringElement(cv::MORPH_ELLIPSE,cv::Size(35,35));
  /* cv::morphologyEx(imMask_l, imMask_l, cv::MORPH_DILATE, element); */
  cv::morphologyEx(imMask_l, imMask_l, cv::MORPH_OPEN, element);
  cv::bitwise_not(imMask_l, imMask_l);
  ROS_INFO("type is %d",imMask_l.type());
}


