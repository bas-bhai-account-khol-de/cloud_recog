#ifndef RECO_H_INCLUDED
#define RECO_H_INCLUDED
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <cstring>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
using namespace std;
namespace vision = cv;
#define _vi vector<vision::Mat>

#ifdef DEBUG_LOGIC
#include"debug.h"
#define DEBUG_MODE true
#endif // DEBUG_LOGIC




/**
*   This initilize ORB and FLann Matcher
*
*   @param none \n  <br>
*   @brief This initilize necessary requirments
*   @return void
*
*
*************************/
void init_reco(int v=0) ;

/**
*   This function imports all the given files of a folder
*   @brief Import Images
*   @param folder_name = PATH TO FOLDER TO BE IMPORTED
*   @return MAT Vector  vector<cv::MAT>()
**********/
_vi *import_from_directory(string folder_name = "");

/**
 * @brief extracts features of every image in mat vector
 * @param parameter-images Mat vector pointer \n <br>
 * @return returns pointer of vector of image keypoint and descrioptors
 *
 ********************/
void compute_feature(_vi *images, unsigned int max_features = 500 );


#endif // RECO_H_INCLUDED
