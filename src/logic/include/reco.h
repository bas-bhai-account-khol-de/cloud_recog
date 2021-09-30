#ifndef RECO_H_INCLUDED
#define RECO_H_INCLUDED
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <cstring>
#include<ctime>

//***************************Vision Libraries********************************//
#include <opencv2/opencv.hpp>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/cudaarithm.hpp>
#include <opencv2/cudaimgproc.hpp>
#include <opencv2/cudafeatures2d.hpp>
#include <opencv2/video.hpp>
#include <opencv2/videoio.hpp>
//--------------------------------END vision Libraries---------------------------//
using namespace std;
namespace vision = cv;
#define _vi vector<vision::Mat>

#ifdef DEBUG_LOGIC
#include"debug.h"
#define DEBUG_MODE true
#endif // DEBUG_LOGIC

typedef vector<vision::cuda::GpuMat> _vgpuM;
typedef vector<vector<vision::KeyPoint>> _vkps;



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
void import_from_directory(_vi &images,string folder_name = "");

/**
 * @brief extracts features of every image in mat vector
 * @param parameter-images Mat vector pointer \n <br>
 * @return returns pointer of vector of image keypoint and descrioptors
 *
 ********************/
void compute_feature(_vi &images,_vkps &kp_dataset,_vgpuM &desc_dataset, unsigned int max_features = 500 );

/**
 * This function starts camera and starts matching image from given
 * dataset kp and des
 * @brief This fuctions intializes camera matching
 * @param
 *
 *
 */
void compare_video(_vkps &keypoints,_vgpuM &descriptors,_vi &images);

/**
 * @brief this function compares images with database
 * @param image to bec compared
 * @param keypoints cpu keypoints
 * @param desc gpu keypoints
 *
 */
void compare_image(string query_image_path , _vkps &keypoints,_vgpuM &descriptors,_vi &images);

#endif // RECO_H_INCLUDED
