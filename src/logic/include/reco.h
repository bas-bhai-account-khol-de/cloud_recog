#ifndef RECO_H_INCLUDED
#define RECO_H_INCLUDED
#include <iostream>
#include <dirent.h>
#include <sys/types.h>
#include <cstring>
#include<ctime>

//***************************Vision Libraries********************************//
#include "common.h"
//--------------------------------END vision Libraries---------------------------//

//**************************External Libraries**************************************//
#include "cluster_finder.h"
//----------------------END External Libraries------------------------------//

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
