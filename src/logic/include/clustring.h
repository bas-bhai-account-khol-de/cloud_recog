#ifndef CLUSTRING_H_INCLUDED
#define CLUSTRING_H_INCLUDED
#include "reco.h"
#include <ctime>

struct _scalarized_image_coordinates
{
    _vision_images image;
    vector<float> scalarized_discriptor;
};

namespace ml_pack_clustering{
/**
 * @brief this function clusters your image
 * @param images images that are to be clustered together
 * @param k number of cluster
 * @param features the number of features
 * ********************************************/
void cluster(_visi &images,int k=2,int features = 200);

/**
 * @brief this function converts the descriptors to scalar values
 * @param descriptions this is the array of images that are to be converted
 * @param scalar_coordinates the scalarized descriptors are filed in this array
 *
 * ***************************/

void scalarize_images(vector<_image_description> &descriptions,vector<_scalarized_image_coordinates> &scalar_coordinates,int features=200);

/**
 * @brief this function finds distance between 2 points
 * 
 * 
 * ****/
float distance(vector<float> a,vector<float> b );


vector<float> find_centroid(vector<vector<float>> &cluster);



}





#endif
