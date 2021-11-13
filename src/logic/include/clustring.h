#ifndef CLUSTRING_H_INCLUDED
#define CLUSTRING_H_INCLUDED
#include "reco.h"
#include <ctime>
#include <map>

struct _scalarized_image_coordinates
{
    _image_description *image;
    vector<float> scalarized_discriptor;
};

namespace ml_pack_clustering{
/**
 * @brief this functions clusters your image
 * @param images images that are to be clustered togather
 * @param k number of cluster
 * @param features the number of freature
 * ********************************************/
void cluster(vector<_image_description> &image_descriptions,vector<vector<float>> &centroids ,vector<vector<_scalarized_image_coordinates>>&clusters ,int k=2,int features = 500);

/**
 * @brief this function converts the descriptos to scalar values
 * @param descriptons this is the array of images that are to be converted
 * @param scalar_coordinates the scalarized descriptos are filed in this array
 *
 * ***************************/

void scalarize_images(vector<_image_description> &descriptions,vector<_scalarized_image_coordinates> &scalar_coordinates,int features=500);

/**
 * @brief this function converts the descriptos to scalar values
 * @param descriptons this is the array of images that are to be converted
 * @param scalar_coordinates the scalarized descriptos are filed in this array
 *
 * ***************************/

void scalarize_image(_image_description&descriptions,_image_description &reference,_scalarized_image_coordinates &scalar_coordinates,int features=500);

/**
 * @brief this function finds distance between 2 points
 *);
 *
 * ****/
float distance(vector<float> a,vector<float> b );


vector<float> find_centroid(vector<_scalarized_image_coordinates> &cluster);


/**
 * This function starts camera and starts matching image from given
 * dataset kp and des
 * @brief This fuctions intializes camera matching
 * @param  parameter-image_dataset pointer of vctor of vector of image_descriptor
 *
 *
 */
void compare_video_in_clusters(_image_description &refence,vector<vector<_scalarized_image_coordinates>>&clusters,vector<vector<float>> &centroids ,  int main_cluster =4,int features=500);


void predictCluster(map<float,int> &cluster_order,_scalarized_image_coordinates &scalarized_image, vector<vector<float>> &centroids );


}





#endif
