#ifndef CLUSTER_FINDER_H
#define CLUSTER_FINDER_H

//****************************Self Define Headers****************************//
#include "common.h"
//-------------------------END Self Define Headers--------------------------//


/**
 * @brief this function finds the average of all the contours found
 * @param parameter-image image whose contours are to be found
 * @return returns a float value avarage
 **************************************/
float _image_contours_average(vision::Mat image);

/**
 * @brief this function finds the average of all the gradient magnitudes found
 * @param parameter-image image whose image grdient are to be found 
 * @return returns a float value standard deviation
 **************************************/
float _image_gradient_magnitude(vision::Mat image);



/**
 * @brief this function diplace average countours of current frame
 * @param None
 * @return returns void
 **************************************/
void _video_countours_average();


/**
 * @brief this function diplace average gradiant of current frame
 * @param None
 * @return returns void
 **************************************/
void _video_gradient_magnitude();


#endif // CLUSTER_FINDER_H
