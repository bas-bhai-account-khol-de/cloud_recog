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


#endif // CLUSTER_FINDER_H
