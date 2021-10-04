#include "cluster_finder.h"

float _image_contours_average(vision::Mat image)
{

	vision::Mat canny_output;
	vector<vector<vision::Point>> contours;
	vision::Mat gray, blur;
	//vision::cvtColor(image, gray, CV_BGR2GRAY);
	vision::GaussianBlur(image, blur, vision::Size(7,7), 7/3);

	// Detect edges using canny
	vision::Canny(blur, canny_output, 180, 220, 3);
	// Find contours
	vision::findContours(canny_output, contours, vision::RETR_TREE, vision::CHAIN_APPROX_SIMPLE);
	//imshow("Canny", canny_output);

	/*Find average size of the contours in the image*/
	float average=0;
	if (contours.size()!=0){
		float sum=0;
		for (int i=0; i<contours.size(); i++){
			sum = sum + contours.at(i).size();
		}

		average = sum/contours.size();
	}

    return average;
}
