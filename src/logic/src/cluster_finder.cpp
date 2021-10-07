#include "cluster_finder.h"


float StandardDeviation(vector<float> population, float N)
{
	float sum=0, av=0, dev_sum=0, std=0;
	if (N!=0){

		for (int i=0; i<N; i++)
			sum = sum + population.at(i);

		av = sum/N;

		for (int i=0; i<N; i++)
			dev_sum = dev_sum + pow(population.at(i)-av, 2);

		std = sqrt(dev_sum/N);
	}

	return std;
}

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


void _video_countours_average()
{
     dbg_messsage("VIDEO_COMPARE: starting camera ....");
    vision::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "ERROR - Failed to open VideoCapture for device " + std::to_string(0) + "\n" << std::flush;
        return;
    }
    vision::cuda::Stream stream;

   while(true)
  {
    // capture frame
    vision::Mat frame;
    if(!cap.read(frame))
    {
      continue;

    }
    vision::cvtColor(frame,frame,vision::COLOR_BGR2GRAY);
    float average  = _image_contours_average(frame);
    dbg_d(average);
    vision::imshow("video",frame);
    if (vision::waitKey(5) == 27)
    {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        break;
    }
}
}



float _image_gradient_magnitude(vision::Mat image)
{
	int ddepth = CV_8UC1;
	vector<float> orientations, magnitudes, grad_feat;
	vision::Mat gray = image;
//	vision::cvtColor(image, gray, vision::COLOR_BGR2GRAY);
	vision::GaussianBlur(gray, gray, vision::Size(7,7), 7/3);

	vision::Mat grad_x, grad_y;
	vision::Sobel(gray, grad_x, ddepth, 1, 0, 3, 1, 0, vision::BORDER_DEFAULT);
	vision::Sobel(gray, grad_y, ddepth, 0, 1, 3, 1, 0, vision::BORDER_DEFAULT);

	float rx=0, ry=0, orient=0, mag=0, sum_orient=0, sum_mag=0;
	for (int i=0; i<grad_x.rows; i++){
		for (int j=0; j<grad_x.cols; j++){
			rx = (float)grad_x.at<uchar>(i,j);
			ry = (float)grad_y.at<uchar>(i,j);

			orient = atan2(ry, rx);
			mag = sqrt(pow(rx,2) + pow(ry,2));
			//cout << "Orient:" << orient << "  Mag:" << mag << endl;
			orientations.push_back(orient);
			magnitudes.push_back(mag);
		}
	}

	float stdOrient = StandardDeviation(orientations, orientations.size());
	float stdMag = StandardDeviation(magnitudes, magnitudes.size());

	grad_feat.push_back(stdOrient);
	grad_feat.push_back(stdMag);

	return stdOrient;

}




void _video_gradient_magnitude()
{
    dbg_messsage("VIDEO_COMPARE: starting camera ....");
    vision::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "ERROR - Failed to open VideoCapture for device " + std::to_string(0) + "\n" << std::flush;
        return;
    }
    vision::cuda::Stream stream;

   while(true)
  {
    // capture frame
    vision::Mat frame;
    if(!cap.read(frame))
    {
      continue;

    }
    vision::cvtColor(frame,frame,vision::COLOR_BGR2GRAY);
    float average  = _image_gradient_magnitude(frame);
    dbg_d(average);
    vision::imshow("video",frame);
    if (vision::waitKey(5) == 27)
    {
        cout << "Esc key is pressed by user. Stoppig the video" << endl;
        break;
    }
}
}
