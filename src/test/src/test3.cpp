#include <stdio.h>
#include <iostream>
#include <math.h>
#include <time.h>
#include "opencv2/core.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/ml.hpp"
#include <ctime>
#include "reco.h"


using namespace std;
using namespace cv;

vector<float> x, y, radius;
float StandardDeviation(vector<float> population, float N);
vector<float> ContourFeatures(Mat image);
vector<float> GradientOrientation(Mat image);
vector<float> FindFeatures(Mat img);
float EuclideanDistance(vector<float> vec1, vector<float> vec2);
float CosineSimilarity(vector<float> vec1, vector<float> vec2);
float FindMaxima(int scale_levels, Mat scale_Norm[], double sigmas[]);
bool CheckMaxima(Mat scale_Norm[], int scale_levels, int value, int i, int j);
bool CheckWider(Mat img, int value, int i, int j);
vector<float> BlobDetection(Mat src_image);
float findLines(Mat img);

int main()
{

	Mat find_image ;
	Mat compare_image;
	int comp_len = 7;

	find_image= imread("/media/prakhar/Linux FIles/Flam Apps/Testing and experiment/DataSets/Colorful Patterns/Image.orig47.jpg");
	compare_image= imread("/media/prakhar/Linux FIles/Flam Apps/Testing and experiment/DataSets/Colorful Patterns/Image.orig34.jpg");


	vector<float> comp = FindFeatures(compare_image);
	vector<float> comp2 ;
	comp2.assign(comp.begin(), comp.begin()+comp_len);

	vector<float> fin = FindFeatures(find_image);
	vector<float> fin2;
	dbg_d(fin.size());

	fin2.assign(fin.begin(), fin.begin()+comp_len);

	cout<<"Euclidean Distance "<< EuclideanDistance(fin2,comp2)<<endl;
	clock_t start =clock();
	cout<<"Cosine Similarity " << CosineSimilarity(fin2,comp2)<<endl;
    clock_t stop =clock();
    dbg_d((stop-start));


	dbg_messsage("VIDEO_COMPARE: starting camera ....");
    vision::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "ERROR - Failed to open VideoCapture for device " + std::to_string(0) + "\n" << std::flush;
        return 0;
    }

	while (true)
	{

		vision::Mat frame;
    if(!cap.read(frame))
    {
      continue;
    }

	comp = FindFeatures(frame);
	comp2.assign(comp.begin(), comp.begin()+comp_len);

	cout<<"Euclidean Distance "<< EuclideanDistance(fin2,comp2)<<endl;
	cout<<"Cosine Similarity " << CosineSimilarity(fin2,comp2)<<endl;

	 vision::imshow("video",frame);

    if (vision::waitKey(5) == 27)
  	{
   	cout << "Esc key is pressed by user. Stoppig the video" << endl;
   	break;
  	}

	}




    return 0;
}



void videoComp()
{



}

vector<float> GradientOrientation(Mat image){
	int ddepth = CV_8UC1;
	vector<float> orientations, magnitudes, grad_feat;
	Mat gray;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, gray, Size(7,7), 7/3);

	Mat grad_x, grad_y;
	Sobel(gray, grad_x, ddepth, 1, 0, 3, 1, 0, BORDER_DEFAULT);
	Sobel(gray, grad_y, ddepth, 0, 1, 3, 1, 0, BORDER_DEFAULT);

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

	return grad_feat;
}





vector<float> ContourFeatures(Mat image){
	vector<float> contours_feat;
	Mat canny_output;
	vector<vector<Point>> contours;
	Mat gray, blur;
	cvtColor(image, gray, COLOR_BGR2GRAY);
	GaussianBlur(gray, blur, Size(7,7), 7/3);

	// Detect edges using canny
	Canny(blur, canny_output, 180, 220, 3);
	// Find contours
	findContours(canny_output, contours, RETR_TREE, CHAIN_APPROX_SIMPLE);
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

	contours_feat.push_back(average);
	contours_feat.push_back(contours.size());

	return contours_feat;
}


/*Return the feature vector of an image*/
vector<float> FindFeatures(Mat img){
	vector<float> features;

//	Blob features (x, y, radius, numberOf)
	vector<float> blob_f = BlobDetection(img);
	for (int i=0; i<blob_f.size(); i++){
		features.push_back(blob_f.at(i));
	}

	/*Surf features (x, y, angle, numberOf)*/
	// vector<float> sift_f = SiftFeatures(img);
	// for (int i=0; i<sift_f.size(); i++){
	// 	features.push_back(sift_f.at(i));
	// }

	/*Color features (red, green, blue)*/
	// vector<float> color_feat = AverageColours(img);
	// for (int i=0; i<color_feat.size(); i++){
	// 	features.push_back(color_feat.at(i));
	// }

	/*Average Intensity feature*/
	// float I_average = AverageIntensity(img);
	// features.push_back(I_average);

	/*Color histogram features (red, green, blue)*/
//	vector<float> hist_feat = colorHistograms(img);
//	for (int i=0; i<hist_feat.size(); i++){
//		features.push_back(hist_feat.at(i));
//	}

	/*Hough transform number of lines features (num of lines)*/
	float hough = findLines(img);
	features.push_back(hough);

	/*Gradient orientation features (orientation, magnitude)*/
	vector<float> grad_f = GradientOrientation(img);
	for (int i=0; i<grad_f.size(); i++){
		features.push_back(grad_f.at(i));
	}

	/*Find Contour features (average size, number of)*/
	vector<float> contour_f = ContourFeatures(img);
	for (int i=0; i<contour_f.size(); i++){
		features.push_back(contour_f.at(i));
	}

	return features;
}


float EuclideanDistance(vector<float> vec1, vector<float> vec2){
	float euclid=0, average=0;
	vector<float> diff;
	for (int i=0; i<vec1.size(); i++){
		float value1 = vec1.at(i);
		float value2 = vec2.at(i);

		euclid = euclid + pow((value1 - value2), 2);

		average = average + abs(value1-value2);
		diff.push_back(abs(value1-value2));
	}

	float std = StandardDeviation(diff, diff.size());

	//return sqrt(euclid);
	return average/vec1.size();
	//return std;
}


float CosineSimilarity(vector<float> vec1, vector<float> vec2){
	double cosine=0, sumAB=0, sumA=0, sumB=0;

	for (int i=0; i<vec1.size(); i++){
		float value1 = vec1.at(i);
		float value2 = vec2.at(i);

		sumAB = sumAB + (value1*value2);
		sumA = sumA + pow(value1, 2);
		sumB = sumB + pow(value2, 2);
	}

	float par1 = sqrt(sumA);
	float par2 = sqrt(sumB);

	cosine = sumAB/(par1*par2);

	return cosine;
}

float StandardDeviation(vector<float> population, float N){
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




vector<float> BlobDetection(Mat src_image){
	vector<float> blobs;
	const int scale_levels = 7;
	Mat gray_image;
	Mat	scale_Gaussian[scale_levels];
	Mat scale_Laplacian[scale_levels];
	Mat scale_Norm[scale_levels];
	double sigmas[scale_levels];
	int kernel_size = 3, scale = 1, delta = 0, ddepth = CV_8UC1;

	/*Convert to grayscale*/
	cvtColor(src_image, gray_image, COLOR_RGB2GRAY);

	/*Create scale space*/
	for (int i=0; i<scale_levels; i++){

		double sigma_gaussian = (i+1)*2;
		//double sigma_gaussian = (i+1)*2;
		int kernel_gaussian = ((i+1) * 4) + 1;
		//int kernel_gaussian = sigma_gaussian*3;
		//if (kernel_gaussian%2==0) kernel_gaussian++;

		GaussianBlur(gray_image, scale_Gaussian[i], Size(kernel_gaussian,kernel_gaussian), sigma_gaussian);

		Laplacian(scale_Gaussian[i], scale_Laplacian[i], ddepth, kernel_size, scale, delta);

		scale_Norm[i] = scale_Laplacian[i] * sigma_gaussian^2;
		//scale_Norm[i] = scale_Norm[i] * 2;
		sigmas[i] = sigma_gaussian;

	}

	/*Iterate through all scale spaces and find blob positions*/
	float maxima_number = FindMaxima(scale_levels, scale_Norm, sigmas);

	/*Calculate standard deviation of coordinates of blobs and radius for features*/
	// float stdX = StandardDeviation(x, maxima_number);
	// float stdY = StandardDeviation(y, maxima_number);
	float stdR = StandardDeviation(radius, maxima_number);

	// blobs.push_back(stdX);
	// blobs.push_back(stdY);
	blobs.push_back(stdR);
	blobs.push_back(maxima_number);

	//cout << "Counter:" << maxima_number << "  SumX:" << sum_x << endl;
	//cout << "X:" << stdX << "  Y:" << stdY << "  R:" << stdR << "  Number of:" << maxima_number << endl;

	x.clear();
	y.clear();
	radius.clear();

	return blobs;
}

float FindMaxima(int scale_levels, Mat scale_Norm[], double sigmas[]){
	float maxima_counter = 0;
	int threshold = 125;
	for (int s=1; s<scale_levels-1; s++){
		/*Get space scale and sigma*/
		cv::Mat img = scale_Norm[s];
		double sigma = sigmas[s];

		/*Iterate through pixels*/
		for (int i=2; i<img.rows-2; i++){
			for (int j=2; j<img.cols-2; j++){

				int value = img.at<uchar>(i,j);

				/*If the intensity value is lower than the theshold, then the pixel is not a maxima*/
				if (value < threshold) continue;

				/*Check if a pixel is a maximum in all space scales*/
				if (CheckMaxima(scale_Norm, scale_levels, value, i, j)) continue;

				/*Check if a pixel is the maximum from its neighbors in its space scale*/
				if ( img.at<uchar>(i-1,j) >= value || img.at<uchar>(i+1,j) >= value || img.at<uchar>(i,j-1) >= value ||
					img.at<uchar>(i,j+1) >= value || img.at<uchar>(i-1,j-1) >= value || img.at<uchar>(i+1,j-1) >= value ||
					img.at<uchar>(i-1,j+1) >= value || img.at<uchar>(i+1,j+1) >= value ){
					continue;
				}

				/*Check wider in the neighborhood*/
				if (CheckWider(img, value, i, j)) continue;

				/*Check if a pixel is the maximum from its neighbors in the previous space scale*/
				cv::Mat scalePrevious = scale_Norm[s-1];
				if ( scalePrevious.at<uchar>(i-1,j) >= value || scalePrevious.at<uchar>(i+1,j) >= value ||
					scalePrevious.at<uchar>(i,j-1) >= value || scalePrevious.at<uchar>(i,j+1) >= value ||
					scalePrevious.at<uchar>(i-1,j-1) >= value || scalePrevious.at<uchar>(i+1,j-1) >= value ||
					scalePrevious.at<uchar>(i-1,j+1) >= value || scalePrevious.at<uchar>(i+1,j+1) >= value ||
					scalePrevious.at<uchar>(i,j) >= value){
					continue;
				}

				/*Check wider in the neighborhood*/
				if (CheckWider(scalePrevious, value, i, j)) continue;

				/*Check if a pixel is the maximum from its neighbors in the next space scale*/
				cv::Mat scaleNext = scale_Norm[s+1];
				if ( scaleNext.at<uchar>(i-1,j) >= value || scaleNext.at<uchar>(i+1,j) >= value ||
					scaleNext.at<uchar>(i,j-1) >= value || scaleNext.at<uchar>(i,j+1) >= value ||
					scaleNext.at<uchar>(i-1,j-1) >= value || scaleNext.at<uchar>(i+1,j-1) >= value ||
					scaleNext.at<uchar>(i-1,j+1) >= value || scaleNext.at<uchar>(i+1,j+1) >= value ||
					scaleNext.at<uchar>(i,j) >= value){
					continue;
				}

				/*Check wider in the neighborhood*/
				if (CheckWider(scaleNext, value, i, j)) continue;

				/*Calculate radius of blob and store the coordinates and the radius*/
				double rad = sigma*1.414;
				//cout << "Found Maxima at:(" << i << "," << j << ") Sigma:" << sigma << " Radius:" << rad << " Intensity:" << value << endl;
				// x.push_back(i);
				// y.push_back(j);
				radius.push_back(rad);
				maxima_counter++;
			}
		}
	}
	return maxima_counter;
}


bool CheckMaxima(Mat scale_Norm[], int scale_levels, int value, int i, int j){
	bool notMaxima = false;
	for (int scale=0; scale<scale_levels; scale++){
		Mat sc = scale_Norm[scale];
		/*If the pixel has lower intensity then it is not a maximum*/
		if (value < sc.at<uchar>(i,j)){
			notMaxima = true;
			break;
		}
	}

	return notMaxima;
}



bool CheckWider(Mat img, int value, int i, int j){
	if ( img.at<uchar>(i-2,j-2) >= value || img.at<uchar>(i-2,j-1) >= value || img.at<uchar>(i-2,j) >= value ||
		img.at<uchar>(i-2,j+1) >= value || img.at<uchar>(i-2,j+2) >= value || img.at<uchar>(i-1,j+2) >= value ||
		img.at<uchar>(i,j+2) >= value || img.at<uchar>(i+1,j+2) >= value || img.at<uchar>(i+2,j+2) >= value ||
		img.at<uchar>(i+2,j+1) >= value || img.at<uchar>(i+2,j) >= value || img.at<uchar>(i+2,j-1) >= value ||
		img.at<uchar>(i+2,j-2) >= value || img.at<uchar>(i+1,j-2) >= value || img.at<uchar>(i,j-2) >= value ||
		img.at<uchar>(i-1,j-2) >= value ){
			return true;
		}

	return false;
}


float findLines(Mat img)
{

  Mat dst, gblur;
  GaussianBlur(img, gblur, Size(7,7), 7/3);
  Canny(gblur, dst, 60, 60, 3);
  //cvtColor(dst, cdst, CV_GRAY2BGR);

  vector<Vec2f> lines;
  HoughLines(dst, lines, 1, CV_PI/180, 100, 0, 0 );

  //vector<float> numOfLines = lines.size();
  //cout << (float)lines.size() << endl;

  return (float)lines.size();
}
