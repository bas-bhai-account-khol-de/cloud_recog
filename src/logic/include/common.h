#ifndef COMMON_H_INCLUDED
#define COMMON_H_INCLUDED

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

//**********************Namespace*****************************************//
using namespace std;
namespace vision = cv;
//-----------------------END Namespace------------------------------------------//

//************************Structs*************************//

struct _vision_images
{
    vision::Mat image;
    string location;
};

//-----------------------END Structs----------------------//


//***************************************Type Defines*****************************//

typedef vector<vision::Mat> _vi ;
typedef vector<_vision_images> _visi;
typedef vector<vision::cuda::GpuMat> _vgpuM;
typedef vector<vector<vision::KeyPoint>> _vkps;
//---------------------------------END Type Defines---------------------------//


//********************************Debug Libraries*****************************//
#ifdef DEBUG_LOGIC
#include"debug.h"
#define DEBUG_MODE true
#endif // DEBUG_LOGIC
//--------------------------------------END Debug Libraries-----------------------------------//


//****************Common Functions***************************//
/*Calculate standard deviation for a population of numbers*/
float StandardDeviation(vector<float> population, float N);
//------------------END Common Functions-------------------------------//


#endif // COMMON_H_INCLUDED
