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



//***************************************Type Defines*****************************//

#define _vi vector<vision::Mat>
typedef vector<vision::cuda::GpuMat> _vgpuM;
typedef vector<vector<vision::KeyPoint>> _vkps;
//---------------------------------END Type Defines---------------------------//


//********************************Debug Libraries*****************************//
#ifdef DEBUG_LOGIC
#include"debug.h"
#define DEBUG_MODE true
#endif // DEBUG_LOGIC
//--------------------------------------END Debug Libraries-----------------------------------//


#endif // COMMON_H_INCLUDED
