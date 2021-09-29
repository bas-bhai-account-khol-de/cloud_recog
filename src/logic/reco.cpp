#include"reco.h"


void init_reco(int v)
{
    cout<<"Recognization library initilized int v"<<endl;
}


vector<vision::Mat> *import_from_directory(string folder_name)
{
    vector<vision::Mat> data_set_images;

    if(folder_name == "")
    {
        cerr<< "Please enter a valid address......";
        return &data_set_images;
    }
    struct dirent *entry;
    DIR *dir = opendir(folder_name.c_str());

    if (dir == NULL) {
        return &data_set_images;
    }
    while ((entry = readdir(dir)) != NULL) {
    if(!strcmp(".",entry->d_name)|| !strcmp("..",entry->d_name))
    {
        continue;
    }
    cout << entry->d_name << endl;
    vision::Mat image_temp = vision::imread(folder_name+"/"+entry->d_name,cv::IMREAD_GRAYSCALE);
    data_set_images.push_back(image_temp);

    }
    closedir(dir);
    cout<<data_set_images.size()<<endl;
    return &data_set_images;

}




void compute_feature(_vi *images,unsigned int max_features )
{
    dbg_messsage("Copmutation started")
    vision::cuda::Stream current_stream;
    vision::cuda::GpuMat gpuimage;
    vision::cuda::GpuMat gpuimage2;
    dbg_messsage("ran until decleration")
    vision::Mat image = vision::imread("/media/prakhar/Linux FIles/Flam Apps/Testing and experiment/DataSets/school_bag/Image_1 Bag.jpg",vision::IMREAD_GRAYSCALE);
    vision::Mat image2 = vision::imread("/media/prakhar/Linux FIles/Flam Apps/Testing and experiment/DataSets/school_bag/Image_1 Bag.jpg",vision::IMREAD_GRAYSCALE);
    vision::imshow("frame",image);
    vision::waitKey(0);
    vision::Mat temp;
    dbg_messsage("ran until decleration")
    gpuimage.upload(image);
    gpuimage2.upload(image2);
    vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create();
    p_orb_d->setBlurForDescriptor(true);
    vision::cuda::GpuMat kp_d;
    vision::cuda::GpuMat kp_d2;
    vision::cuda::GpuMat desc_d;
    vision::cuda::GpuMat desc_d2;
    p_orb_d->detectAndComputeAsync(gpuimage,cv::cuda::GpuMat(), kp_d, desc_d, false, current_stream);
    p_orb_d->detectAndComputeAsync(gpuimage2,cv::cuda::GpuMat(), kp_d2, desc_d2, false, current_stream);
    dbg_messsage("Feature points detected and computed")
    std::vector<std::vector<cv::DMatch> > knn_matches;
    cv::Ptr<cv::cuda::DescriptorMatcher> matcher = cv::cuda::DescriptorMatcher::createBFMatcher(cv::NORM_HAMMING);
    matcher->knnMatch(desc_d2, desc_d, knn_matches, 2);
    std::vector<cv::DMatch> matches;
    std::vector< std::vector<cv::DMatch> > matchescpu;
    for(std::vector<std::vector<cv::DMatch> >::const_iterator it = knn_matches.begin(); it != knn_matches.end(); ++it)
    {
    if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.6)
    {
        matches.push_back((*it)[0]);
    }
    }

    cv::Mat imgRes;
    dbg_d(matches.size())
    //Display and save the image with matches
    std::vector<cv::KeyPoint> keypoints;
    std::vector<cv::KeyPoint> keypoints2;
    p_orb_d->convert(kp_d,keypoints);
    p_orb_d->convert(kp_d2,keypoints2);
    //matcher->knnMatchConvert(matches,matchescpu);
    cv::drawMatches(image2,keypoints2,image,keypoints,matches,imgRes);
    cv::imshow("imgRes", imgRes);
    cv::imwrite("GPU_ORB-matching.png", imgRes);
    cv::waitKey(0);

}
