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
    dbg_messsage("ran until decleration")
    vision::Mat image = vision::imread("/media/prakhar/Linux FIles/Flam Apps/Testing and experiment/DataSets/school_bag/Image_1 Bag.jpg",vision::IMREAD_GRAYSCALE);
    vision::imshow("frame",image);
    vision::waitKey(0);
    vision::Mat temp;
    dbg_messsage("ran until decleration")
    gpuimage.upload(image);
    vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create();
    p_orb_d->setBlurForDescriptor(true);
    vision::cuda::GpuMat kp_d;
    vision::cuda::GpuMat desc_d;
    p_orb_d->detectAndComputeAsync(gpuimage,cv::cuda::GpuMat(), kp_d, desc_d, false, current_stream);
    dbg_messsage("Feature points detected and computed")


}
