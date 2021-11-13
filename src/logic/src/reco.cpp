#include"reco.h"
#include <cstdio>


void init_reco(int v)
{
    cout<<"Recognization library initilized int v"<<endl;
}


void import_from_directory(_vi &images,string folder_name )
{


    if(folder_name == "")
    {
        cerr<< "DATASET_LOADING: Please enter a valid address......";
        assert(folder_name!= "");
        return ;
    }
    struct dirent *entry;
    DIR *dir = opendir(folder_name.c_str());

    if (dir == NULL) {
        return ;
    }
    int i=0;
    while ((entry = readdir(dir)) != NULL) {
    if(!strcmp(".",entry->d_name)|| !strcmp("..",entry->d_name))
    {
        continue;
    }
    i++;
    cout << "DATASET_LOADING: "<<entry->d_name << " "<<i<<endl;
    vision::Mat image_temp = vision::imread(folder_name+"/"+entry->d_name,vision::IMREAD_GRAYSCALE);
    images.push_back(image_temp);

    }
    closedir(dir);
    dbg_d(images.size());
    return ;

}




void import_from_directory(_visi &images,string folder_name )
{


    if(folder_name == "")
    {
        cerr<< "DATASET_LOADING: Please enter a valid address......";
        assert(folder_name!= "");
        return ;
    }
    struct dirent *entry;
    DIR *dir = opendir(folder_name.c_str());

    if (dir == NULL) {
        return ;
    }
    int i=0;
    while ((entry = readdir(dir)) != NULL) {
    if(!strcmp(".",entry->d_name)|| !strcmp("..",entry->d_name))
    {
        continue;
    }
    i++;
    cout << "DATASET_LOADING: "<<entry->d_name << " "<<i<<endl;
    _vision_images temp_images;
    string location = folder_name+"/"+entry->d_name;
    temp_images.image  = vision::imread(location,vision::IMREAD_GRAYSCALE);
    temp_images.location = location;
    images.push_back(temp_images);

    }
    closedir(dir);
    dbg_d(images.size());
    return ;

}








void compute_feature(_vi &images,_vkps &kp_dataset,_vgpuM &desc_dataset,unsigned int max_features )
{

    dbg_messsage("FEATURE_EXTRACTION: Copmutation started");
    //*********************************************************
    //INITILIZATIONS

    vision::cuda::Stream current_stream;
    vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create(max_features);
    p_orb_d->setBlurForDescriptor(true);
    //---------------------------------------------------
    int i=0;
    for (auto img : images)
    {
        i++;
        try
        {
        vision::cuda::GpuMat gpuimage;
        gpuimage.upload(img);
        vision::cuda::GpuMat kp_d;
        vision::cuda::GpuMat desc_d;
        p_orb_d->detectAndComputeAsync(gpuimage,vision::cuda::GpuMat(), kp_d, desc_d, false, current_stream);
        current_stream.waitForCompletion();
        std::vector<vision::KeyPoint> keypointsCPU;
        p_orb_d->convert(kp_d,keypointsCPU);
        kp_dataset.push_back(keypointsCPU);
        desc_dataset.push_back(desc_d);}
        catch(...)
        {
            dbg_messsage("FEATURE_EXTRACTION: image failes ");
            dbg_d(i);
        }
    }

    dbg_messsage("FEATURE_EXTRACTION: calculated FEATURES for every image");

}



void compute_feature(_visi &images,vector<_image_description> &image_desc,unsigned int max_features )
{

    dbg_messsage("FEATURE_EXTRACTION: Copmutation started");
    //*********************************************************
    //INITILIZATIONS

    vision::cuda::Stream current_stream;
    vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create(max_features);
    p_orb_d->setBlurForDescriptor(true);
    //---------------------------------------------------
    int i=0;
    for (auto img : images)
    {
        i++;
        try
        {vision::cuda::GpuMat gpuimage;
        gpuimage.upload(img.image);
        vision::cuda::GpuMat kp_d;
        vision::cuda::GpuMat desc_d;
        p_orb_d->detectAndComputeAsync(gpuimage,vision::cuda::GpuMat(), kp_d, desc_d, false, current_stream);
        current_stream.waitForCompletion();
        _image_description image_temp ;
        image_temp.image = img;
        image_temp.desc=desc_d;
        image_temp.kp = kp_d;
        image_desc.push_back(image_temp);
        // std::vector<vision::KeyPoint> keypointsCPU;
        // p_orb_d->convert(kp_d,keypointsCPU);
        // kp_dataset.push_back(keypointsCPU);
        // desc_dataset.push_back(desc_d);
        }
        catch(...)
        {
            dbg_messsage("FEATURE_EXTRACTION: image failes ");
            dbg_d(i);
        }
    }

    dbg_messsage("FEATURE_EXTRACTION: calculated FEATURES for every image");

}



void compare_image(string query_image_path , _vkps &keypoints,_vgpuM &descriptors,_vi &images)
{
    vision::Mat image_temp = vision::imread(query_image_path,vision::IMREAD_GRAYSCALE);
    vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create();
    p_orb_d->setBlurForDescriptor(true);
    vision::cuda::GpuMat kp_d;
    vision::cuda::GpuMat desc_d;
    vision::cuda::GpuMat gpuimage;
    gpuimage.upload(image_temp);
    vision::cuda::Stream current_stream;
    p_orb_d->detectAndComputeAsync(gpuimage,vision::cuda::GpuMat(), kp_d, desc_d, false, current_stream);
    current_stream.waitForCompletion();
    vector<vision::KeyPoint> keypoints_temp;
    p_orb_d->convert(kp_d,keypoints_temp);
    vision::Ptr<vision::cuda::DescriptorMatcher> matcher = vision::cuda::DescriptorMatcher::createBFMatcher(vision::NORM_HAMMING);
    dbg_messsage("COMPARE_IMAGE: initilized matcher");
    vector<int> feature_point_match_count;
    clock_t begin = clock();
    for(int i=0;i<descriptors.size();i++)
    {
        int feature_point_matches_count=0;
        std::vector<std::vector<vision::DMatch> > knn_matches_temp;
        matcher->knnMatch(desc_d,descriptors[i],knn_matches_temp,2);
        for(std::vector<std::vector<vision::DMatch> >::const_iterator it = knn_matches_temp.begin(); it != knn_matches_temp.end(); ++it)
        {
            if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.6)
            {
               feature_point_matches_count++;
            }
        }
        feature_point_match_count.push_back(feature_point_matches_count);

    }
    clock_t end = clock();
    std::cout << double(end-begin) / CLOCKS_PER_SEC  << std::endl;
    int index_max_element  = distance(feature_point_match_count.begin(),max_element(feature_point_match_count.begin(),feature_point_match_count.end()));
    dbg_d(index_max_element);

    vision::imshow("frame",images[index_max_element]);
    vision::waitKey(0);


}

void compare_video(_vkps &keypoints,_vgpuM &descriptors,_vi &images)
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

    // upload to GPU
    vision::cuda::GpuMat frame_d;
    frame_d.upload(frame, stream);

    // convert to grayscale
    vision::cuda::GpuMat frame_gray_d;
    vision::cuda::cvtColor(frame_d, frame_gray_d,vision::COLOR_BGR2GRAY, 0, stream);

    // create CUDA ORB feature detector
    vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create();
    p_orb_d->setBlurForDescriptor(true);

    // detect and compute features
    vision::cuda::GpuMat kp_d;
    vision::cuda::GpuMat desc_d;
    p_orb_d->detectAndComputeAsync(frame_gray_d, vision::cuda::GpuMat(), kp_d, desc_d, false, stream);
    stream.waitForCompletion();
    vision::Ptr<vision::cuda::DescriptorMatcher> matcher = vision::cuda::DescriptorMatcher::createBFMatcher(vision::NORM_HAMMING);
    vector<int> feature_point_match_count;
    for(int i=0;i<descriptors.size();i++)
    {
        int feature_point_matches_count=0;
        std::vector<std::vector<vision::DMatch> > knn_matches_temp;
        matcher->knnMatch(desc_d,descriptors[i],knn_matches_temp,2);
        for(std::vector<std::vector<vision::DMatch> >::const_iterator it = knn_matches_temp.begin(); it != knn_matches_temp.end(); ++it)
        {
            if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.7)
            {
               feature_point_matches_count++;
            }
        }
        feature_point_match_count.push_back(feature_point_matches_count);

    }


    int index_max_element  = distance(feature_point_match_count.begin(),max_element(feature_point_match_count.begin(),feature_point_match_count.end()));
    dbg_d(feature_point_match_count[index_max_element]);
    if(20<feature_point_match_count[index_max_element])
    {
        vision::imshow("frame",images[index_max_element]);
        vision::waitKey(0);
        break;
    }

    vision::imshow("video",frame);

    if (vision::waitKey(5) == 27)
  {
   cout << "Esc key is pressed by user. Stoppig the video" << endl;
   break;
  }
  }
}

void folder_images_image_descriptions(vector<_image_description> &folder_image_desc,string folder_name)
{
    _visi images_from_folder ;
    import_from_directory(images_from_folder,folder_name);

    compute_feature(images_from_folder,folder_image_desc);
    for (auto img : folder_image_desc )
    {
        img.image.image.release();
    }


}


void compare_video_in_batch(vector<vector<_image_description>> &image_dataset, int number_of_clusters )
{
    string loc="";
    dbg_messsage("VIDEO_COMPARE: starting camera ....");
    vision::VideoCapture cap(0);

    if (!cap.isOpened())
    {
        std::cerr << "ERROR - Failed to open VideoCapture for device " + std::to_string(0) + "\n" << std::flush;
        return;
    }
    vision::cuda::Stream stream;
    int image_cluster_number =0;
   while(true)
  {
    // capture frame
    vision::Mat frame;
    if(!cap.read(frame))
    {
      continue;
    }

    // upload to GPU
    vision::cuda::GpuMat frame_d;
    frame_d.upload(frame, stream);

    // convert to grayscale
    vision::cuda::GpuMat frame_gray_d;
    vision::cuda::cvtColor(frame_d, frame_gray_d,vision::COLOR_BGR2GRAY, 0, stream);

    // create CUDA ORB feature detector
    vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create();
    p_orb_d->setBlurForDescriptor(true);

    // detect and compute features
    vision::cuda::GpuMat kp_d;
    vision::cuda::GpuMat desc_d;
    p_orb_d->detectAndComputeAsync(frame_gray_d, vision::cuda::GpuMat(), kp_d, desc_d, false, stream);
    stream.waitForCompletion();
    vision::Ptr<vision::cuda::DescriptorMatcher> matcher = vision::cuda::DescriptorMatcher::createBFMatcher(vision::NORM_HAMMING);
    vector<int> feature_point_match_count;
    for(int i=0;i<image_dataset[image_cluster_number].size();i++)
    {
        int feature_point_matches_count=0;
        std::vector<std::vector<vision::DMatch> > knn_matches_temp;
        matcher->knnMatch(desc_d,image_dataset[image_cluster_number][i].desc,knn_matches_temp,2);
        for(std::vector<std::vector<vision::DMatch> >::const_iterator it = knn_matches_temp.begin(); it != knn_matches_temp.end(); ++it)
        {
            if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.67)
            {
               feature_point_matches_count++;
            }
        }
        feature_point_match_count.push_back(feature_point_matches_count);
        if(20 <feature_point_matches_count)
        {
            dbg_messsage("image found");
            vision::Mat show = vision::imread(image_dataset[image_cluster_number][i].image.location);
            loc = image_dataset[image_cluster_number][i].image.location;
            vision::imshow("found Image",show);

            break;
        }

    }


    // int index_max_element  = distance(feature_point_match_count.begin(),max_element(feature_point_match_count.begin(),feature_point_match_count.end()));
    // dbg_d(feature_point_match_count[index_max_element]);
    // if(20<feature_point_match_count[index_max_element])
    // {
    //     vision::imshow("frame",images[index_max_element]);
    //     vision::waitKey(0);
    //     break;
    // }

    vision::imshow("video",frame);
    image_cluster_number++;
    if(image_cluster_number >= number_of_clusters)
    {
        image_cluster_number =0;
    }

    if (vision::waitKey(1) == (int)('p'))
  {
   cout << "Deleting" << endl;
    if(loc!="")
    {
        remove(loc.c_str());
        loc="";
    }
  }
    else if (vision::waitKey(1) == 27)
  {
   cout << "Esc key is pressed by user. Stoppig the video" << endl;
   break;
  }
  }

}




