#include "clustring.h"



void ml_pack_clustering::cluster(vector<_image_description> &image_descriptions, vector<vector<float>> &centroids,vector<vector<_scalarized_image_coordinates>> &clusters,int k, int features)
{

    int number_epoch = 20;

    // compute_feature(images,image_descriptions,features);
    vector<_scalarized_image_coordinates> scalar_coordinates;
    scalarize_images(image_descriptions,scalar_coordinates,features);
    int dimention = scalar_coordinates[0].scalarized_discriptor.size();
    dbg_d(dimention);

    vector<vector<float>> data;
    for (int i =0; i< scalar_coordinates.size();i++)
    {
        data.push_back(scalar_coordinates[i].scalarized_discriptor);
    }

// ;
//     cv::Mat matAngles(data.size(), data.at(0).size(), CV_64FC1);
//     for(int i=0; i<matAngles.rows; ++i)
//      for(int j=0; j<matAngles.cols; ++j)
//           matAngles.at<float>(i, j) = data.at(i).at(j);


//     vision::Mat result,center;
//     vision::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 10, 1.0);
//     vision::kmeans(data,k,result,criteria,500,vision::KMEANS_PP_CENTERS,center);
//     std::cout << "Rows: " << center.rows << std::endl; //Hopefully 256
//     std::cout << "Cols: " << center.cols << std::endl; //Hopefully 256
//     // vision::imshow("m1",matAngles);
//     dbg_d(matAngles.rows)
//     vision::waitKey();



    for (int i=0;i <k;i++) {
        vector<_scalarized_image_coordinates> temp;
        clusters.push_back(temp);
    }
    srand(time(0));
    //generate centroids
    for (int i = 0; i < k; ++i) {
        centroids.push_back(data.at(rand() % data.size()));
    }


    while(number_epoch)
    {
        dbg_d(number_epoch);
        number_epoch--;
        for(int i =0;i< scalar_coordinates.size();i++)
        {
            float min_dis = std::numeric_limits<float>::max();
            int clusterId=0;
            for (int j=0;j<centroids.size();j++)
            {
                auto temp = distance(scalar_coordinates[i].scalarized_discriptor,centroids[j]);
                if(temp < min_dis) {min_dis = temp;
                clusterId =j;
                }


            }

            clusters[clusterId].push_back(scalar_coordinates[i]);
        }
        if(!number_epoch){break;}
        for (int i = 0; i < k; ++i)
        {
            centroids[i] = find_centroid(clusters[i]);
            clusters[i].clear();
        }


    }


    // for(int i=0;i<centroids.size();i++)
    // {
    //     for (int j=195;j<200;j++)
    //     {
    //         // cout << centroids[i][j]<<" ";
    //         dbg_d(centroids[i][j]);
    //     }
    //     // cout<<endl;
    // }





}









    void ml_pack_clustering::scalarize_images(vector<_image_description> &image_descriptions,vector<_scalarized_image_coordinates> &scalar_coordinates,int features)
    {


        _image_description zeroth =  image_descriptions[0];

        dbg_messsage("[CLUSTRING] scalarizing images")
        vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create(features);
        p_orb_d->setBlurForDescriptor(true);

        vision::cuda::Stream stream;
        vision::cuda::GpuMat zeroth_gpu;
        zeroth_gpu.upload(zeroth.image.image);
        vision::cuda::GpuMat zero_kep;
        vision::cuda::GpuMat zero_desc;
        p_orb_d->detectAndComputeAsync(zeroth_gpu,vision::cuda::GpuMat(), zero_kep, zero_desc, false, stream);
        vision::Ptr<vision::cuda::DescriptorMatcher> matcher = vision::cuda::DescriptorMatcher::createBFMatcher(vision::NORM_HAMMING);
        for (int i =0; i < image_descriptions.size();i++)
        {
            vision::cuda::GpuMat temp_gpu;
            temp_gpu.upload(image_descriptions[i].image.image);
            vision::cuda::GpuMat temp_kep;
            vision::cuda::GpuMat temp_desc;
            p_orb_d->detectAndComputeAsync(temp_gpu,vision::cuda::GpuMat(), temp_kep, temp_desc, false, stream);
            int feature_point_matches_count=0;
            std::vector<std::vector<vision::DMatch> > knn_matches_temp;
            matcher->knnMatch(zero_desc,temp_desc,knn_matches_temp,features);
            {
                auto distance_from_zeroth_discriptor  = knn_matches_temp.begin();
                if(distance_from_zeroth_discriptor->size() < features)
                {
                    dbg_messsage("image does not have enough Features")
                    continue;
                }
                else
                {
                    _scalarized_image_coordinates image_temp ;
                    image_temp.image =&image_descriptions[i];

                    for(int j =0;j<features;j++)
                    {
                        image_temp.scalarized_discriptor.push_back((*distance_from_zeroth_discriptor)[j].distance);
                    }
                    scalar_coordinates.push_back(image_temp);
                }

            }
        }



    }










    float ml_pack_clustering::distance(vector<float> a,vector<float> b )
    {
        assert(a.size() == b.size());

        float distance =0;

        for (int i=0;i<a.size(); i++)
        {
            distance += pow((a[i]-b[i]) , 2);
        }

        return sqrt(distance);
        }



vector<float> ml_pack_clustering::find_centroid(vector<_scalarized_image_coordinates> &cluster)
{
    vector<float> newCentroid;

    for (int j=0;j<cluster[0].scalarized_discriptor.size();j++)
    {
        float temp =0;
        for(int i=0;i< cluster.size();i++)
    {
        temp+= cluster[i].scalarized_discriptor[j];
    }
    temp /= cluster.size();
    newCentroid.push_back(temp);
    }

return newCentroid;

}



void ml_pack_clustering:: scalarize_image(_image_description&descriptions,_image_description &reference,_scalarized_image_coordinates &scalar_coordinates,int features)
{
    _image_description zeroth =  reference;

        dbg_messsage("[CLUSTRING] scalarizing images")
        vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create(features);
        p_orb_d->setBlurForDescriptor(true);

        // vision::cuda::Stream stream;
        // vision::cuda::GpuMat zeroth_gpu;
        // zeroth_gpu.upload(zeroth.image.image);
        // vision::cuda::GpuMat zero_kep;
        // vision::cuda::GpuMat zero_desc;
        // p_orb_d->detectAndComputeAsync(zeroth_gpu,vision::cuda::GpuMat(), zero_kep, zero_desc, false, stream);
        vision::Ptr<vision::cuda::DescriptorMatcher> matcher = vision::cuda::DescriptorMatcher::createBFMatcher(vision::NORM_HAMMING);

         vision::cuda::GpuMat temp_gpu;
            temp_gpu.upload(descriptions.image.image);
            // vision::cuda::GpuMat temp_kep;
            // vision::cuda::GpuMat temp_desc;
            // p_orb_d->detectAndComputeAsync(temp_gpu,vision::cuda::GpuMat(), temp_kep, temp_desc, false, stream);
            int feature_point_matches_count=0;
            std::vector<std::vector<vision::DMatch> > knn_matches_temp;
            matcher->knnMatch(reference.desc,descriptions.desc,knn_matches_temp,features);
            {
                auto distance_from_zeroth_discriptor  = knn_matches_temp.begin();
                if(distance_from_zeroth_discriptor->size() < features)
                {
                    dbg_messsage("image does not have enough Features")

                }
                else
                {
                    _scalarized_image_coordinates image_temp ;
                    image_temp.image = &descriptions;

                    for(int j =0;j<features;j++)
                    {
                        image_temp.scalarized_discriptor.push_back((*distance_from_zeroth_discriptor)[j].distance);
                    }
                    scalar_coordinates = image_temp;
                }

            }


}




void ml_pack_clustering:: compare_video_in_clusters(_image_description &refence,vector<vector<_scalarized_image_coordinates>>&clusters,vector<vector<float>> &centroids, int main_cluster, int features)
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

    while(true)
  {
    try{
    // capture frame
    vision::Mat frame;
    if(!cap.read(frame))
    {
      continue;
    }

    // upload to GPU
    vision::cuda::GpuMat frame_d;
    frame_d.upload(frame, stream);
    stream.waitForCompletion();

    // convert to grayscale
    vision::cuda::GpuMat frame_gray_d;
    vision::cuda::cvtColor(frame_d, frame_gray_d,vision::COLOR_BGR2GRAY, 0, stream);

    // create CUDA ORB feature detector
    vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create(features);
    p_orb_d->setBlurForDescriptor(true);

    // detect and compute features
    vision::cuda::GpuMat kp_d;
    vision::cuda::GpuMat desc_d;
    p_orb_d->detectAndComputeAsync(frame_gray_d, vision::cuda::GpuMat(), kp_d, desc_d, false, stream);
    stream.waitForCompletion();
    _image_description frameDesc;
    frameDesc.desc = desc_d;
    frameDesc.kp =kp_d;

    // dbg_messsage("extracted features");

    _scalarized_image_coordinates frameScalar;
    scalarize_image(frameDesc,refence,frameScalar,features);
    if(frameScalar.scalarized_discriptor.size() < features)
    {
        continue;
    }
    // dbg_messsage("Scalarized");
    map <float,int> cluster_distance_map;
    predictCluster(cluster_distance_map,frameScalar,centroids);

    if(cluster_distance_map.size()<main_cluster)
    {
        main_cluster = cluster_distance_map.size();
    }
    vector<int> ranks;
    for (auto c:cluster_distance_map)
    {
        ranks.push_back(c.second);
    }
    // dbg_d(clusters[ranks[1]].size());

    vision::Ptr<vision::cuda::DescriptorMatcher> matcher = vision::cuda::DescriptorMatcher::createBFMatcher(vision::NORM_HAMMING);
    vector<int> feature_point_match_count;
    for(int i=0;i<main_cluster;i++)
    {
        for (int j =0; j< clusters[ranks[i]].size();j++)
        {
            int feature_point_matches_count=0;
            dbg_d(feature_point_matches_count);
            std::vector<std::vector<vision::DMatch> > knn_matches_temp;
            matcher->knnMatch(desc_d,clusters[ranks[i]][j].image->desc,knn_matches_temp,2);
            for(std::vector<std::vector<vision::DMatch> >::const_iterator it = knn_matches_temp.begin(); it != knn_matches_temp.end(); ++it)
                {
                    if(it->size() > 1 && (*it)[0].distance/(*it)[1].distance < 0.7)
                    {
                    feature_point_matches_count++;
                    }
                }
            if(feature_point_matches_count > 20)
            {
                dbg_d(clusters[ranks[i]][j].image->image.location);
                vision::Mat imag = vision::imread(clusters[ranks[i]][j].image->image.location);
                vision::imshow("frame",imag);
                vision::waitKey(0);
                vision::destroyWindow("frame");
            }

        }


    }

    vision::imshow("video",frame);
    vision::waitKey(1);
    // dbg_messsage("showing image");


    }
    catch (std::exception &e)
    {
        dbg_d(e.what());
    }
    }

}



void ml_pack_clustering:: predictCluster(map<float,int> &cluster_order,_scalarized_image_coordinates &scalarized_image, vector<vector<float>> &centroids )
{

    for(int i=0;i< centroids.size();i++)
    {
        float dist = distance(centroids[i],scalarized_image.scalarized_discriptor);
        cluster_order.insert(make_pair(dist,i));
    }
}
