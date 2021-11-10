#include "clustring.h"


// This function does KMeans Clustering on ORB feature descriptors of all the images 
//and then using the BF Matcher to find the best matched image from whole database.

void ml_pack_clustering::cluster(_visi &images,int k, int features)
{

    int number_epoch = 20;
    vector<_image_description> image_descriptions;
    compute_feature(images,image_descriptions,features);
    vector<_scalarized_image_coordinates> scalar_coordinates;
    scalarize_images(image_descriptions,scalar_coordinates,features);
    int dimention = scalar_coordinates[0].scalarized_discriptor.size();
    dbg_d(dimention);

    vector<vector<float>> data;
    for (int i =0; i< scalar_coordinates.size();i++)
    {
        data.push_back(scalar_coordinates[i].scalarized_discriptor);
    }

    vector<vector<float>> centroids;
    vector<vector<vector<float>>> clusters(k);
    for (int i=0;i <k;i++) {
        vector<vector<float>> temp;
        clusters.push_back(temp);
    }
    srand(time(0));
    //generate centroids
    for (int i = 0; i < k; ++i) {
        centroids.push_back(data.at(rand() % data.size()));
    }

    // Alloting a cluster ID to every cluster
    while(number_epoch)
    {
        dbg_d(number_epoch);
        number_epoch--;
        for(int i =0;i< data.size();i++)
        {
            float min_dis = std::numeric_limits<float>::max();
            int clusterId=0;
            for (int j=0;j<centroids.size();j++)
            {
                auto temp = distance(data[i],centroids[j]);
                if(temp < min_dis) {min_dis = temp;
                clusterId =j;
                }


            }

            clusters[clusterId].push_back(data[i]);
        }

        for (int i = 0; i < k; ++i)
        {
            centroids[i] = find_centroid(clusters[i]);
            clusters[i].clear();
        }


    }


    for(int i=0;i<centroids.size();i++)
    {
        for (int j=195;j<200;j++)
        {
            cout << centroids[i][j]<<" ";
        }
        cout<<endl;
    }

}

//The function scalarize_images detect the ORB feature descriptors and further finding the best matches
// It also finds the distance all the descriptors from a pre considered origin(Zeroth Descriptor) and returning them in a vector 'scalar_coordinantes'

void ml_pack_clustering::scalarize_images(vector<_image_description> &image_descriptions,vector<_scalarized_image_coordinates> &scalar_coordinates,int features)
{


    _image_description zeroth =  image_descriptions[0];

    dbg_message("[CLUSTERING] scalarizing images")
    vision::Ptr<vision::cuda::ORB> p_orb_d = vision::cuda::ORB::create();
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
            to distance_from_zeroth_discriptor  = knn_matches_temp.begin();
            if(distance_from_zeroth_discriptor->size() < features)
            {
                dbg_messsage("Image does not have enough Features")
                continue;
            }
            else
            {
                _scalarized_image_coordinates image_temp ;
                image_temp.image.location =image_descriptions[i].image.location;

                for(int j =0;j<features;j++)
                {
                    image_temp.scalarized_discriptor.push_back((*distance_from_zeroth_discriptor)[j].distance);
                }
                scalar_coordinates.push_back(image_temp);
            }

        }
    }



}









// Finding euclidean distance from the centroid 
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


// Finding Centroid for further clustering
vector<float> ml_pack_clustering:: find_centroid(vector<vector<float>> &cluster)
{
    vector<float> newCentroid;

    for (int j=0;j<cluster[0].size();j++)
    {
        float temp =0;
        for(int i=0;i< cluster.size();i++)
    {
        temp+= cluster[i][j];

    }
    temp /= cluster.size();
    newCentroid.push_back(temp);
    }

return newCentroid;

}
