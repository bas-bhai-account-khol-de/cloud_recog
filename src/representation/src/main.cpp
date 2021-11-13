#include <iostream>
#include "clustring.h"
#include "sources.h"



using namespace std;
using namespace ml_pack_clustering;

int main()
{
    // init_reco();
    // _vi dataset_images;
    // import_from_directory(dataset_images,SCHOOL_BAG_DATASET_FOLDER);
    // _vkps kp_dataset;
    // _vgpuM desc_dataset;
    // compute_feature(dataset_images,kp_dataset,desc_dataset);
    // compare_video(kp_dataset,desc_dataset,dataset_images);
    // // compare_image("/media/prakhar/Linux FIles/Flam Apps/Testing and experiment/DataSets/school_bag/Image_49 Bag.jpg",kp_dataset,desc_dataset,dataset_images);

    // vision::Mat test_images =  vision::imread("/media/prakhar/Linux FIles/Flam Apps/Testing and experiment/DataSets/school_bag/Image_48 Bag.jpg",vision::IMREAD_GRAYSCALE);
    // float average = _image_gradient_magnitude(test_images);
    // dbg_d(average);

    // vision::waitKey(1000);


    // //_video_countours_average();
    // _video_gradient_magnitude();

    // int number_of_folders = 2;
    // string base  = "/media/prakhar/Linux FIles/Flam Apps/Testing and experiment/DataSets/";
    //   DIR *dir = opendir(base.c_str());
    // vector <string> folders;
    // struct dirent *entry = readdir(dir);

    // while (entry != NULL && number_of_folders)
    // {
    //     if (entry->d_type == DT_DIR && strcmp(entry->d_name,".") &&  strcmp(entry->d_name,".."))
    //         printf("%s\n", entry->d_name);
    //      number_of_folders--;
    //     folders.push_back(string(entry->d_name));
    //     entry = readdir(dir);
    // }

    // closedir(dir);

    // vision::waitKey(1000);
    // // string folders[number_of_folders]={"Buildings","apple fruit","books","laptop","mug","school bag","smartphone","Trees","Tshirt","Water Bottle"};
    // vector<vector<_image_description>> images_in_dataset;

    // for (int i = 0;i < folders.size();i++)
    // {
    // vector<_image_description> images_temp;
    // folder_images_image_descriptions(images_temp,base+folders[i]);
    // images_in_dataset.push_back(images_temp);
    // }

    // compare_video_in_batch(images_in_dataset,folders.size());

    _visi images;

    import_from_directory(images,"/media/prakhar/Linux FIles/Flam Apps/Testing and experiment/DataSets/beseter");
    vector<_image_description> image_descriptions;
    compute_feature(images,image_descriptions);

    vector<vector<float>> centroids;
    vector<vector<_scalarized_image_coordinates>> clusters;
    cluster(image_descriptions,centroids,clusters);
    dbg_messsage("clustered");
    dbg_d(clusters[0].size());
    compare_video_in_clusters(image_descriptions[0],clusters,centroids);



    return 0;
}
