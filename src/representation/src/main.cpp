#include <iostream>
#include"reco.h"
#include "sources.h"

using namespace std;

int main()
{
    init_reco();
    _vi dataset_images;
    import_from_directory(dataset_images,SCHOOL_BAG_DATASET_FOLDER);
    _vkps kp_dataset;
    _vgpuM desc_dataset;
    compute_feature(dataset_images,kp_dataset,desc_dataset);
    compare_video(kp_dataset,desc_dataset,dataset_images);
    // compare_image("/media/prakhar/Linux FIles/Flam Apps/Testing and experiment/DataSets/school_bag/Image_49 Bag.jpg",kp_dataset,desc_dataset,dataset_images);
    return 0;
}
