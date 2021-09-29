#include <iostream>
#include"reco.h"
#include "sources.h"

using namespace std;

int main()
{
    init_reco();
    vector<vision::Mat> *dataset_images;
    dataset_images=import_from_directory(SCHOOL_BAG_DATASET_FOLDER);
    compute_feature(dataset_images);
    return 0;
}
