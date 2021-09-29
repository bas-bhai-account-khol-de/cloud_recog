
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
