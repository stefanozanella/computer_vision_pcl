#include <pcl/console/parse.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>    
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/people/ground_based_people_detection_app.h>
#include <pcl/io/pcd_io.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>

typedef pcl::PointXYZRGBA PointT;
typedef pcl::PointCloud<PointT> PointCloudT;

/// PCL viewer ///
pcl::visualization::PCLVisualizer viewer("PCL Viewer");

enum { COLS = 640, ROWS = 480 };

int print_help()
{
  cout << "*******************************************************" << std::endl;
  cout << "Ground based people detection app options:" << std::endl;
  cout << "   --help    <show_this_help>" << std::endl;
  cout << "   --svm     <path_to_svm_file>" << std::endl;
  cout << "   --conf    <minimum_HOG_confidence (default = -1.5)>" << std::endl;
  cout << "   --min_h   <minimum_person_height (default = 1.3)>" << std::endl;
  cout << "   --max_h   <maximum_person_height (default = 2.3)>" << std::endl;
  cout << "   --sample  <sampling_factor (default = 1)>" << std::endl;
  cout << "   --pcd     <path_to_pcd_file>" << std::endl;
  cout << "*******************************************************" << std::endl;
  return 0;
}

struct callback_args{
	// structure used to pass arguments to the callback function
	PointCloudT::Ptr clicked_points_3d;
	pcl::visualization::PCLVisualizer::Ptr viewerPtr;
};

void
pp_callback (const pcl::visualization::PointPickingEvent& event, void* args)
{
  struct callback_args* data = (struct callback_args *)args;
  if (event.getPointIndex () == -1)
    return;
  PointT current_point;
  event.getPoint(current_point.x, current_point.y, current_point.z);
  data->clicked_points_3d->points.push_back(current_point);
  // Draw clicked points in red:
  pcl::visualization::PointCloudColorHandlerCustom<PointT> red (data->clicked_points_3d, 255, 0, 0);
  data->viewerPtr->removePointCloud("clicked_points");
  data->viewerPtr->addPointCloud(data->clicked_points_3d, red, "clicked_points");
  data->viewerPtr->setPointCloudRenderingProperties(pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 10, "clicked_points");
  std::cout << current_point.x << " " << current_point.y << " " << current_point.z << std::endl;
}

int main (int argc, char** argv)
{
  if(pcl::console::find_switch (argc, argv, "--help") || pcl::console::find_switch (argc, argv, "-h"))
        return print_help();

  /// Dataset Parameters:
  std::string filename = "../dataset/people/five_people.pcd";
  std::string svm_filename = "../dataset/trainedLinearSVMForPeopleDetectionWithHOG.yaml";
  float min_confidence = -1.5;
  float min_height = 1.3;
  float max_height = 2.3;
  float voxel_size = 0.06;
  float sampling_factor = 1;
  Eigen::Matrix3f rgb_intrinsics_matrix;
  rgb_intrinsics_matrix << 525, 0.0, 319.5, 0.0, 525, 239.5, 0.0, 0.0, 1.0; // Kinect RGB camera intrinsics

  // Read if some parameters are passed from command line:
  pcl::console::parse_argument (argc, argv, "--svm", svm_filename);
  pcl::console::parse_argument (argc, argv, "--conf", min_confidence);
  pcl::console::parse_argument (argc, argv, "--min_h", min_height);
  pcl::console::parse_argument (argc, argv, "--max_h", max_height);
  pcl::console::parse_argument (argc, argv, "--sample", sampling_factor);
  pcl::console::parse_argument (argc, argv, "--pcd", filename);

  // Read Kinect data:
  PointCloudT::Ptr cloud = PointCloudT::Ptr (new PointCloudT);
  if (pcl::io::loadPCDFile(filename, *cloud) < 0)
  {
    cerr << "Failed to read test file `"<< filename << "`." << endl;
    return (-1);
  } 

  // Create classifier for people detection:
  pcl::people::PersonClassifier<pcl::RGB> person_classifier;
  person_classifier.loadSVMFromFile(svm_filename);   // load trained SVM

  // People detection app initialization:
  pcl::people::GroundBasedPeopleDetectionApp<PointT> people_detector;    // people detection object
  people_detector.setVoxelSize(voxel_size);                        // set the voxel size
  people_detector.setIntrinsics(rgb_intrinsics_matrix);            // set RGB camera intrinsic parameters
  people_detector.setClassifier(person_classifier);                // set person classifier
  people_detector.setHeightLimits(min_height, max_height);         // set person classifier
  people_detector.setSamplingFactor(sampling_factor);              // set a downsampling factor to the point cloud (for increasing speed)
//  people_detector.setSensorPortraitOrientation(true);              // set sensor orientation to vertical

  // Display pointcloud:
  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
//  viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");
//  viewer.setCameraPosition(0,0,-2,0,-1,0,0);
//
//  // Add point picking callback to viewer:
//  struct callback_args cb_args;
//  PointCloudT::Ptr clicked_points_3d (new PointCloudT);
//  cb_args.clicked_points_3d = clicked_points_3d;
//  cb_args.viewerPtr = pcl::visualization::PCLVisualizer::Ptr(&viewer);
//  viewer.registerPointPickingCallback (pp_callback, (void*)&cb_args);
//  std::cout << "Shift+click on three floor points, then press 'Q'..." << std::endl;
//
//  // Spin until 'Q' is pressed:
//  viewer.spin();
//  std::cout << "done." << std::endl;
  
  // Ground plane estimation:
  Eigen::VectorXf ground_coeffs(4);
//  std::vector<int> clicked_points_indices;
//  for (unsigned int i = 0; i < clicked_points_3d->points.size(); i++)
//    clicked_points_indices.push_back(i);
//  pcl::SampleConsensusModelPlane<PointT> model_plane(clicked_points_3d);
//  model_plane.computeModelCoefficients(clicked_points_indices,ground_coeffs);
//  std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;
  // Manual ground plane estimation

  // Automatic ground plane estimation
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (4000);
  seg.setDistanceThreshold (0.08);
  seg.setEpsAngle(0.5);
  seg.setAxis(Eigen::Vector3f(0, 1, 0));
  seg.setInputCloud (cloud);
  seg.segment (*inliers, *coefficients);
  if (inliers->indices.size () == 0)
  {
    std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  std::cout << "Automatic ground plane: " << coefficients->values[0] << " " << coefficients->values[1] << " " << coefficients->values[2] << " " << coefficients->values[3] << std::endl;
  // Automatic ground plane estimation

  int flip = coefficients->values[1] > 0 ? 1 : -1;
  ground_coeffs[0] = coefficients->values[0] * flip;
  ground_coeffs[1] = coefficients->values[1] * flip;
  ground_coeffs[2] = coefficients->values[2] * flip;
  ground_coeffs[3] = coefficients->values[3] * flip;
  std::cout << "Ground plane: " << ground_coeffs(0) << " " << ground_coeffs(1) << " " << ground_coeffs(2) << " " << ground_coeffs(3) << std::endl;

  // Perform people detection on the new cloud:
  std::vector<pcl::people::PersonCluster<PointT> > clusters;   // vector containing persons clusters
  people_detector.setInputCloud(cloud);
  people_detector.setGround(ground_coeffs);                    // set floor coefficients
  people_detector.compute(clusters);                           // perform people detection
std::cout << "A" << std::endl;
  ground_coeffs = people_detector.getGround();                 // get updated floor coefficients

  // Extract the inliers
  pcl::ExtractIndices<PointT> extract;
  pcl::PointCloud<PointT>::Ptr ground_plane (new pcl::PointCloud<PointT>);
  extract.setInputCloud (cloud);
  extract.setIndices (inliers);
  extract.setNegative (false);		// to make filter method to return "outliers" instead of "inliers"
  extract.filter (*ground_plane);
  std::cerr << "PointCloud representing the planar component: " << ground_plane->width * ground_plane->height << " data points." << std::endl;


  // Initialize new viewer:
  pcl::visualization::PCLVisualizer viewer("PCL Viewer");          // viewer initialization
  viewer.setCameraPosition(0,0,-2,0,-1,0,0);

  // Draw cloud and people bounding boxes in the viewer:
  viewer.addPointCloud<PointT> (cloud, rgb, "input_cloud");

  // Draw auto-estimated ground plane
  viewer.addPlane(*coefficients);
  viewer.addPointCloud(ground_plane,
      pcl::visualization::PointCloudColorHandlerCustom<PointT>(ground_plane, 255,0,0),
      "ground");

  unsigned int k = 0;
  for(std::vector<pcl::people::PersonCluster<PointT> >::iterator it = clusters.begin(); it != clusters.end(); ++it)
  {
    if(it->getPersonConfidence() > min_confidence)             // draw only people with confidence above a threshold
    {
      // draw theoretical person bounding box in the PCL viewer:
      it->drawTBoundingBox(viewer, k);
      k++;
    }
  }
  std::cout << k << " people found" << std::endl;
  viewer.spin();

  return 0;
}

//  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
//  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
//  // Create the segmentation object
//  pcl::SACSegmentation<PointT> seg;
//  // Optional
//  seg.setOptimizeCoefficients (true);
//  // Mandatory
//  seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
//  seg.setMethodType (pcl::SAC_RANSAC);
//  seg.setMaxIterations (1000);
//  seg.setDistanceThreshold (0.1);
//  seg.setEpsAngle(0.5);
//  seg.setAxis(Eigen::Vector3f(0, 1, 0));
//
//  // Create the filtering object
//  pcl::ExtractIndices<PointT> extract;
//
//  int i = 0, nr_points = (int) cloud_filtered->points.size ();
//  // Extract planes while 30% of the original cloud is still there
//  while (cloud_filtered->points.size () > 0.3 * nr_points)
//  {
//    // Segment the largest planar component from the remaining cloud
//    seg.setInputCloud (cloud_filtered);
//    seg.segment (*inliers, *coefficients);
//    if (inliers->indices.size () == 0)
//    {
//      std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
//      break;
//    }
//
//    std::cerr << "Plane coefficients" << std::endl;
//    std::cerr << coefficients->values[0] << " " << coefficients->values[1] << " " << coefficients->values[2] << " " << coefficients->values[3] << std::endl << std::endl;
//    // Extract the inliers
//    extract.setInputCloud (cloud_filtered);
//    extract.setIndices (inliers);
//    extract.setNegative (false);
//    extract.filter (*plane_cloud);
//    std::cerr << "PointCloud representing the planar component: " << plane_cloud->width * plane_cloud->height << " data points." << std::endl;
//
//    // Create the filtering object
//    extract.setNegative (true);		// to make filter method to return "outliers" instead of "inliers"
//    extract.filter (*remaining_cloud);
//    cloud_filtered.swap (remaining_cloud);
//    i++;
//
