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
  PointCloudT::Ptr cloud_filtered = PointCloudT::Ptr (new PointCloudT);
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

  // Display pointcloud:
  pcl::VoxelGrid<PointT> sor;
  sor.setInputCloud (cloud);
  sor.setLeafSize (0.01f, 0.01f, 0.01f);
  sor.filter (*cloud_filtered);

  pcl::visualization::PointCloudColorHandlerRGBField<PointT> rgb(cloud);
  // Automatic ground plane estimation
  Eigen::VectorXf ground_coeffs(4);
  pcl::ModelCoefficients::Ptr coefficients (new pcl::ModelCoefficients ());
  pcl::PointIndices::Ptr inliers (new pcl::PointIndices ());
  pcl::SACSegmentation<PointT> seg;
  seg.setOptimizeCoefficients (true);
  seg.setModelType (pcl::SACMODEL_PERPENDICULAR_PLANE);
  seg.setMethodType (pcl::SAC_RANSAC);
  seg.setMaxIterations (4000);
  seg.setDistanceThreshold (0.05);
  seg.setEpsAngle(0.6);
  seg.setAxis(Eigen::Vector3f(0, 1, 0));

  seg.setInputCloud (cloud_filtered);
  seg.segment (*inliers, *coefficients);
  if (inliers->indices.size () == 0)
  {
    std::cerr << "Could not estimate a planar model for the given dataset." << std::endl;
  }

  std::cout << "Automatic ground plane: " << coefficients->values[0] << " " << coefficients->values[1] << " " << coefficients->values[2] << " " << coefficients->values[3] << std::endl;
  // Automatic ground plane estimation

  ground_coeffs[0] = coefficients->values[0];
  ground_coeffs[1] = coefficients->values[1];
  ground_coeffs[2] = coefficients->values[2];
  ground_coeffs[3] = coefficients->values[3];

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
  extract.setInputCloud (cloud_filtered);
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
