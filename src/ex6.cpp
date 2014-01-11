#include <vector>
#include <string>
#include <iostream>
#include <cstdlib> // TODO: Remove!!!!

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/filters/voxel_grid.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/search/kdtree.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/sample_consensus/sac_model_plane.h>
#include <pcl/sample_consensus/model_types.h>
#include <pcl/segmentation/sac_segmentation.h>
#include <pcl/filters/extract_indices.h>
#include <pcl/registration/icp.h>

using std::cout;
using std::cerr;
using std::endl;
using std::flush;
using std::string;
using std::vector;
using std::stringstream;
using pcl::io::loadPCDFile;
using pcl::ComparisonOps::GT;
using pcl::ComparisonOps::LT;
using pcl::visualization::PCLVisualizer;
using pcl::ModelCoefficients;
using pcl::PointIndices;
using pcl::SACMODEL_PERPENDICULAR_PLANE;
using pcl::SAC_RANSAC;

typedef pcl::PointXYZRGB Point;
typedef pcl::PointCloud<Point> PointCloud;
typedef std::vector<PointCloud::Ptr> PointClouds;
typedef pcl::ConditionalRemoval<Point> RemovalFilter;
typedef pcl::ConditionAnd<Point> RemovalCondition;
typedef pcl::FieldComparison<Point> FieldComparison;
typedef pcl::visualization::PointCloudColorHandlerRGBField<Point> RGBColor;
typedef pcl::visualization::PointCloudColorHandlerCustom<Point> CustomColor;
typedef pcl::SACSegmentation<Point> SACSegmentation;
typedef pcl::ExtractIndices<Point> ExtractIndices;

// TODO: Move this inside main, it's here just so that it's near to
// work-in-progress code.
int views_no = 6;
int src_idx = 1;  // TODO: Remove!!!!
int tgt_idx = 0;  // TODO: Remove!!!!

void DO_STUFF(PointClouds& views) {
  PointCloud::Ptr tgt = views.at(tgt_idx);
  PointCloud::Ptr src = views.at(src_idx);

  // Keypoints extraction
  PointCloud::Ptr src_keypoints (new PointCloud);
  PointCloud::Ptr tgt_keypoints (new PointCloud);

  pcl::SIFTKeypoint<Point, Point> sift;
  sift.setSearchMethod(
      pcl::search::KdTree<Point>::Ptr(new pcl::search::KdTree<Point>));
  sift.setScales(0.01f, 3, 2);
  sift.setMinimumContrast(0);

  sift.setInputCloud(src);
  sift.compute(*src_keypoints);

  sift.setInputCloud(tgt);
  sift.compute(*tgt_keypoints);

  // Normal estimation
  pcl::PointCloud<pcl::Normal>::Ptr src_normals (new pcl::PointCloud<pcl::Normal>);
  pcl::PointCloud<pcl::Normal>::Ptr tgt_normals (new pcl::PointCloud<pcl::Normal>);

  pcl::NormalEstimationOMP<Point, pcl::Normal> normal_estimation;
  normal_estimation.setSearchMethod(pcl::search::KdTree<Point>::Ptr(new pcl::search::KdTree<Point>));
  normal_estimation.setRadiusSearch(0.1);

  normal_estimation.setInputCloud(src);
  normal_estimation.compute(*src_normals);

  normal_estimation.setInputCloud(tgt);
  normal_estimation.compute(*tgt_normals);

  // Feature estimation
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr src_features (new pcl::PointCloud<pcl::FPFHSignature33>);
  pcl::PointCloud<pcl::FPFHSignature33>::Ptr tgt_features (new pcl::PointCloud<pcl::FPFHSignature33>);

  pcl::FPFHEstimationOMP<Point, pcl::Normal, pcl::FPFHSignature33> feature_estimation;
  feature_estimation.setSearchMethod(pcl::search::KdTree<Point>::Ptr(new pcl::search::KdTree<Point>));
  feature_estimation.setRadiusSearch(0.1);

  feature_estimation.setSearchSurface(src);
  feature_estimation.setInputCloud(src_keypoints);
  feature_estimation.setInputNormals(src_normals);
  feature_estimation.compute(*src_features);

  feature_estimation.setSearchSurface(tgt);
  feature_estimation.setInputCloud(tgt_keypoints);
  feature_estimation.setInputNormals(tgt_normals);
  feature_estimation.compute(*tgt_features);

  // Correspondence estimation
  pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> est;
  pcl::Correspondences correspondences, inliers;
  
  est.setInputSource(src_features);
  est.setInputTarget(tgt_features);
  est.determineCorrespondences(correspondences);

  // Correspondence rejection
  pcl::registration::CorrespondenceRejectorSampleConsensus<Point> rejector;
  rejector.setInputSource(src_keypoints);
  rejector.setInputTarget(tgt_keypoints);
  rejector.setInputCorrespondences(boost::make_shared<const pcl::Correspondences>(correspondences));
  rejector.setInlierThreshold(0.05);
  rejector.setMaxIterations(5000);
  rejector.getCorrespondences(inliers);

  // Transformation estimation
  pcl::registration::TransformationEstimationSVD<Point, Point> estimator;
  Eigen::Matrix4f transformation;
  estimator.estimateRigidTransformation(*src_keypoints, *tgt_keypoints, inliers, transformation);

  PointCloud::Ptr pre_aligned (new PointCloud);
  PointCloud::Ptr aligned (new PointCloud);
  pcl::transformPointCloud(*src, *pre_aligned, transformation);
  
  pcl::IterativeClosestPoint<Point, Point> icp;
  icp.setMaxCorrespondenceDistance(0.50);
  icp.setRANSACOutlierRejectionThreshold(0.05);
  icp.setTransformationEpsilon(0.000001);
  icp.setMaximumIterations(600);
  icp.setInputSource(pre_aligned);
  icp.setInputTarget(tgt);
  icp.align(*aligned);

  cout << (icp.hasConverged() ? "Alignment succeeded!" : "Alignment failed.") << endl;
  cout << "Alignment score: " << icp.getFitnessScore() << endl;

// DEBUG
  PCLVisualizer viewer ("Point Cloud Viewer");
  viewer.initCameraParameters();

  viewer.addPointCloud<Point>(
    src,
    CustomColor(src, 64, 0, 0),
    "src");

  viewer.addPointCloud<Point>(
    tgt,
    CustomColor(tgt, 0, 64, 0),
    "tgt");

	viewer.addPointCloudNormals<Point, pcl::Normal>(src,
      src_normals,
      100,
      0.02,
      "src_normals");

	viewer.addPointCloudNormals<Point, pcl::Normal>(tgt,
      tgt_normals,
      100,
      0.02,
      "tgt_normals");

  viewer.addPointCloud<Point>(
    src_keypoints,
    CustomColor(src_keypoints, 255, 0, 0),
    "src_keypoints");

  viewer.addPointCloud<Point>(
    tgt_keypoints,
    CustomColor(tgt_keypoints, 0, 255, 0),
    "tgt_keypoints");

  viewer.addPointCloud<Point>(
    aligned,
    CustomColor(aligned, 0, 0, 255),
    "aligned");

  viewer.addCorrespondences<Point>(src_keypoints, tgt_keypoints, inliers);

  viewer.spin();
// DEBUG
}

/**
 * Simple class that returns different RGB colors every time.
 */
struct ColorMaker {
  ColorMaker(int _size) :
    size(_size)
  {}

  int r(int idx) {
    return (80 + (255 / size) * 4 * idx) % 255;
  }

  int g(int idx) {
    return (130 + (255 / size) * 3 * idx ) % 255;
  }

  int b(int idx) {
    return (240 + (255 / size) * 2 * idx) % 255;
  }

  private:

  int size;
};

/**
 * Returns a proper path to a PCD file given an index.
 */
string filename_for_cloud(int k) {
  stringstream ss;
  ss << "../dataset/nao/" << k << ".pcd";
  return ss.str();
}

/**
 * The condition by which we decide whether a point should remain in
 * the cloud we're going to align or not.
 *
 * The rule has been found out by trial-and-error and allows to keep the
 * smallest cloud portion containing the robot in every given view. The rule is
 * basically the following:
 *
 *    (1.1 < z < 1.7) ^ (-0.7 < x < 0.5) 
 */
RemovalCondition::Ptr trimming_rule() {
  RemovalCondition::Ptr condition (new RemovalCondition);
  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("z", GT, 1.1)));
  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("z", LT, 1.7)));
  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("x", GT, -0.7)));
  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("x", LT, 0.5)));

  return condition;
}

/**
 * Performs ground plane removal on a point cloud.
 */
class GroundPlaneEraser {
  private:
    ModelCoefficients::Ptr coefficients;
    SACSegmentation seg;
    ExtractIndices extract;
    PointIndices::Ptr inliers;

  public:
    GroundPlaneEraser() :
      coefficients (new ModelCoefficients),
      inliers (new PointIndices) 
    {
      seg.setOptimizeCoefficients(true);
      seg.setModelType(SACMODEL_PERPENDICULAR_PLANE);
      seg.setMethodType(SAC_RANSAC);
      seg.setMaxIterations(4000);
      seg.setDistanceThreshold(0.01);
      seg.setEpsAngle(0.6);
      seg.setAxis(Eigen::Vector3f(0, 1, 0));
      extract.setNegative(true);
    }

    void remove_ground_plane(PointCloud::Ptr &cloud) {
      seg.setInputCloud(cloud);
      seg.segment(*inliers, *coefficients);

      extract.setInputCloud(cloud);
      extract.setIndices(inliers);
      extract.filter(*cloud);
    }
};

/**
 * Trims all clouds in given list by applying the defined trimming rule and by
 * removing the ground plane.
 *
 * See `trimming_rule()` for further details.
 */
void trim_clouds(PointClouds& clouds) {
  PointClouds trimmed_clouds;
  RemovalFilter filter (trimming_rule());
  GroundPlaneEraser gpe;

  for_each(clouds.begin(), clouds.end(),
      [&trimmed_clouds, &filter, &gpe](PointCloud::Ptr &pc)
  {
    PointCloud::Ptr trimmed (new PointCloud);
    filter.setInputCloud(pc);
    filter.filter(*trimmed);

    gpe.remove_ground_plane(trimmed);

    trimmed_clouds.push_back(trimmed);
  });

  clouds.swap(trimmed_clouds);
}

/**
 * Removes all invalid (NaN, Inf) points from all the point clouds in the given
 * list.
 */
void sanitize_clouds(PointClouds& clouds) {
  PointClouds sanitized_clouds;
  vector<int> indices;

  for_each(clouds.begin(), clouds.end(), [&sanitized_clouds, &indices](PointCloud::Ptr &pc) {
    PointCloud::Ptr sanitized (new PointCloud);
    removeNaNFromPointCloud(*pc, *sanitized, indices);
    sanitized_clouds.push_back(sanitized);
  });

  clouds.swap(sanitized_clouds);
}

/**
 * Loads the point cloud contained into `filename` and pushes it to `store`.
 */
void load_point_cloud(const string& filename, PointClouds& store) {
  PointCloud::Ptr cloud (new PointCloud);

  if (loadPCDFile<Point>(filename, *cloud) == -1) {
    cerr << "Couldn't read the pcd file." << endl;
    exit(-1);
  }

  store.push_back(cloud);
}

/**
 * Loads all source point clouds.
 */
void load_point_clouds(PointClouds& views) {
  for (int k = 1; k <= views_no; k++) {
    load_point_cloud(filename_for_cloud(k), views);
  }
}

void start_ui(PointClouds& views) {
  PCLVisualizer viewer ("Point Cloud Viewer");
  viewer.initCameraParameters();

  ColorMaker c (views.size());
  int idx = 1;
  for_each(views.begin(), views.end(), [&viewer, &idx, &c](PointCloud::Ptr &pc) {
    stringstream ss;
    ss << "point_cloud_" << idx;
    viewer.addPointCloud<Point>(
      pc,
      CustomColor(pc, c.r(idx), c.g(idx), c.b(idx)),
      ss.str());
    idx++;
  });

  viewer.spin();
}

int main(int argc, char **argv) {
  src_idx = atoi(argv[2]); // TODO: Remove !!!!!
  tgt_idx = atoi(argv[1]); // TODO: Remove !!!!!

  PointClouds views;

  cout << "Loading point clouds..." << flush;
  load_point_clouds(views);
  cout << "done." << endl;

  cout << "Trimming raw clouds..." << flush;
  trim_clouds(views);
  cout << "done." << endl;

  cout << "Sanitizing clouds..." << flush;
  sanitize_clouds(views);
  cout << "done." << endl;

  cout << "Doing stuff..." << flush;
  DO_STUFF(views);
  cout << "done." << endl;

  cout << "Visualizing..." << endl;
  start_ui(views);

  cout << "Quitting." << endl;

  return 0;
}

//#include <vector>
//#include <string>
//#include <iostream>
//#include <algorithm>
//#include <cmath>
//
//#include <pcl/io/pcd_io.h>
//#include <pcl/point_types.h>
//#include <pcl/filters/filter.h>
//#include <pcl/filters/voxel_grid.h>
//#include <pcl/filters/conditional_removal.h>
//#include <pcl/visualization/pcl_visualizer.h>
//#include <pcl/features/normal_3d_omp.h>
//#include <pcl/features/fpfh_omp.h>
//#include <pcl/keypoints/sift_keypoint.h>
//#include <pcl/search/kdtree.h>
//#include <pcl/registration/correspondence_estimation.h>
//#include <pcl/registration/correspondence_rejection_sample_consensus.h>
//#include <pcl/registration/transformation_estimation_svd.h>
//#include <pcl/registration/ia_ransac.h>
//#include <pcl/registration/icp.h>
//
//using std::vector;
//using std::string;
//using std::stringstream;
//using pcl::io::loadPCDFile;
//using pcl::removeNaNFromPointCloud;
//using pcl::visualization::PCLVisualizer;
//using pcl::visualization::PointCloudColorHandlerRGBField;
//using pcl::ComparisonOps::GT;
//using pcl::ComparisonOps::LT;
//using pcl::Correspondences;
//using boost::make_shared;
//
//typedef pcl::PointXYZRGB Point;
//typedef pcl::FPFHSignature33 Feature;
//typedef pcl::PointCloud<Point> PointCloud;
//typedef pcl::PointCloud<Feature> Features;
//typedef pcl::PointCloud<pcl::Normal> Normals;
//typedef pcl::VoxelGrid<Point> VoxelGrid;
//typedef pcl::ConditionalRemoval<Point> RemovalFilter;
//typedef pcl::ConditionAnd<Point> RemovalCondition;
//typedef pcl::FieldComparison<Point> FieldComparison;
//typedef pcl::SIFTKeypoint<Point, Point> KeypointDetector;
//typedef pcl::search::KdTree<Point> KdTree;
//typedef pcl::NormalEstimationOMP<Point, pcl::Normal> NormalEstimator;
//typedef pcl::FPFHEstimationOMP<Point, pcl::Normal, Feature> FeatureEstimator;
//typedef pcl::registration::CorrespondenceEstimation<Feature, Feature> CorrespondenceEstimator;
//typedef pcl::registration::CorrespondenceRejectorSampleConsensus<Point> CorrespondenceRejector;
//typedef pcl::registration::TransformationEstimationSVD<Point, Point> TransformationEstimator;
//typedef Eigen::Matrix4f Matrix;
//typedef pcl::IterativeClosestPoint<Point, Point> IterativeClosestPoint;
//typedef pcl::SampleConsensusInitialAlignment<Point, Point, Feature> InitialAlignment;
//
//string filename_for_cloud(int k) {
//  stringstream ss;
//  ss << "../dataset/nao/" << k << ".pcd";
//  return ss.str();
//}
//
//void load_and_store_point_cloud(const string& filename, vector<ColorPointCloud::Ptr>& store) {
//  ColorPointCloud::Ptr cloud (new ColorPointCloud);
//
//  if (loadPCDFile<ColorPoint>(filename, *cloud) == -1) {
//    PCL_ERROR("Couldn't read the pcd file.\n");
//    exit(-1);
//  }
//
//  store.push_back(cloud);
//}
//
//void show_cloud_in_viewport(PCLVisualizer &viewer, const ColorPointCloud::Ptr &pc, int index) {
//  stringstream point_cloud_tag, label_tag, label;
//  point_cloud_tag << "source_" << index;
//  label_tag << "source_" << index << "_viewport_label";
//  label << "Source point cloud " << index;
//
//  float xmin, xmax, ymin, ymax;
//  xmin = 1.0 / 3 * ((index - 1) % 3);
//  xmax = xmin + 1.0 / 3;
//  ymin = 1.0 / 2 * ((index - 1) / 3);
//  ymax = ymin + 1.0 / 2;
//
//	viewer.createViewPort(xmin, ymin, xmax, ymax, index);
//	viewer.setBackgroundColor(0, 0, 0, index);
//	viewer.addCoordinateSystem(0.1, index);
//	viewer.addText(label.str(), 10, 10, label_tag.str(), index);
//	viewer.addPointCloud<ColorPoint>(
//      pc,
//      PointCloudColorHandlerRGBField<ColorPoint>(pc),
//      point_cloud_tag.str(),
//      index);
//}
//
//void sanitize_clouds(vector<ColorPointCloud::Ptr>& clouds) {
//  vector<ColorPointCloud::Ptr> sanitized_clouds;
//  vector<int> indices;
//  for_each(clouds.begin(), clouds.end(), [&sanitized_clouds, &indices](ColorPointCloud::Ptr &pc) {
//    ColorPointCloud::Ptr sanitized (new ColorPointCloud);
//    removeNaNFromPointCloud(*pc, *sanitized, indices);
//    sanitized_clouds.push_back(sanitized);
//  });
//
//  clouds.swap(sanitized_clouds);
//}
//
//void trim_clouds(vector<ColorPointCloud::Ptr>& clouds) {
//  vector<ColorPointCloud::Ptr> trimmed_clouds;
//
//  RemovalCondition::Ptr condition (new RemovalCondition);
//  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("z", GT, 1.1)));
//  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("z", LT, 1.7)));
//  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("x", GT, -0.7)));
//  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("x", LT, 0.5)));
//  RemovalFilter filter (condition);
//
//  for_each(clouds.begin(), clouds.end(), [&trimmed_clouds, &filter](ColorPointCloud::Ptr &pc) {
//    ColorPointCloud::Ptr trimmed (new ColorPointCloud);
//    filter.setInputCloud(pc);
//    filter.filter(*trimmed);
//    trimmed_clouds.push_back(trimmed);
//  });
//
//  clouds.swap(trimmed_clouds);
//}
//
//void downsample_clouds(vector<ColorPointCloud::Ptr>& clouds, float box_size) {
//  VoxelGrid sor;
//  sor.setLeafSize(box_size, box_size, box_size);
//
//  vector<ColorPointCloud::Ptr> downsampled_clouds;
//  for_each(clouds.begin(), clouds.end(), [&downsampled_clouds, &sor](ColorPointCloud::Ptr &pc) {
//    ColorPointCloud::Ptr downsampled (new ColorPointCloud);
//
//	  sor.setInputCloud(pc);
//	  sor.filter(*downsampled);
//
//    downsampled_clouds.push_back(downsampled);
//  });
//
//  clouds.swap(downsampled_clouds);
//}
//
//void show_transformed_point_clouds(const vector<ColorPointCloud::Ptr>& views) {
//  PCLVisualizer viewer ("PCL Viewer");
//	viewer.initCameraParameters();
//
//  int index = 1;
//  for_each(views.begin(), views.end(), [&index, &viewer](const ColorPointCloud::Ptr &pc) {
//    show_cloud_in_viewport(viewer, pc, index);
//    index++;
//  });
//
//  viewer.spin();
//}
//
//class MultiCloudRegistration {
//  public:
//
//  MultiCloudRegistration() {};
//
//  void register_clouds(vector<ColorPointCloud::Ptr>& clouds) {
//    ref_cloud = clouds.front();
//    ref_features = features_of(ref_cloud);
//
//    pre_aligned_clouds.push_back(ref_cloud);
//
//    for (int k = 1; k < clouds.size(); k++) {
//      pre_aligned_clouds.push_back(align_to_reference(clouds.at(k)));
//    }
//
//    clouds.swap(pre_aligned_clouds);
//  };
//
//  private:
//
//  ColorPointCloud::Ptr ref_cloud;
//  Features::Ptr ref_features;
//  vector<ColorPointCloud::Ptr> pre_aligned_clouds;
//
//  ColorPointCloud::Ptr align_to_reference(ColorPointCloud::Ptr pc) {
////    CorrespondenceEstimator est;
////    Correspondences correspondences, inliers;
////    CorrespondenceRejector rejector;
////    TransformationEstimator estimator;
////    Matrix transformation;
////    
//    auto pc_features = features_of(pc);
////
////    est.setInputSource(pc_features);
////    est.setInputTarget(ref_features);
////    est.determineCorrespondences(correspondences);
////
////    rejector.setInputSource(pc);
////    rejector.setInputTarget(ref_cloud);
////    rejector.setInputCorrespondences(make_shared<const Correspondences>(correspondences));
////    rejector.setInlierThreshold(0.05);
////    rejector.getCorrespondences(inliers);
////
////    estimator.estimateRigidTransformation(*pc, *ref_cloud, inliers, transformation);
////
//    ColorPointCloud::Ptr pre_aligned_pc (new ColorPointCloud);
////    ColorPointCloud::Ptr aligned_pc (new ColorPointCloud);
////    transformPointCloud(*pc, *pre_aligned_pc, transformation);
//
////    IterativeClosestPoint icp;
////    icp.setMaxCorrespondenceDistance(0.50);
////    icp.setRANSACOutlierRejectionThreshold(0.05);
////    icp.setTransformationEpsilon(0.000001);
////    icp.setMaximumIterations(600);
////    icp.setInputSource(pre_aligned_pc);
////    icp.setInputTarget(ref_cloud);
////    icp.align(*aligned_pc);
////
////    cout << (icp.hasConverged() ? "Alignment succeeded!" : "Alignment failed.") << endl;
////    cout << "Alignment score: " << icp.getFitnessScore() << endl;
//
//    InitialAlignment ia;
//    ia.setMinSampleDistance(0.01);
//    ia.setMaxCorrespondenceDistance(0.5);
//    //ia.setMaximumIterations();
//    ia.setInputSource(pc);
//    ia.setSourceFeatures(pc_features);
//    ia.setInputTarget(ref_cloud);
//    ia.setTargetFeatures(ref_features);
//    ia.align(*pre_aligned_pc);
//
////    PCLVisualizer viewer ("Correspondence Viewer");
////    viewer.initCameraParameters();
////    viewer.addPointCloud<ColorPoint>(
////      pc,
////      PointCloudColorHandlerRGBField<ColorPoint>(pc),
////      "aligned");
////    viewer.addPointCloud<ColorPoint>(
////      ref_cloud,
////      PointCloudColorHandlerRGBField<ColorPoint>(ref_cloud),
////      "original");
////    viewer.addCorrespondences<ColorPoint>(pc, ref_cloud, inliers);
////    viewer.spin();
//
//    return pre_aligned_pc;
//  };
//
//  Features::Ptr features_of(ColorPointCloud::Ptr pc) {
//    Normals::Ptr normals (new Normals);
//    ColorPointCloud::Ptr keypoints (new ColorPointCloud);
//    Features::Ptr fpfhs (new Features);
//
//    KeypointDetector detector;
//    detector.setSearchMethod(KdTree::Ptr(new KdTree));
//    detector.setScales(0.002, 5, 3);
//    detector.setMinimumContrast(0);
//    detector.setInputCloud(pc);
//    detector.compute(*keypoints);
//
////    PCLVisualizer viewer ("Keypoint Viewer");
////    viewer.initCameraParameters();
////    viewer.setBackgroundColor(255, 255, 255);
////    viewer.addPointCloud<ColorPoint>(
////      keypoints,
////      PointCloudColorHandlerRGBField<ColorPoint>(keypoints),
////      "keypoints");
////    viewer.spin();
//
//	  NormalEstimator ne;
//	  ne.setSearchMethod(KdTree::Ptr(new KdTree));
//	  ne.setNumberOfThreads(2);
//	  ne.setRadiusSearch(0.05);
//	  ne.setInputCloud(pc);
//	  ne.compute(*normals);
//
//	  FeatureEstimator fpfh;
//    fpfh.setSearchSurface(pc);
//	  fpfh.setInputCloud(keypoints);
//	  fpfh.setInputNormals(normals);
//	  fpfh.setRadiusSearch(0.1);
//	  fpfh.setNumberOfThreads(2);
//	  fpfh.compute(*fpfhs);
//
//    return fpfhs;
//  }
//};
//
//ColorPointCloud::Ptr merge_point_clouds(vector<ColorPointCloud::Ptr> &clouds) {
//  ColorPointCloud::Ptr merged (new ColorPointCloud);
//
//  int k = 0;
//  for_each(clouds.begin(), clouds.end(), [&k, &merged](ColorPointCloud::Ptr &pc) {
//    // pick a color among the spectrum, based on k
//    for_each(pc->points.begin(), pc->points.end(), [&merged](ColorPoint &pt) {
//      // set point color to the new one
//      merged->points.push_back(pt);
//    });
//    k++;
//  });
//
//  return merged;
//}
//
//void show_merged_point_cloud(ColorPointCloud::Ptr &merged) {
//  PCLVisualizer viewer ("Merged clouds");
//	viewer.initCameraParameters();
//
//	viewer.setBackgroundColor(0, 0, 0);
//	viewer.addCoordinateSystem(0.1);
//	viewer.addPointCloud<ColorPoint>(
//      merged,
//      PointCloudColorHandlerRGBField<ColorPoint>(merged),
//      "merged_point_cloud");
//
//  viewer.spin();
//}
//
//int main(int argc, char** argv) {
//  int views_no = 2;
//  vector<ColorPointCloud::Ptr> views;
//
//  for (int k = 1; k <= views_no; k++) {
//    load_and_store_point_cloud(filename_for_cloud(k), views);
//  }
//
//  PCL_INFO("Loaded raw point clouds\n");
//
//  PCL_INFO("Preparing data\n");
//  sanitize_clouds(views);
//  trim_clouds(views);
//  //downsample_clouds(views, 0.05f);
//
//  PCL_INFO("Processing data\n");
//  MultiCloudRegistration reg;
//  reg.register_clouds(views);
//
//  PCL_INFO("Visualizing data\n");
//
//  ColorPointCloud::Ptr merged = merge_point_clouds(views);
//
//  show_transformed_point_clouds(views);
//  show_merged_point_cloud(merged);
//
//  return 0;
//}
