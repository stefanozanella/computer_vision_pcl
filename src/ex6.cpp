#include <vector>
#include <string>
#include <iostream>
#include <algorithm>
#include <cmath>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/search/kdtree.h>
#include <pcl/registration/correspondence_estimation.h>
#include <pcl/registration/correspondence_rejection_sample_consensus.h>
#include <pcl/registration/transformation_estimation_svd.h>
#include <pcl/registration/icp.h>

using std::vector;
using std::string;
using std::stringstream;
using pcl::io::loadPCDFile;
using pcl::removeNaNFromPointCloud;
using pcl::visualization::PCLVisualizer;
using pcl::visualization::PointCloudColorHandlerRGBField;
using pcl::ComparisonOps::GT;
using pcl::ComparisonOps::LT;
using pcl::Correspondences;
using boost::make_shared;

typedef pcl::PointCloud<pcl::PointXYZRGB> ColorPointCloud;
typedef pcl::PointXYZRGB ColorPoint;
typedef pcl::PointCloud<pcl::FPFHSignature33> Features;
typedef pcl::PointCloud<pcl::Normal> Normals;
typedef pcl::VoxelGrid<pcl::PointXYZRGB> VoxelGrid;
typedef pcl::ConditionalRemoval<pcl::PointXYZRGB> RemovalFilter;
typedef pcl::ConditionAnd<pcl::PointXYZRGB> RemovalCondition;
typedef pcl::FieldComparison<pcl::PointXYZRGB> FieldComparison;
typedef pcl::SIFTKeypoint<pcl::PointXYZRGB, pcl::PointXYZRGB> KeypointDetector;
typedef pcl::search::KdTree<pcl::PointXYZRGB> KdTree;
typedef pcl::NormalEstimationOMP<pcl::PointXYZRGB, pcl::Normal> NormalEstimator;
typedef pcl::FPFHEstimationOMP<pcl::PointXYZRGB, pcl::Normal, pcl::FPFHSignature33> FeatureEstimator;
typedef pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> CorrespondenceEstimator;
typedef pcl::registration::CorrespondenceRejectorSampleConsensus<pcl::PointXYZRGB> CorrespondenceRejector;
typedef pcl::registration::TransformationEstimationSVD<pcl::PointXYZRGB, pcl::PointXYZRGB> TransformationEstimator;
typedef Eigen::Matrix4f Matrix;
typedef pcl::IterativeClosestPoint<pcl::PointXYZRGB, pcl::PointXYZRGB> IterativeClosestPoint;

string filename_for_cloud(int k) {
  stringstream ss;
  ss << "../dataset/nao/" << k << ".pcd";
  return ss.str();
}

void load_and_store_point_cloud(const string& filename, vector<ColorPointCloud::Ptr>& store) {
  ColorPointCloud::Ptr cloud (new ColorPointCloud);

  if (loadPCDFile<ColorPoint>(filename, *cloud) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    exit(-1);
  }

  store.push_back(cloud);
}

void show_cloud_in_viewport(PCLVisualizer &viewer, const ColorPointCloud::Ptr &pc, int index) {
  stringstream point_cloud_tag, label_tag, label;
  point_cloud_tag << "source_" << index;
  label_tag << "source_" << index << "_viewport_label";
  label << "Source point cloud " << index;

  float xmin, xmax, ymin, ymax;
  xmin = 1.0 / 3 * ((index - 1) % 3);
  xmax = xmin + 1.0 / 3;
  ymin = 1.0 / 2 * ((index - 1) / 3);
  ymax = ymin + 1.0 / 2;

	viewer.createViewPort(xmin, ymin, xmax, ymax, index);
	viewer.setBackgroundColor(0, 0, 0, index);
	viewer.addCoordinateSystem(0.1, index);
	viewer.addText(label.str(), 10, 10, label_tag.str(), index);
	viewer.addPointCloud<ColorPoint>(
      pc,
      PointCloudColorHandlerRGBField<ColorPoint>(pc),
      point_cloud_tag.str(),
      index);
}

void sanitize_clouds(vector<ColorPointCloud::Ptr>& clouds) {
  vector<ColorPointCloud::Ptr> sanitized_clouds;
  vector<int> indices;
  for_each(clouds.begin(), clouds.end(), [&sanitized_clouds, &indices](ColorPointCloud::Ptr &pc) {
    ColorPointCloud::Ptr sanitized (new ColorPointCloud);
    removeNaNFromPointCloud(*pc, *sanitized, indices);
    sanitized_clouds.push_back(sanitized);
  });

  clouds.swap(sanitized_clouds);
}

void trim_clouds(vector<ColorPointCloud::Ptr>& clouds) {
  vector<ColorPointCloud::Ptr> trimmed_clouds;

  RemovalCondition::Ptr condition (new RemovalCondition);
  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("z", GT, 1.1)));
  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("z", LT, 1.7)));
  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("x", GT, -0.7)));
  condition->addComparison(FieldComparison::ConstPtr(new FieldComparison("x", LT, 0.5)));
  RemovalFilter filter (condition);

  for_each(clouds.begin(), clouds.end(), [&trimmed_clouds, &filter](ColorPointCloud::Ptr &pc) {
    ColorPointCloud::Ptr trimmed (new ColorPointCloud);
    filter.setInputCloud(pc);
    filter.filter(*trimmed);
    trimmed_clouds.push_back(trimmed);
  });

  clouds.swap(trimmed_clouds);
}

void downsample_clouds(vector<ColorPointCloud::Ptr>& clouds, float box_size) {
  VoxelGrid sor;
  sor.setLeafSize(box_size, box_size, box_size);

  vector<ColorPointCloud::Ptr> downsampled_clouds;
  for_each(clouds.begin(), clouds.end(), [&downsampled_clouds, &sor](ColorPointCloud::Ptr &pc) {
    ColorPointCloud::Ptr downsampled (new ColorPointCloud);

	  sor.setInputCloud(pc);
	  sor.filter(*downsampled);

    downsampled_clouds.push_back(downsampled);
  });

  clouds.swap(downsampled_clouds);
}

void show_transformed_point_clouds(const vector<ColorPointCloud::Ptr>& views) {
  PCLVisualizer viewer ("PCL Viewer");
	viewer.initCameraParameters();

  int index = 1;
  for_each(views.begin(), views.end(), [&index, &viewer](const ColorPointCloud::Ptr &pc) {
    show_cloud_in_viewport(viewer, pc, index);
    index++;
  });

  viewer.spin();
}

class MultiCloudRegistration {
  public:

  MultiCloudRegistration() {};

  void register_clouds(vector<ColorPointCloud::Ptr>& clouds) {
    ref_cloud = clouds.front();
    ref_features = features_of(ref_cloud);

    pre_aligned_clouds.push_back(ref_cloud);

    for (int k = 1; k < clouds.size(); k++) {
      pre_aligned_clouds.push_back(align_to_reference(clouds.at(k)));
    }

    clouds.swap(pre_aligned_clouds);
  };

  private:

  ColorPointCloud::Ptr ref_cloud;
  Features::Ptr ref_features;
  vector<ColorPointCloud::Ptr> pre_aligned_clouds;

  ColorPointCloud::Ptr align_to_reference(ColorPointCloud::Ptr pc) {
    CorrespondenceEstimator est;
    Correspondences correspondences, inliers;
    CorrespondenceRejector rejector;
    TransformationEstimator estimator;
    Matrix transformation;
    
    auto pc_features = features_of(pc);

    est.setInputSource(pc_features);
    est.setInputTarget(ref_features);
    est.determineCorrespondences(correspondences);

    rejector.setInputSource(pc);
    rejector.setInputTarget(ref_cloud);
    rejector.setInputCorrespondences(make_shared<const Correspondences>(correspondences));
    rejector.setInlierThreshold(0.05);
    rejector.getCorrespondences(inliers);

    estimator.estimateRigidTransformation(*pc, *ref_cloud, inliers, transformation);

    ColorPointCloud::Ptr pre_aligned_pc (new ColorPointCloud);
    ColorPointCloud::Ptr aligned_pc (new ColorPointCloud);
    transformPointCloud(*pc, *pre_aligned_pc, transformation);

//    IterativeClosestPoint icp;
//    icp.setMaxCorrespondenceDistance(0.50);
//    icp.setRANSACOutlierRejectionThreshold(0.05);
//    icp.setTransformationEpsilon(0.000001);
//    icp.setMaximumIterations(600);
//    icp.setInputSource(pre_aligned_pc);
//    icp.setInputTarget(ref_cloud);
//    icp.align(*aligned_pc);
//
//    cout << (icp.hasConverged() ? "Alignment succeeded!" : "Alignment failed.") << endl;
//    cout << "Alignment score: " << icp.getFitnessScore() << endl;

    PCLVisualizer viewer ("Correspondence Viewer");
    viewer.initCameraParameters();
    viewer.addPointCloud<ColorPoint>(
      pc,
      PointCloudColorHandlerRGBField<ColorPoint>(pc),
      "aligned");
    viewer.addPointCloud<ColorPoint>(
      ref_cloud,
      PointCloudColorHandlerRGBField<ColorPoint>(ref_cloud),
      "original");
    viewer.addCorrespondences<ColorPoint>(pc, ref_cloud, inliers);
    viewer.spin();

    return pre_aligned_pc;
  };

  Features::Ptr features_of(ColorPointCloud::Ptr pc) {
    Normals::Ptr normals (new Normals);
    ColorPointCloud::Ptr keypoints (new ColorPointCloud);
    Features::Ptr fpfhs (new Features);

    KeypointDetector detector;
    detector.setSearchMethod(KdTree::Ptr(new KdTree));
    detector.setScales(0.002, 5, 3);
    detector.setMinimumContrast(0);
    detector.setInputCloud(pc);
    detector.compute(*keypoints);

//    PCLVisualizer viewer ("Keypoint Viewer");
//    viewer.initCameraParameters();
//    viewer.setBackgroundColor(255, 255, 255);
//    viewer.addPointCloud<ColorPoint>(
//      keypoints,
//      PointCloudColorHandlerRGBField<ColorPoint>(keypoints),
//      "keypoints");
//    viewer.spin();

	  NormalEstimator ne;
	  ne.setSearchMethod(KdTree::Ptr(new KdTree));
	  ne.setNumberOfThreads(2);
	  ne.setRadiusSearch(0.05);
	  ne.setInputCloud(pc);
	  ne.compute(*normals);

	  FeatureEstimator fpfh;
    fpfh.setSearchSurface(pc);
	  fpfh.setInputCloud(keypoints);
	  fpfh.setInputNormals(normals);
	  fpfh.setRadiusSearch(0.1);
	  fpfh.setNumberOfThreads(2);
	  fpfh.compute(*fpfhs);

    return fpfhs;
  }
};

ColorPointCloud::Ptr merge_point_clouds(vector<ColorPointCloud::Ptr> &clouds) {
  ColorPointCloud::Ptr merged (new ColorPointCloud);

  int k = 0;
  for_each(clouds.begin(), clouds.end(), [&k, &merged](ColorPointCloud::Ptr &pc) {
    // pick a color among the spectrum, based on k
    for_each(pc->points.begin(), pc->points.end(), [&merged](ColorPoint &pt) {
      // set point color to the new one
      merged->points.push_back(pt);
    });
    k++;
  });

  return merged;
}

void show_merged_point_cloud(ColorPointCloud::Ptr &merged) {
  PCLVisualizer viewer ("Merged clouds");
	viewer.initCameraParameters();

	viewer.setBackgroundColor(0, 0, 0);
	viewer.addCoordinateSystem(0.1);
	viewer.addPointCloud<ColorPoint>(
      merged,
      PointCloudColorHandlerRGBField<ColorPoint>(merged),
      "merged_point_cloud");

  viewer.spin();
}

int main(int argc, char** argv) {
  int views_no = 2;
  vector<ColorPointCloud::Ptr> views;

  for (int k = 1; k <= views_no; k++) {
    load_and_store_point_cloud(filename_for_cloud(k), views);
  }

  PCL_INFO("Loaded raw point clouds\n");

  PCL_INFO("Preparing data\n");
  sanitize_clouds(views);
  trim_clouds(views);
  //downsample_clouds(views, 0.05f);

  PCL_INFO("Processing data\n");
  MultiCloudRegistration reg;
  reg.register_clouds(views);

  PCL_INFO("Visualizing data\n");

  ColorPointCloud::Ptr merged = merge_point_clouds(views);

  show_transformed_point_clouds(views);
  show_merged_point_cloud(merged);

  return 0;
}
