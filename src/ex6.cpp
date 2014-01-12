/**
 * PCL Lab - Ex #6
 *
 * Performs multi-cloud registration.
 *
 * Author: Stefano Zanella
 * Date: 10/01/2014
 */

#include <vector>
#include <string>
#include <iostream>

#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/conditional_removal.h>
#include <pcl/visualization/pcl_visualizer.h>
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
using pcl::transformPointCloud;
using pcl::removeNaNFromPointCloud;
using pcl::Correspondences;

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
typedef Eigen::Matrix4f Matrix;
typedef pcl::SIFTKeypoint<Point, Point> Keypoint;
typedef pcl::search::KdTree<Point> KdTree;
typedef pcl::PointCloud<pcl::Normal> Normals;
typedef pcl::NormalEstimationOMP<Point, pcl::Normal> NormalEstimation;
typedef pcl::PointCloud<pcl::FPFHSignature33> Features;
typedef pcl::FPFHEstimationOMP<Point, pcl::Normal, pcl::FPFHSignature33> FeatureEstimation;
typedef pcl::registration::CorrespondenceEstimation<pcl::FPFHSignature33, pcl::FPFHSignature33> CorrespondenceEstimation;
typedef pcl::registration::CorrespondenceRejectorSampleConsensus<Point> CorrespondenceRejector;
typedef pcl::registration::TransformationEstimationSVD<Point, Point> TransformationEstimation;
typedef pcl::IterativeClosestPoint<Point, Point> IterativeClosestPoint;

/**
 * Performs multi-view robot registration (e.g. registration of multiple point clouds
 * representing different views of the robot).
 */
class RobotRegistration {
  public :
    void align(PointClouds& views) {
      sanitize_clouds(views);
    
      Matrix transformation;
      for (int k = 0; k < views.size() - 1; k += 2) {
        transformation = pairwise_align(views.at(k+1), views.at(k));
        transformPointCloud(*views.at(k+1), *views.at(k+1), transformation);
      }
    
      for (int k = 1; k < views.size() - 2; k += 2) {
        transformation = pairwise_align(views.at(k+1), views.at(k));
        transformPointCloud(*views.at(k+1), *views.at(k+1), transformation);
        transformPointCloud(*views.at(k+2), *views.at(k+2), transformation);
      }
    }

  private :
    Matrix pairwise_align(PointCloud::Ptr& src, PointCloud::Ptr& tgt) {
      // Keypoints extraction
      PointCloud::Ptr src_keypoints (new PointCloud);
      PointCloud::Ptr tgt_keypoints (new PointCloud);
    
      Keypoint sift;
      sift.setSearchMethod(KdTree::Ptr(new KdTree));
      sift.setScales(0.01f, 3, 2);
      sift.setMinimumContrast(0);
    
      sift.setInputCloud(src);
      sift.compute(*src_keypoints);
    
      sift.setInputCloud(tgt);
      sift.compute(*tgt_keypoints);
    
      // Normal estimation
      Normals::Ptr src_normals (new Normals);
      Normals::Ptr tgt_normals (new Normals);
    
      NormalEstimation normal_estimation;
      normal_estimation.setSearchMethod(KdTree::Ptr(new KdTree));
      normal_estimation.setRadiusSearch(0.1);
    
      normal_estimation.setInputCloud(src);
      normal_estimation.compute(*src_normals);
    
      normal_estimation.setInputCloud(tgt);
      normal_estimation.compute(*tgt_normals);
    
      // Feature estimation
      Features::Ptr src_features (new Features);
      Features::Ptr tgt_features (new Features);
    
      FeatureEstimation feature_estimation;
      feature_estimation.setSearchMethod(KdTree::Ptr(new KdTree));
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
      CorrespondenceEstimation est;
      Correspondences correspondences, inliers;
      
      est.setInputSource(src_features);
      est.setInputTarget(tgt_features);
      est.determineCorrespondences(correspondences);
    
      // Correspondence rejection
      CorrespondenceRejector rejector;
      rejector.setInputSource(src_keypoints);
      rejector.setInputTarget(tgt_keypoints);
      rejector.setInputCorrespondences(boost::make_shared<const Correspondences>(correspondences));
      rejector.setInlierThreshold(0.05);
      rejector.setMaxIterations(5000);
      rejector.getCorrespondences(inliers);
    
      // Transformation estimation
      TransformationEstimation estimator;
      Matrix transformation;
      estimator.estimateRigidTransformation(*src_keypoints, *tgt_keypoints, inliers, transformation);
    
      PointCloud::Ptr pre_aligned (new PointCloud);
      PointCloud::Ptr aligned (new PointCloud);
      transformPointCloud(*src, *pre_aligned, transformation);
      
      IterativeClosestPoint icp;
      icp.setMaxCorrespondenceDistance(0.50);
      icp.setRANSACOutlierRejectionThreshold(0.05);
      icp.setTransformationEpsilon(0.000001);
      icp.setMaximumIterations(600);
      icp.setInputSource(pre_aligned);
      icp.setInputTarget(tgt);
      icp.align(*aligned);
    
      cout << (icp.hasConverged() ? "Alignment succeeded!" : "Alignment failed.") << endl;
      cout << "Alignment score: " << icp.getFitnessScore() << endl;
    
      return icp.getFinalTransformation() * transformation;
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
};

/**
 * Extracts the robot from a series of point clouds by excluding points outside
 * of a given box and by removing the ground plane using SAC segmentation.
 */
class RobotExtractor {
  public :
    void extract_from(PointClouds& clouds) {
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

  private :
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
};

/**
 * Handles loading of points clouds representing different views of the same
 * object from a given directory.
 *
 * Point cloud files are expected to be contained into a single directory,
 * following the filename format: <basename><index><suffix>
 * where <index> is a char representing an integral index: 1, 2, ...
 */
class PCDLoader {
  public:
    PCDLoader(const string& _basename = "./", const string& _extension = ".pcd") :
      basename(_basename),
      extension(_extension)
    {}

    /**
     * Loads the given number of source point clouds and stores them into the
     * passed array.
     */
    void load(const int count, PointClouds& views) {
      for (int k = 1; k <= count; k++) {
        load_point_cloud(filename_for_cloud(k), views);
      }
    }

  private :
    string basename, extension;

    /**
     * Returns a proper path to a PCD file given an index.
     */
    string filename_for_cloud(int k) {
      stringstream ss;
      ss << basename << k << extension;
      return ss.str();
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
};

/**
 * Main UI for the application. Manages two separate windows showing the
 * overlay of given views both with original colors and with a different color
 * per point cloud to aid visual analysis.
 */
class UI {
  public :
    UI(const PointClouds& _views) :
      color_overlay_viewer ("Color overlay viewer"),
      result_viewer ("Original registration viewer"),
      views (_views)
    {
      color_overlay_viewer.initCameraParameters();
      result_viewer.initCameraParameters();
    }

    void start() {
      ColorMaker c (views.size());
      int idx = 1;
      for_each(views.begin(), views.end(),
        [this, &idx, &c](PointCloud::Ptr &pc) {
        stringstream ss;
        ss << "point_cloud_" << idx;
    
        this->result_viewer.addPointCloud<Point>(
          pc,
          RGBColor(pc),
          ss.str());
    
        this->color_overlay_viewer.addPointCloud<Point>(
          pc,
          CustomColor(pc, c.r(idx), c.g(idx), c.b(idx)),
          ss.str());
    
        idx++;
      });
    
      color_overlay_viewer.spin();
      result_viewer.spin();
    }

  private :
    PCLVisualizer color_overlay_viewer, result_viewer;
    PointClouds views;

    /**
     * Simple class that implements pseudo-random RGB color shifting parameterized
     * on an integral index.
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
};

int main(int argc, char **argv) {
  PointClouds views;

  cout << "Loading point clouds..." << flush;
  PCDLoader("../dataset/nao/", ".pcd").load(6, views);
  cout << "done." << endl;

  cout << "Extracting robots from raw clouds..." << flush;
  RobotExtractor().extract_from(views);
  cout << "done." << endl;

  cout << "Aligning views..." << flush;
  RobotRegistration().align(views);
  cout << "done." << endl;

  cout << "Visualizing..." << endl;
  UI(views).start();

  cout << "Quitting." << endl;
  return 0;
}
