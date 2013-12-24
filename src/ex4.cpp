#include <iostream>
#include <sstream>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/histogram_visualizer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/features/fpfh_omp.h>
#include <pcl/filters/filter.h>
#include <pcl/keypoints/sift_keypoint.h>
#include <pcl/search/kdtree.h>

using std::cout;
using std::endl;
using std::vector;
using std::stringstream;

using pcl::PointCloud;
using pcl::PointXYZ;
using pcl::PointXYZRGB;
using pcl::PointWithScale;
using pcl::Normal;
using pcl::io::loadPCDFile;
using pcl::removeNaNFromPointCloud;
using pcl::NormalEstimationOMP;
using pcl::SIFTKeypoint;
using pcl::search::KdTree;
using pcl::visualization::PCLVisualizer;
using pcl::visualization::PointPickingEvent;
using pcl::visualization::PCLHistogramVisualizer;
using pcl::visualization::PointCloudColorHandlerRGBField;
using pcl::visualization::PointCloudColorHandlerCustom;
using pcl::FPFHSignature33;
using pcl::FPFHEstimationOMP;

struct CallbackArgs {
	PCLHistogramVisualizer histViewer;
	PointCloud<FPFHSignature33>::Ptr fpfhs;

  CallbackArgs(PointCloud<FPFHSignature33>::Ptr features) :
    histViewer(PCLHistogramVisualizer()),
    fpfhs(features)
  {}
};

void pp_callback (const PointPickingEvent& event, void* args) {
  if (event.getPointIndex () == -1) {
    cout << "Sorry, couldn't identify selected point." << endl;
    return;
  }

	CallbackArgs* data = (CallbackArgs*) args;
  stringstream windowName;
  windowName << "FPFH for point " << event.getPointIndex();
  data->histViewer.addFeatureHistogram(
      *(data->fpfhs),
      "fpfh",
      event.getPointIndex(),
      windowName.str(), 
      640, 200);
}

int main(int argc, char** argv) {
  PointCloud<PointXYZRGB>::Ptr cloud_in (new PointCloud<PointXYZRGB>);
	PointCloud<Normal>::Ptr normals (new PointCloud<Normal>);
  PointCloud<PointXYZRGB>::Ptr keypoints (new PointCloud<PointXYZRGB>);
	PointCloud<FPFHSignature33>::Ptr fpfhs (new PointCloud<FPFHSignature33>);

  if (loadPCDFile<PointXYZRGB>("../dataset/minimouse1_segmented.pcd", *cloud_in) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    return -1;
  }

  cout << "Loaded point cloud: " << cloud_in->width << " x " << cloud_in->height << endl;

  vector<int> indices;
  removeNaNFromPointCloud(*cloud_in, *cloud_in, indices);

  SIFTKeypoint<PointXYZRGB, PointXYZRGB> detector;
  detector.setSearchMethod(
      KdTree<PointXYZRGB>::Ptr(new KdTree<PointXYZRGB>));
  detector.setScales(0.01, 3, 2);
  detector.setMinimumContrast(0);
  detector.setInputCloud(cloud_in);
  detector.compute(*keypoints);

	NormalEstimationOMP<PointXYZRGB, Normal> ne;
	ne.setSearchMethod(KdTree<PointXYZRGB>::Ptr(new KdTree<PointXYZRGB>));
	ne.setNumberOfThreads(2);
	ne.setRadiusSearch(0.03);
	ne.setInputCloud(cloud_in);

	cout << "Computing normals...please wait...";
	ne.compute(*normals);
	cout << "done." << endl;

	FPFHEstimationOMP<PointXYZRGB, Normal, FPFHSignature33> fpfh;
  fpfh.setSearchSurface(cloud_in);
	fpfh.setInputCloud(keypoints);
	fpfh.setInputNormals(normals);
	fpfh.setRadiusSearch(0.03);
	fpfh.setNumberOfThreads(2);

	cout << "Computing features...please wait...";
	fpfh.compute(*fpfhs);
	cout << "done." << endl;

	int normalsVisualizationStep = 100;
	float normalsScale = 0.02;

  PCLVisualizer viewer ("PCL Viewer");
	viewer.initCameraParameters();

	int left_viewport (0);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, left_viewport);
	viewer.setBackgroundColor(0, 0, 0.5, left_viewport);
	viewer.addCoordinateSystem(0.1, left_viewport);
	viewer.addText("Original point cloud", 10, 10, "left_viewport_label", left_viewport);
	viewer.addPointCloud<PointXYZRGB>(
      cloud_in,
      PointCloudColorHandlerRGBField<PointXYZRGB>(cloud_in),
      "input_cloud_left",
      left_viewport);
	viewer.addPointCloudNormals<PointXYZRGB, Normal>(cloud_in, normals, normalsVisualizationStep, normalsScale, "normals", left_viewport);

	int right_viewport (0);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, right_viewport);
	viewer.setBackgroundColor(0, 0, 0, right_viewport);
	viewer.addCoordinateSystem(0.1, right_viewport);
	viewer.addText("Keypoints", 10, 10, "right_viewport_label", right_viewport);
  viewer.addPointCloud(
      keypoints,
      PointCloudColorHandlerCustom<PointXYZRGB>(keypoints, 255, 255, 255),
      "keypoints_cloud",
      right_viewport);
	viewer.registerPointPickingCallback(pp_callback, (void *) new CallbackArgs(fpfhs));

  cout << "Visualizing..." << endl;

  viewer.spin();

  return 0;
}
