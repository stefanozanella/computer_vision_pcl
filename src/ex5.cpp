#include <iostream>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/registration/icp.h>
#include <pcl/visualization/pcl_visualizer.h>

using std::cout;
using std::endl;
using std::vector;

using pcl::PointCloud;
using pcl::PointXYZ;
using pcl::io::loadPCDFile;
using pcl::removeNaNFromPointCloud;
using pcl::IterativeClosestPoint;
using pcl::visualization::PCLVisualizer;
using pcl::visualization::PointCloudColorHandlerCustom;

int main(int argc, char** argv) {
  PointCloud<PointXYZ>::Ptr src_cloud_1 (new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr src_cloud_2 (new PointCloud<PointXYZ>);
  PointCloud<PointXYZ> target; 

  if (loadPCDFile<PointXYZ>("../dataset/capture0001.pcd", *src_cloud_1) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    return -1;
  }
  vector<int> indices;
  removeNaNFromPointCloud(*src_cloud_1, *src_cloud_1, indices);

  cout << "Loaded point cloud: " << src_cloud_1->width << " x " << src_cloud_1->height << endl;

  if (loadPCDFile<PointXYZ>("../dataset/capture0002.pcd", *src_cloud_2) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    return -1;
  }
  removeNaNFromPointCloud(*src_cloud_2, *src_cloud_2, indices);

  cout << "Loaded point cloud: " << src_cloud_2->width << " x " << src_cloud_2->height << endl;

  IterativeClosestPoint<PointXYZ, PointXYZ> icp;
  icp.setInputSource(src_cloud_1);
  icp.setInputTarget(src_cloud_2);
  icp.align(target);

  cout << (icp.hasConverged() ? "Alignment succeeded!" : "Alignment failed.") << endl;
  cout << "Alignment score: " << icp.getFitnessScore() << endl;

  PCLVisualizer viewer ("PCL Viewer");
	viewer.initCameraParameters();

	int source_1_viewport, source_2_viewport, target_viewport;

	viewer.createViewPort(0.0, 0.5, 0.5, 1.0, source_1_viewport);
	viewer.setBackgroundColor(0, 0, 0, source_1_viewport);
	viewer.addCoordinateSystem(0.1, source_1_viewport);
	viewer.addText("Source point cloud 1", 10, 10, "source_1_viewport_label", source_1_viewport);
	viewer.addPointCloud<PointXYZ>(
      src_cloud_1,
      PointCloudColorHandlerCustom<PointXYZ>(src_cloud_1, 0, 255, 0),
      "source_1",
      source_1_viewport);

	viewer.createViewPort(0.0, 0.0, 0.5, 0.5, source_2_viewport);
	viewer.setBackgroundColor(0, 0, 0, source_2_viewport);
	viewer.addCoordinateSystem(0.1, source_2_viewport);
	viewer.addText("Source point cloud 2", 10, 10, "source_2_viewport_label", source_2_viewport);
	viewer.addPointCloud<PointXYZ>(
      src_cloud_2,
      PointCloudColorHandlerCustom<PointXYZ>(src_cloud_2, 255, 0, 0),
      "source_2",
      source_2_viewport);

	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, target_viewport);
	viewer.setBackgroundColor(0, 0, 0, target_viewport);
	viewer.addCoordinateSystem(0.1, target_viewport);
	viewer.addText("Aligned clouds", 10, 10, "target_viewport_label", target_viewport);
  PointCloud<PointXYZ>::Ptr target_ptr (&target);
	viewer.addPointCloud<PointXYZ>(
      target_ptr,
      PointCloudColorHandlerCustom<PointXYZ>(target_ptr, 255, 255, 255),
      "target",
      target_viewport);

  cout << "Visualizing..." << endl;

  viewer.spin();

  return 0;
}
