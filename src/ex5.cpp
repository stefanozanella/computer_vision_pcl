/**
 * PCL Lab - Ex #5
 *
 * Register two pre-aligned point clouds using ICP algorithm.
 *
 * Author: Stefano Zanella
 * Date: 08/01/2014
 */

#include <iostream>
#include <algorithm>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/visualization/pcl_visualizer.h>

using std::cout;
using std::endl;
using std::for_each;
using std::vector;

using pcl::PointCloud;
using pcl::PointXYZ;
using pcl::PointXYZRGB;
using pcl::io::loadPCDFile;
using pcl::removeNaNFromPointCloud;
using pcl::IterativeClosestPointNonLinear;
using pcl::IterativeClosestPoint;
using pcl::visualization::PCLVisualizer;
using pcl::visualization::PointCloudColorHandlerCustom;
using pcl::VoxelGrid;
using pcl::transformPointCloud;

int main(int argc, char** argv) {
  PointCloud<PointXYZ>::Ptr src_cloud (new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr tgt_cloud (new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr src_cloud_ds (new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr tgt_cloud_ds (new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr target (new PointCloud<PointXYZ>);

  if (loadPCDFile<PointXYZ>("../dataset/capture0001.pcd", *src_cloud) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    return -1;
  }
  cout << "Loaded point cloud: " << src_cloud->width << " x " << src_cloud->height << endl;

  if (loadPCDFile<PointXYZ>("../dataset/capture0002.pcd", *tgt_cloud) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    return -1;
  }
  cout << "Loaded point cloud: " << tgt_cloud->width << " x " << tgt_cloud->height << endl;

  vector<int> indices;
  removeNaNFromPointCloud(*src_cloud, *src_cloud, indices);
  removeNaNFromPointCloud(*tgt_cloud, *tgt_cloud, indices);

  VoxelGrid<PointXYZ> grid;
  grid.setLeafSize(0.05, 0.05, 0.05);
  grid.setInputCloud(src_cloud);
  grid.filter(*src_cloud_ds);
  grid.setInputCloud(tgt_cloud);
  grid.filter(*tgt_cloud_ds);

  // Non-linear, manually-iterative version
  IterativeClosestPointNonLinear<PointXYZ, PointXYZ> icp;
  icp.setTransformationEpsilon(1e-6);
  icp.setMaxCorrespondenceDistance(0.1);
  icp.setMaximumIterations(2);
  icp.setInputSource(src_cloud_ds);
  icp.setInputTarget(tgt_cloud_ds);

  Eigen::Matrix4f transform = Eigen::Matrix4f::Identity(), prev;
  for (int k = 0; k < 30; k++) {
    icp.align(*src_cloud_ds);
    icp.setInputSource(src_cloud_ds);

    transform = icp.getFinalTransformation() * transform;

    if (fabs((icp.getLastIncrementalTransformation() - prev).sum()) < icp.getTransformationEpsilon())
      icp.setMaxCorrespondenceDistance(icp.getMaxCorrespondenceDistance() - 0.001);
    
    prev = icp.getLastIncrementalTransformation();
  }

  transformPointCloud(*src_cloud, *target, icp.getFinalTransformation() * transform);

  cout << (icp.hasConverged() ? "Alignment succeeded!" : "Alignment failed.") << endl;
  cout << "Alignment score: " << icp.getFitnessScore() << endl;
  cout << "Final cloud size: " << target->width << " x " << target->height << endl;

  PCLVisualizer viewer ("PCL Viewer");
	viewer.initCameraParameters();

	int mix_viewport, target_viewport;

	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, mix_viewport);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, target_viewport);
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addCoordinateSystem(0.1);

	viewer.addText("Unaligned clouds", 10, 10, "mix_viewport_label", mix_viewport);
	viewer.addText("Aligned clouds", 10, 10, "target_viewport_label", target_viewport);

	viewer.addPointCloud<PointXYZ>(
      src_cloud,
      PointCloudColorHandlerCustom<PointXYZ>(src_cloud, 0, 255, 0),
      "cloud_1",
      mix_viewport);
	viewer.addPointCloud<PointXYZ>(
      tgt_cloud,
      PointCloudColorHandlerCustom<PointXYZ>(tgt_cloud, 255, 0, 0),
      "cloud_2",
      mix_viewport);

	viewer.addPointCloud<PointXYZ>(
      target,
      PointCloudColorHandlerCustom<PointXYZ>(target, 0, 255, 0),
      "target",
      target_viewport);
	viewer.addPointCloud<PointXYZ>(
      tgt_cloud,
      PointCloudColorHandlerCustom<PointXYZ>(tgt_cloud, 255, 0, 0),
      "cloud_2_aligned",
      target_viewport);

  cout << "Visualizing..." << endl;

  viewer.spin();

  return 0;
}
