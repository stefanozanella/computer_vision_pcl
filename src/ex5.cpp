#include <iostream>
#include <algorithm>
#include <vector>

#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>
#include <pcl/filters/filter.h>
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
using pcl::visualization::PointCloudColorHandlerRGBField;
using pcl::visualization::PointCloudColorHandlerCustom;

int main(int argc, char** argv) {
  PointCloud<PointXYZ>::Ptr src_cloud_1 (new PointCloud<PointXYZ>);
  PointCloud<PointXYZ>::Ptr src_cloud_2 (new PointCloud<PointXYZ>);
  PointCloud<PointXYZRGB>::Ptr src_cloud_1_col (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr src_cloud_2_col (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB> target;

  if (loadPCDFile<PointXYZ>("../dataset/capture0001.pcd", *src_cloud_1) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    return -1;
  }
  vector<int> indices;
  removeNaNFromPointCloud(*src_cloud_1, *src_cloud_1, indices);

  cout << "Loaded point cloud: " << src_cloud_1->width << " x " << src_cloud_1->height << endl;

  for_each(src_cloud_1->begin(), src_cloud_1->end(), [src_cloud_1_col](PointXYZ &pt) {
      PointXYZRGB p (0, 255, 0);
      p.x = pt.x;
      p.y = pt.y;
      p.z = -pt.z;
      src_cloud_1_col->push_back(p);
  });

  if (loadPCDFile<PointXYZ>("../dataset/capture0002.pcd", *src_cloud_2) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    return -1;
  }
  removeNaNFromPointCloud(*src_cloud_2, *src_cloud_2, indices);

  cout << "Loaded point cloud: " << src_cloud_2->width << " x " << src_cloud_2->height << endl;

  for_each(src_cloud_2->begin(), src_cloud_2->end(), [src_cloud_2_col](PointXYZ &pt) {
      PointXYZRGB p (255, 0, 0);
      p.x = pt.x;
      p.y = pt.y;
      p.z = -pt.z;
      src_cloud_2_col->push_back(p);
  });

  // Non-linear, manually-iterative version
  //IterativeClosestPointNonLinear<PointXYZRGB, PointXYZRGB> icp;
  //icp.setInputSource(src_cloud_1_col);
  //icp.setInputTarget(src_cloud_2_col);
  //icp.setTransformationEpsilon(1e-6);
  //icp.setMaxCorrespondenceDistance(0.1);
  //icp.setMaximumIterations(2);

  //PointCloud<PointXYZRGB>::Ptr tptr (&target);
  //for (int k = 0; k < 30; k++) {
  //  cout << "Registration iteration #" << k << endl;

  //  icp.align(target);
  //  icp.setInputSource(tptr);
  //}

  IterativeClosestPoint<PointXYZRGB, PointXYZRGB> icp;
  icp.setInputSource(src_cloud_1_col);
  icp.setInputTarget(src_cloud_2_col);
  icp.align(target);

  cout << (icp.hasConverged() ? "Alignment succeeded!" : "Alignment failed.") << endl;
  cout << "Alignment score: " << icp.getFitnessScore() << endl;
  cout << "Final cloud size: " << target.width << " x " << target.height << endl;

  PCLVisualizer viewer ("PCL Viewer");
	viewer.initCameraParameters();

	int source_1_viewport, source_2_viewport, mix_viewport, target_viewport;

	viewer.createViewPort(0.0, 0.5, 0.5, 1.0, source_1_viewport);
	viewer.setBackgroundColor(0, 0, 0, source_1_viewport);
	viewer.addCoordinateSystem(0.1, source_1_viewport);
	viewer.addText("Source point cloud 1", 10, 10, "source_1_viewport_label", source_1_viewport);
	viewer.addPointCloud<PointXYZRGB>(
      src_cloud_1_col,
      PointCloudColorHandlerRGBField<PointXYZRGB>(src_cloud_1_col),
      "source_1",
      source_1_viewport);

	viewer.createViewPort(0.0, 0.0, 0.5, 0.5, source_2_viewport);
	viewer.setBackgroundColor(0, 0, 0, source_2_viewport);
	viewer.addCoordinateSystem(0.1, source_2_viewport);
	viewer.addText("Source point cloud 2", 10, 10, "source_2_viewport_label", source_2_viewport);
	viewer.addPointCloud<PointXYZRGB>(
      src_cloud_2_col,
      PointCloudColorHandlerRGBField<PointXYZRGB>(src_cloud_2_col),
      "source_2",
      source_2_viewport);

	viewer.createViewPort(0.5, 0.5, 1.0, 1.0, mix_viewport);
	viewer.setBackgroundColor(0, 0, 0, mix_viewport);
	viewer.addCoordinateSystem(0.1, mix_viewport);
	viewer.addText("Unaligned clouds", 10, 10, "mix_viewport_label", mix_viewport);
  PointCloud<PointXYZRGB>::Ptr mix_cloud (new PointCloud<PointXYZRGB>);
  for_each(src_cloud_1_col->begin(), src_cloud_1_col->end(), [mix_cloud](PointXYZRGB &pt) {
      mix_cloud->points.push_back(pt);
  });

  for_each(src_cloud_2_col->begin(), src_cloud_2_col->end(), [mix_cloud](PointXYZRGB &pt) {
      mix_cloud->points.push_back(pt);
  });

	viewer.addPointCloud<PointXYZRGB>(
      mix_cloud,
      PointCloudColorHandlerRGBField<PointXYZRGB>(mix_cloud),
      "mix",
      mix_viewport);

	viewer.createViewPort(0.5, 0.0, 1.0, 0.5, target_viewport);
	viewer.setBackgroundColor(0, 0, 0, target_viewport);
	viewer.addCoordinateSystem(0.1, target_viewport);
	viewer.addText("Aligned clouds", 10, 10, "target_viewport_label", target_viewport);
  PointCloud<PointXYZRGB>::Ptr target_ptr (&target);
  for_each(src_cloud_2_col->begin(), src_cloud_2_col->end(), [target_ptr](PointXYZRGB &pt) {
      target_ptr->points.push_back(pt);
  });

	viewer.addPointCloud<PointXYZRGB>(
      target_ptr,
      PointCloudColorHandlerRGBField<PointXYZRGB>(target_ptr),
      "target",
      target_viewport);

  cout << "Visualizing..." << endl;

  viewer.spin();

  return 0;
}
