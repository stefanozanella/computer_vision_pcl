#include <iostream>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/io/pcd_io.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/features/normal_3d_omp.h>
#include <pcl/filters/filter.h>

using std::cout;
using std::endl;

using pcl::PointCloud;
using pcl::PointXYZRGB;
using pcl::NormalEstimationOMP;
using pcl::Normal;
using pcl::io::loadPCDFile;
using pcl::search::KdTree;
using pcl::visualization::PCLVisualizer;
using pcl::visualization::PointCloudColorHandlerRGBField;

int main(int argc, char** argv) {
  PointCloud<PointXYZRGB>::Ptr cloud_in (new PointCloud<PointXYZRGB>);
	PointCloud<Normal>::Ptr cloud_coarse_normals (new PointCloud<Normal>);
	PointCloud<Normal>::Ptr cloud_fine_normals (new PointCloud<Normal>);

  if (loadPCDFile<PointXYZRGB>("../dataset/minimouse1_segmented.pcd", *cloud_in) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    return -1;
  }

  cout << "Loaded point cloud: " << cloud_in->width << " x " << cloud_in->height << endl;

	NormalEstimationOMP<PointXYZRGB, Normal> ne;
	ne.setSearchMethod(KdTree<PointXYZRGB>::Ptr(new KdTree<PointXYZRGB>));
	ne.setNumberOfThreads(2);
	ne.setInputCloud(cloud_in);

	cout << "Computing coarse normals...please wait...";
	ne.setRadiusSearch (0.03);
	ne.compute(*cloud_coarse_normals);
	cout << "done." << endl;

	cout << "Computing fine normals...please wait...";
	ne.setRadiusSearch (0.002);
	ne.compute(*cloud_fine_normals);
	cout << "done." << endl;

	int normalsVisualizationStep = 100;
	float normalsScale = 0.02;

	PCLVisualizer viewer("PCL Viewer");
	viewer.initCameraParameters();
	PointCloudColorHandlerRGBField<PointXYZRGB> rgb (cloud_in);

	int left_viewport (0);
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, left_viewport);
	viewer.setBackgroundColor(0, 0, 0.5, left_viewport);
	viewer.addCoordinateSystem(0.1, left_viewport);
	viewer.addText("Coarse grain normals", 10, 10, "left_viewport text", left_viewport);
	viewer.addPointCloud<PointXYZRGB>(
      cloud_in,
      PointCloudColorHandlerRGBField<PointXYZRGB>(cloud_in),
      "input_cloud_left",
      left_viewport);
	viewer.addPointCloudNormals<PointXYZRGB, Normal>(cloud_in, cloud_coarse_normals, normalsVisualizationStep, normalsScale, "normals_left", left_viewport);

	int right_viewport (0);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, right_viewport);
	viewer.setBackgroundColor(0, 0, 0.5, right_viewport);
	viewer.addCoordinateSystem(0.1, right_viewport);
	viewer.addText("Fine grain normals", 10, 10, "right_viewport text", right_viewport);
	viewer.addPointCloud<PointXYZRGB>(
      cloud_in,
      PointCloudColorHandlerRGBField<PointXYZRGB>(cloud_in),
      "input_cloud_right",
      right_viewport);
	viewer.addPointCloudNormals<PointXYZRGB, Normal>(cloud_in, cloud_fine_normals, normalsVisualizationStep, normalsScale, "normals_right", right_viewport);

	cout << "Visualizing..."<< endl;

  viewer.spin();
  return 0;
}

