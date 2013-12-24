#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/visualization/pcl_visualizer.h>

using std::cout;
using std::endl;

using pcl::PointCloud;
using pcl::PointXYZ;
using pcl::PointXYZRGB;
using pcl::VoxelGrid;
using pcl::io::loadPCDFile;
using pcl::visualization::PCLVisualizer;
using pcl::visualization::PointCloudColorHandlerRGBField;

int main(int argc, char** argv) {
  PointCloud<PointXYZ>::Ptr cloud_in (new PointCloud<PointXYZ>);

  if (loadPCDFile<PointXYZ>("../dataset/table_scene_lms400.pcd", *cloud_in) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    return -1;
  }

  cout << "Original point cloud: " << cloud_in->width << " x " << cloud_in->height << endl;

  PointCloud<PointXYZRGB>::Ptr upper_left (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr upper_right (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr bottom_left (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr bottom_right (new PointCloud<PointXYZRGB>);

  for (int k = 0; k < cloud_in->points.size(); k++) {
      PointXYZRGB p;
      p.x = cloud_in->points[k].x;
      p.y = cloud_in->points[k].y;
      p.z = cloud_in->points[k].z;

      if (p.x > 0 && p.y > 0) {
        p.r = 255;
        p.g = 255;
        p.b = 255;

        upper_right->points.push_back(p);
      }

      if (p.x > 0 && p.y < 0) {
        p.r = 0;
        p.g = 0;
        p.b = 255;

        bottom_left->points.push_back(p);
      }

      if (p.x < 0 && p.y > 0) {
        p.r = 0;
        p.g = 255;
        p.b = 0;

        upper_left->points.push_back(p);
      }

      if (p.x < 0 && p.y < 0) {
        p.r = 255;
        p.g = 0;
        p.b = 0;

        bottom_right->points.push_back(p);
      }
  }

  PointCloud<PointXYZRGB>::Ptr upper_left_filtered (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr upper_right_filtered (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr bottom_left_filtered (new PointCloud<PointXYZRGB>);
  PointCloud<PointXYZRGB>::Ptr bottom_right_filtered (new PointCloud<PointXYZRGB>);

	VoxelGrid<PointXYZRGB> sor;
	sor.setInputCloud(upper_left);
	sor.setLeafSize(0.01f, 0.01f, 0.01f);
	sor.filter(*upper_left_filtered);

	sor.setInputCloud(upper_right);
	sor.setLeafSize(0.1f, 0.1f, 0.1f);
	sor.filter(*upper_right_filtered);

	sor.setInputCloud(bottom_left);
	sor.setLeafSize(0.005f, 0.005f, 0.005f);
	sor.filter(*bottom_left_filtered);

	sor.setInputCloud(bottom_right);
	sor.setLeafSize(0.05f, 0.05f, 0.05f);
	sor.filter(*bottom_right_filtered);


  PointCloud<PointXYZRGB>::Ptr cloud_out (new PointCloud<PointXYZRGB>);

  for (int k = 0; k < upper_left_filtered->points.size(); k++) {
    PointXYZRGB p;
    p.x = upper_left_filtered->points[k].x;
    p.y = upper_left_filtered->points[k].y;
    p.z = upper_left_filtered->points[k].z;
    p.r = upper_left_filtered->points[k].r;
    p.g = upper_left_filtered->points[k].g;
    p.b = upper_left_filtered->points[k].b;

    cloud_out->points.push_back(p);
  }

  for (int k = 0; k < upper_right_filtered->points.size(); k++) {
    PointXYZRGB p;
    p.x = upper_right_filtered->points[k].x;
    p.y = upper_right_filtered->points[k].y;
    p.z = upper_right_filtered->points[k].z;
    p.r = upper_right_filtered->points[k].r;
    p.g = upper_right_filtered->points[k].g;
    p.b = upper_right_filtered->points[k].b;

    cloud_out->points.push_back(p);
  }

  for (int k = 0; k < bottom_left_filtered->points.size(); k++) {
    PointXYZRGB p;
    p.x = bottom_left_filtered->points[k].x;
    p.y = bottom_left_filtered->points[k].y;
    p.z = bottom_left_filtered->points[k].z;
    p.r = bottom_left_filtered->points[k].r;
    p.g = bottom_left_filtered->points[k].g;
    p.b = bottom_left_filtered->points[k].b;

    cloud_out->points.push_back(p);
  }

  for (int k = 0; k < bottom_right_filtered->points.size(); k++) {
    PointXYZRGB p;
    p.x = bottom_right_filtered->points[k].x;
    p.y = bottom_right_filtered->points[k].y;
    p.z = bottom_right_filtered->points[k].z;
    p.r = bottom_right_filtered->points[k].r;
    p.g = bottom_right_filtered->points[k].g;
    p.b = bottom_right_filtered->points[k].b;

    cloud_out->points.push_back(p);
  }

  PCLVisualizer viewer ("PCL Viewer");
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addCoordinateSystem(0.1);
	viewer.initCameraParameters();
	viewer.addText("Blue cloud", 10, 10);
  viewer.addPointCloud(
      cloud_out,
      PointCloudColorHandlerRGBField<PointXYZRGB>(cloud_out),
      "cloud");

  cout << "Visualizing..." << endl;

  viewer.spin();
  return 0;
}
