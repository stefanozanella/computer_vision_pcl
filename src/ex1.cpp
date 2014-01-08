/**
 * PCL Lab - Ex #1
 *
 * Load a point cloud and show only the points with x < 0, all colored in blue.
 *
 * Author: Stefano Zanella
 * Date: 08/01/2014
 */

#include <iostream>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

using std::cout;
using std::endl;

using pcl::PointCloud;
using pcl::PointXYZ;
using pcl::PointXYZRGB;
using pcl::io::loadPCDFile;
using pcl::visualization::PCLVisualizer;
using pcl::visualization::PointCloudColorHandlerRGBField;

int main(int argc, char** argv) {
  PointCloud<PointXYZ>::Ptr cloud_in (new PointCloud<PointXYZ>);
  PointCloud<PointXYZRGB>::Ptr cloud_out (new PointCloud<PointXYZRGB>);

  if (loadPCDFile<PointXYZ>("../dataset/table_scene_lms400.pcd", *cloud_in) == -1) {
    PCL_ERROR("Couldn't read the pcd file.\n");
    return -1;
  }

  cout << "Original point cloud: " << cloud_in->width << " x " << cloud_in->height << endl;

  for (int k = 0; k < cloud_in->points.size(); k++) {
		if (cloud_in->points[k].x < 0)
		{
      PointXYZRGB p;
      p.x = cloud_in->points[k].x;
      p.y = cloud_in->points[k].y;
      p.z = cloud_in->points[k].z;

			p.r = 0;
			p.g = 0;
			p.b = 255;

      cloud_out->points.push_back(p);
		}
  }

  PCLVisualizer viewer ("PCL Viewer");
  int left_vp, right_vp;
	viewer.createViewPort(0.0, 0.0, 0.5, 1.0, left_vp);
	viewer.createViewPort(0.5, 0.0, 1.0, 1.0, right_vp);
	viewer.setBackgroundColor(0, 0, 0);
	viewer.addCoordinateSystem(0.1);
	viewer.initCameraParameters();

	viewer.addText("Original cloud", 10, 10, "left_vp_text", left_vp);
	viewer.addText("Segmented cloud", 10, 10, "right_vp_text", right_vp);

  viewer.addPointCloud(
      cloud_in,
      "original",
      left_vp);
  viewer.addPointCloud(
      cloud_out,
      PointCloudColorHandlerRGBField<PointXYZRGB>(cloud_out),
      "segmented",
      right_vp);

  cout << "Visualizing..." << endl;

  viewer.spin();

  return 0;
}
