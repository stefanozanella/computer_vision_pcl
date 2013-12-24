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
		if (cloud_in->points[k].y < 0)
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
