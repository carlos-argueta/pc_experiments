/*
A simple filter to keep only points approximately within the field of view of the camera

by Carlos Argueta

November 2, 2022
*/


#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/frustum_culling.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <string>
#include <thread>
#include <sstream>
#include <iostream>
#include <filesystem>

using namespace std::chrono_literals;

// Convenient typedefs
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloud;

// Convenient structure to handle our pointclouds
struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {};
};


/* Function to load PCD (Point Cloduds) from a directory. 
  param files_in_directory: path to directory containing PCD files
  param &data: vector with the loaded PCDs
*/
void loadData (std::vector<std::filesystem::path> files_in_directory, std::vector<PCD, Eigen::aligned_allocator<PCD> > &data)
{
  PCL_INFO ("\n\nLoading PCD files...\n\n");
  // Go over all of the entries in the path
  for (const auto & entry : files_in_directory){
    // if the entry is a file with extension .pcd, load it
    if (entry.extension() == ".pcd"){
      
      // Create pcd structure, assign it the path of the file as name and load the pcd to the cloud portion
      PCD p;
      p.f_name = entry.string();
      pcl::io::loadPCDFile (entry.string(), *p.cloud);

      // Remove NAN points from the cloud
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*p.cloud,*p.cloud, indices);

      // Add PCD structure to the vector
      data.push_back (p);
    }
  }
    
  
}

/*
  Fuction that uses the FrustumCulling filter to filter out points not possibly visible by the camera
  param: &data vector with the loaded PCDs
*/
void filter(std::vector<PCD, Eigen::aligned_allocator<PCD> > &data){
  cout<<endl<<endl<<"Applying Filter"<<endl;

  PointCloud::Ptr cloud_filtered (new PointCloud);

  // Create the filter
  pcl::FrustumCulling<PointT> fc;
  // The following parameters were defined by trial and error. 
  // You can modify them to better match your expected results
  fc.setVerticalFOV (100);
  fc.setHorizontalFOV (100);
  fc.setNearPlaneDistance (0.0);
  fc.setFarPlaneDistance (150);
   
  // Define the camera pose as a rotation and translation with respect to the LiDAR pose.
  Eigen::Matrix4f camera_pose = Eigen::Matrix4f::Identity();
  Eigen::Matrix3f rotation = Eigen::Quaternionf(0.9969173, 0, 0, 0.0784591  ).toRotationMatrix();
  Eigen::RowVector3f translation(0, 0, 0);
  // This is the most important part, it tells you in which direction to look in the Point Cloud
  camera_pose.block(0,0,3,3) = rotation; 
  camera_pose.block(3,0,1,3) = translation;
  cout<<"Camera Pose "<<endl<<camera_pose<<endl<<endl;
  fc.setCameraPose (camera_pose);
   
  // Go over each Point Cloud and filter it
  for (auto & d : data){
    // Run the filter on the cloud
    PointCloud::Ptr cloud_filtered (new PointCloud);
    fc.setInputCloud (d.cloud);
    fc.filter(*cloud_filtered);
    // Update the cloud 
    d.cloud = cloud_filtered;
    // Replace the PCD file with the filtered data
    pcl::io::savePCDFileASCII (d.f_name, *d.cloud);
    
  }

}

/* A very simple visualisation function to see the results of the filtering
  param cloud: the Point Cloud to visualize
*/
 pcl::visualization::PCLVisualizer::Ptr simpleVis (PointCloud::ConstPtr cloud)
{
  // Open 3D viewer and add point cloud
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<PointT> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}


int main (int argc, char** argv)
{
	
  if (argc < 2){
    PCL_ERROR ("Error: Syntax is: %s <path_to_pcds>\n\n", argv[0]);
    return -1;
  }
  
  // Get the PCD paths using the directory path received as argument then sort them
  std::vector<std::filesystem::path> files_in_directory;
  std::copy(std::filesystem::directory_iterator(argv[1]), std::filesystem::directory_iterator(), std::back_inserter(files_in_directory));
  std::sort(files_in_directory.begin(), files_in_directory.end());
  
  // Load data
  std::vector<PCD, Eigen::aligned_allocator<PCD> > data;
  loadData (files_in_directory, data);

  // Check user input
  if (data.empty ())
  {
    PCL_ERROR ("Syntax is: %s path_to_pcds ", argv[0]);
    
    return (-1);
  }
  PCL_INFO ("Loaded %d Point Clouds.", (int)data.size ());

  // Visualize one original Point Cloud
  pcl::visualization::PCLVisualizer::Ptr viewer;
  viewer = simpleVis(data[0].cloud);

  cout<<endl<<endl<<"Viewing "<<data[0].f_name<<endl<<"Close the viz window to continue."<<endl;


  // Viewer loop
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }


  // Filter and save the Point Clouds
  filter(data);

  // Visualize one filtered Point Cloud
  viewer = simpleVis(data[0].cloud);

  cout<<"Viewing "<<data[0].f_name<<endl;

  // Viewer loop
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }

	return (0);
}