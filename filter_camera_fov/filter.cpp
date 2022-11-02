#include <boost/make_shared.hpp>
#include <pcl/point_types.h>
#include <pcl/point_cloud.h>
#include <pcl/point_representation.h>

#include <pcl/io/pcd_io.h>

#include <pcl/filters/voxel_grid.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/passthrough.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/frustum_culling.h>

#include <pcl/features/normal_3d.h>

#include <pcl/registration/icp.h>
#include <pcl/registration/icp_nl.h>
#include <pcl/registration/transforms.h>

#include <pcl/visualization/pcl_visualizer.h>

#include <string>
#include <thread>
#include <sstream>
#include <iostream>
#include <filesystem>

using namespace std::chrono_literals;

//convenient typedefs
typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloud;

//convenient structure to handle our pointclouds
struct PCD
{
  PointCloud::Ptr cloud;
  std::string f_name;

  PCD() : cloud (new PointCloud) {};
};


/** \brief The LoadData function loads a set of PCD files for later processing
  * \param argc the number of arguments (pass from main ())
  * \param argv the actual command line arguments (pass from main ())
  * \param models the resultant vector of point cloud datasets
  */
void loadData (int argc, char **argv, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{

  std::string extension (".pcd");
  for (int i = 1; i < argc; i++)
  {
    std::string fname = std::string (argv[i]);
    // Needs to be at least 4: .pcd
    if (fname.size () <= extension.size ())
      continue;

    std::transform (fname.begin (), fname.end (), fname.begin (), (int(*)(int))tolower);

    //check that the argument is a pcd file
    if (fname.compare (fname.size () - extension.size (), extension.size (), extension) == 0)
    {
      // Load the cloud and saves it into the global list of models
      PCD m;
      m.f_name = argv[i];
      pcl::io::loadPCDFile (argv[i], *m.cloud);
      //remove NAN points from the cloud
      std::vector<int> indices;
      pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud, indices);

      models.push_back (m);
    }
  }
}


void loadData (std::vector<std::filesystem::path> files_in_directory, std::vector<PCD, Eigen::aligned_allocator<PCD> > &models)
{
  PCL_INFO ("\n\nLoading %d PCD files...\n\n", (int)files_in_directory.size ());
  for (const auto & entry : files_in_directory){
    //std::cout << entry.string() << std::endl;

    PCD m;
    m.f_name = entry.string();
    pcl::io::loadPCDFile (entry.string(), *m.cloud);
    //remove NAN points from the cloud
    std::vector<int> indices;
    pcl::removeNaNFromPointCloud(*m.cloud,*m.cloud, indices);

    models.push_back (m);
  }
    
  
}

void filter(std::vector<PCD, Eigen::aligned_allocator<PCD> > &models, int min_x, int min_y, int min_z, int max_x, int max_y, int max_z){
  cout<<endl<<endl<<"Applying Filter"<<endl;
  cout<<"min x: "<<min_x<<" min y: "<<min_y<<" min z: "<<min_z<<endl;
  cout<<"max x: "<<max_x<<" max y: "<<max_y<<" max z: "<<max_z<<endl<<endl;

  PointCloud::Ptr cloud_filtered (new PointCloud);

  // Create the filtering object
  /*pcl::CropBox<PointT> boxFilter;
  boxFilter.setMin(Eigen::Vector4f(0, -20, -50, 1.0));
  boxFilter.setMax(Eigen::Vector4f(200, 20, 200, 1.0));*/
  
  /*
  pcl::PassThrough<PointT> pass;
  pass.setFilterFieldName ("z");
  pass.setFilterLimits (0.0, 1.0);
  */


  pcl::FrustumCulling<PointT> fc;
  fc.setVerticalFOV (100);
  fc.setHorizontalFOV (100);
  fc.setNearPlaneDistance (0.0);
  fc.setFarPlaneDistance (150);
   
  Eigen::Matrix4f camera_pose;
  //Eigen::Matrix3f mat3 = Eigen::Quaternionf(0.9945219, 0, 0, 0.1045285  ).toRotationMatrix();
  Eigen::Matrix3f mat3 = Eigen::Quaternionf(0.9969173, 0, 0, 0.0784591  ).toRotationMatrix();
  Eigen::Matrix4f mat4 = Eigen::Matrix4f::Identity();
  //Eigen::RowVector3f trans(0.087, 0.060, -0.076);
  Eigen::RowVector3f trans(0, 0, 0);
  mat4.block(0,0,3,3) = mat3;
  mat4.block(3,0,1,3) = trans;
  cout<<"Rotation matrix "<<endl<<mat4<<endl<<endl;
  // .. read or input the camera pose from a registration algorithm.
  fc.setCameraPose (mat4);
   
  
  for (auto & m : models){
    //std::cout<<"Filtering "<<m.f_name<<std::endl;
    //pass.setInputCloud (m.cloud);
    //boxFilter.setInputCloud(m.cloud);
    fc.setInputCloud (m.cloud);

    //pass.setFilterLimitsNegative (true);
    //pass.filter (*cloud_filtered);
    //boxFilter.filter(*cloud_filtered);
    fc.filter(*cloud_filtered);
    
    
    m.cloud = cloud_filtered;

    pcl::io::savePCDFileASCII (m.f_name, *m.cloud);
    
  }




}

 pcl::visualization::PCLVisualizer::Ptr simpleVis (PointCloud::ConstPtr cloud)
{
  // --------------------------------------------
  // -----Open 3D viewer and add point cloud-----
  // --------------------------------------------
  pcl::visualization::PCLVisualizer::Ptr viewer (new pcl::visualization::PCLVisualizer ("3D Viewer"));
  viewer->setBackgroundColor (0, 0, 0);
  viewer->addPointCloud<PointT> (cloud, "sample cloud");
  viewer->setPointCloudRenderingProperties (pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "sample cloud");
  viewer->addCoordinateSystem (1.0);
  viewer->initCameraParameters ();
  return (viewer);
}

int argv_to_int(char * argv){

  std::istringstream ss(argv);
  int x;
  if (!(ss >> x)) {
    std::cerr << "Invalid number: " << argv << '\n';
    return -100000;
  } else if (!ss.eof()) {
    std::cerr << "Trailing characters after number: " << argv << '\n';
    return -100000;
  }
  return x;
}


int main (int argc, char** argv)
{
	//std::cout<<argc;
  if (argc < 2){
    PCL_ERROR ("Error: Syntax is: %s <path_to_pcds>\nor\n%s <path_to_pcds> <min_x> <min_y> <min_z> <max_x> <max_y> <max_z>\n\n", argv[0],argv[0]);
    return -1;
  }else if(argc > 2 && argc < 8){
    PCL_ERROR ("Error: Syntax is: %s <path_to_pcds>\nor\n%s <path_to_pcds> <min_x> <min_y> <min_z> <max_x> <max_y> <max_z>\n\n", argv[0],argv[0]);
    return -1;
  }

  int min_x = 0, min_y = -20, min_z = -50;
  int max_x = 200, max_y = 20, max_z = 200;

  if(argc > 2){
    min_x = argv_to_int(argv[2]);
    min_y = argv_to_int(argv[3]);
    min_z = argv_to_int(argv[4]);

    max_x = argv_to_int(argv[5]);
    max_y = argv_to_int(argv[6]);
    max_z = argv_to_int(argv[7]);
  }


  // Get the PCD paths
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
  PCL_INFO ("Loaded %d datasets.", (int)data.size ());

  filter(data, min_x, min_y, min_z, max_x, max_y, max_z);

  pcl::visualization::PCLVisualizer::Ptr viewer;
  viewer = simpleVis(data[0].cloud);

  cout<<"Viewing "<<data[0].f_name<<endl;

  //--------------------
  // -----Main loop-----
  //--------------------
  while (!viewer->wasStopped ())
  {
    viewer->spinOnce (100);
    std::this_thread::sleep_for(100ms);
  }


	return (0);
}