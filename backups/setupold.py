from setuptools import setup
from launch_utils.utils import add_data_files, add_entry_points

package_name = 'caleuche_vrx_gz'

data_files = [('share/ament_index/resource_index/packages', ['resource/' + package_name]),
              ('share/' + package_name, ['package.xml'])]

# directory of folder name (key) with script type (value) that will be copy to share folder
folder_file_dir = {'launch':'.launch.py',  'config':'.y*', 'rviz':'.rviz', 'urdf':'all', 'worlds':'.sdf'}
data_files = add_data_files(data_files, package_name, folder_file_dir)

# ADD yours script nodes
script_nodes = ['twist4thruster'] 
entry_points = add_entry_points(script_nodes, package_name)

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='novi',
    maintainer_email='cristian.nova@uc.cl',
    description='TODO: Package description',
    license='Apache 2.0',
    tests_require=['pytest'],
    entry_points=entry_points,
)
