from setuptools import find_packages, setup

package_name = 'depth_ros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='mi',
    maintainer_email='mi@todo.todo',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'depth_checker = depth_ros.1_gz_depth_checker:main',
            'yolo_checker = depth_ros.2_gz_yolo_rgb_checker:main',
            'yolo_depth_checker = depth_ros.3_gz_yolo_bbox_depth_checker:main',
            'tf_trans = depth_ros.4_gz_tf_transform:main',
            'object_xyz = depth_ros.5_gz_yolo_depth_object_xyz:main',
            'tf_point = depth_ros.6_gz_tf_point_marker_publisher:main',    
            'object_xyz_stable = depth_ros.5_gz_yolo_depth_object_xyz_stable:main',        
        ],
    },
)
