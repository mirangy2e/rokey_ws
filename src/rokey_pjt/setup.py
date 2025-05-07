from setuptools import find_packages, setup

package_name = 'rokey_pjt'

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
            'crop_image = rokey_pjt.1_0_tb4_crop_image:main',
            'capture_image = rokey_pjt.1_1_tb4_capture_image:main',
            'cont_cap_image = rokey_pjt.1_2_tb4_cont_capture_image:main',
            'det_obj = rokey_pjt.1_3_tb4_yolov8_obj_det:main',
            'det_obj_thread = rokey_pjt.1_4_tb4_yolov8_obj_det_thread:main',
            'det_obj_track = rokey_pjt.1_5_tb4_yolov8_obj_det_track:main',
            'depth_check = rokey_pjt.2_1_tb4_depth_checker:main',
            'yolo_depth_checker = rokey_pjt.3_tb4_yolo_bbox_depth_checker:main',
            'tf_trans = rokey_pjt.4_tb4_tf_transform:main',
            'object_xyz = rokey_pjt.5_tb4_yolo_depth_object_xyz:main',
            'object_xyz_marker = rokey_pjt.5_tb4_yolo_depth_object_xyz_marker:main',   
            'nav_to_pose = rokey_pjt.6_nav_to_pose:main',
            'nav_to_pose_sc = rokey_pjt.6_nav_to_pose_sc:main',
            'nav_through_poses = rokey_pjt.6_nav_through_poses:main',
            'nav_through_poses_sc = rokey_pjt.6_nav_through_poses_sc:main',
            'follow_waypoints = rokey_pjt.6_follow_waypoints:main',
            'follow_waypoints_sc = rokey_pjt.6_follow_waypoints_sc:main', 
            'goal_pub = rokey_pjt.7_obj_goal_pose_publisher:main', 
            'goal_sub = rokey_pjt.7_obj_goal_pose_subscriber:main', 
        ],
    },
)
