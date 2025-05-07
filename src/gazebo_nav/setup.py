from setuptools import find_packages, setup

package_name = 'gazebo_nav'

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
            'current_pose = gazebo_nav.current_pose_checker:main',
            'nav_to_pose = gazebo_nav.nav_to_pose:main',
            'nav_to_pose_sc = gazebo_nav.nav_to_pose_sc:main',
            'nav_through_poses = gazebo_nav.nav_through_poses:main',
            'nav_through_poses_sc = gazebo_nav.nav_through_poses_sc:main',
            'follow_waypoints = gazebo_nav.follow_waypoints:main',
            'follow_waypoints_sc = gazebo_nav.follow_waypoints_sc:main',
            'goal_pub = gazebo_nav.obj_goal_pose_publisher:main',
            'goal_sub = gazebo_nav.obj_goal_pose_subscriber:main',
            # 'create_path = gazebo_nav.create_path:main',
            # 'patrol_loop = gazebo_nav.patrol_loop:main',
            # 'mail_delivery = gazebo_nav.mail_delivery:main',
        ],
    },
)
