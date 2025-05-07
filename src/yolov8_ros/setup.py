from setuptools import find_packages, setup

package_name = 'yolov8_ros'

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
            'image_pub = yolov8_ros.9_1_image_publisher:main',
            'image_sub = yolov8_ros.9_2_image_subscriber:main',
            'data_pub = yolov8_ros.9_3_data_publisher:main',
            'data_sub = yolov8_ros.9_4_data_subscriber:main',
            'yolo_pub = yolov8_ros.9_5_yolo_publisher:main',
            'yolo_sub = yolov8_ros.9_6_yolo_subscriber:main',
        ],
    },
)
