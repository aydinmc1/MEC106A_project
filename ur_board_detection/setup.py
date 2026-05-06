from setuptools import find_packages, setup

package_name = 'ur_board_detection'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        # Register package in the ament resource index
        (
            'share/ament_index/resource_index/packages',
            ['resource/' + package_name],
        ),
        # Install package.xml so downstream packages can find metadata
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='kishantm1',
    maintainer_email='kishantm2004@gmail.com',
    description=(
        'ROS2 detection package for the Operation board game using a UR7e '
        'robot arm and RealSense D435i camera.'
    ),
    license='MIT',
    entry_points={
        'console_scripts': [
            'board_detector_node = ur_board_detection.board_detector_node:main',
            'move_above_board_node = ur_board_detection.move_above_board_node:main',
        ],
    },
)
