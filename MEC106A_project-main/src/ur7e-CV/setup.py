from setuptools import find_packages, setup
import os
from glob import glob

package_name = "ur7e_camera_vision"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml") + glob("config/*.json")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="Camera vision nodes for UR7e operation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "color_mapper_node = ur7e_camera_vision.color_mapper_node:main",
            "camera_info_node = ur7e_camera_vision.camera_info_node:main",
            "board_detector_node = ur7e_camera_vision.board_detector_node:main",
            "move_above_board_node = ur7e_camera_vision.move_above_board_node:main",
            "static_camera_tf_node = ur7e_camera_vision.static_camera_tf_node:main",
        ],
    },
)
