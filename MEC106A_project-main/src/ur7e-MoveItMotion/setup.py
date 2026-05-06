from setuptools import find_packages, setup
from glob import glob

package_name = "ur7e_motion"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.py")),
        (f"share/{package_name}/config", glob("config/*.yaml")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Your Name",
    maintainer_email="you@example.com",
    description="MoveIt2 motion wrappers for UR7e",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "motion_server_node = ur7e_motion.motion_server_node:main",
        ],
    },
)
