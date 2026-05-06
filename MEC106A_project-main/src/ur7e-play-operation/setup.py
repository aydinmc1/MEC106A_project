from setuptools import find_packages, setup
from glob import glob

package_name = "ur7e_play_operation"

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
    description="Operation state machine for UR7e play operation",
    license="MIT",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "operation_node = ur7e_play_operation.operation_node:main",
        ],
    },
)
