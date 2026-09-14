from glob import glob
import os

from setuptools import setup

package_name = 'eco65_visual_grasp'

setup(
    name=package_name,
    version='0.1.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='proton',
    maintainer_email='proton@example.com',
    description='ECO65 eye-in-hand ArUco visual grasp node.',
    license='MIT',
    entry_points={
        'console_scripts': [
            'eco65_visual_grasp_node = eco65_visual_grasp.eco65_visual_grasp_node:main',
            'eco65_knob_detector_node = eco65_visual_grasp.eco65_knob_detector_node:main',
        ],
    },
)
