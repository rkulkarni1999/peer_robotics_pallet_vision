from setuptools import find_packages, setup

package_name = 'peer_robotics_pallet_vision'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/pallet_segmentor.launch.py']),
        ('share/' + package_name + '/launch', ['launch/pallet_detector.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pear_group',
    maintainer_email='you@example.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'pallet_detector = peer_robotics_pallet_vision.pallet_detector:main', 
            'detection_display_node = peer_robotics_pallet_vision.display_detection:main',
            'pallet_segmentor = peer_robotics_pallet_vision.pallet_segmentor:main', 
            'segmentation_display_node = peer_robotics_pallet_vision.display_segmentation:main',
        ],
    },
)
