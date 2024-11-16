from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup_args = generate_distutils_setup(
    packages=[
        'tauv_vision',
    ],
    scripts=[
        'scripts/yolact'
    ],
    package_dir={'': 'src'},
    requires=['rospy'],
)
setup(**setup_args)
