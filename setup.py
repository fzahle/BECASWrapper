
from setuptools import setup, find_packages
import platform
import sys

# openmdao installs plugins with develop -N (no-deps)
# but we need to install oct2py from the master branch at github
if sys.argv[1] == 'develop' and len(sys.argv) == 3:
    sys.argv.pop(2)

kwargs = {'author': '',
 'author_email': '',
 'classifiers': ['Intended Audience :: Science/Research',
                 'Topic :: Scientific/Engineering'],
 'description': '',
 'download_url': '',
 'include_package_data': True,
 'install_requires': ['oct2py'],
 'keywords': ['openmdao'],
 'license': '',
 'maintainer': '',
 'maintainer_email': '',
 'name': 'becas_wrapper',
 'package_data': {'becas_wrapper': []},
 'package_dir': {'': 'src'},
 'packages': ['becas_wrapper'],
 'dependency_links':['https://github.com/blink1073/oct2py/tarball/master#egg=oct2py'],
 'url': '',
 'version': '0.1',
 'zip_safe': False}

if 'Windows' not in platform.platform():
    kwargs['install_requires'].append('pexpect')


setup(**kwargs)

