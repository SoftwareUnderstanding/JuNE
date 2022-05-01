import os
from distutils.core import setup
from setuptools import find_packages  # type: ignore


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


with open('requirements.txt', 'r') as f:
    install_requires = list()
    dependency_links = list()
    for line in f:
        re = line.strip()
        if re:
            if re.startswith('git+') or re.startswith('svn+') or re.startswith('hg+'):
                dependency_links.append(re)
            else:
                install_requires.append(re)

packages = find_packages()

version = {}
with open("JuNE/__init__.py") as fp:
    exec(fp.read(), version)

setup(
    name='JuNE',
    version=version["__version__"],
    packages=packages,
    url='https://github.com/DanielRodriguezRguez/TFG-Extraccion-metadatos-Jupyter-Notebook',
    license='BSD-3-Clause',
    author='Daniel Rodriguez',
    description='Extractor de metadatos sobre Jupyter Notebooks.',
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    include_package_data=True,
    install_requires=install_requires,
    dependency_links=dependency_links,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Programming Language :: Python",
    ],
    entry_points={
        'console_scripts': [
            'JuNE = JuNE.main:main',
        ],
    }
)
