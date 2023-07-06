from os import path
from setuptools import setup, find_packages


def load_readme_text():
    """Load in README file as a string."""
    try:
        dir_path = path.abspath(path.dirname(__file__))
        with open(path.join(dir_path, 'README.md'), encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""


def load_requirements():
    """Load in requirements.txt as a list of strings."""
    try:
        dir_path = path.abspath(path.dirname(__file__))
        with open(path.join(dir_path, 'requirements.txt'), encoding='utf-8') as f:
            install_requires = [line.strip() for line in f.readlines()]
            return install_requires
    except FileNotFoundError:
        return ""


setup(
    name='quantum_linear_systems',
    version='0.0.1',
    long_description=load_readme_text(),
    long_description_content_type='text/markdown',
    python_requires='>=3.8',
    packages=find_packages(),
    install_requires=load_requirements(),
    zip_safe=False,
    include_package_data=True,
    platforms='any',

)
