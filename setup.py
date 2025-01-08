from setuptools import setup, find_packages
import os

def read_requirements(file_path):
    """Reads requirements from a file."""
    requirements = []
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    requirements.append(line)
    return requirements

setup(
    name='aicap',
    version='0.1.0',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    install_requires=read_requirements('requirements.txt'),
    python_requires='>=3.8',
    author='HubertDomaros',
    author_email='your.email@example.com',
    description='A short description of your project',
    url='https://github.com/HubertDomaros/ai-capstone-proj',
)
