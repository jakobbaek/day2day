import io
import os
from setuptools import setup, find_packages

def read_requirements():
    """Read requirements from requirements.txt."""
    with open('requirements.txt', 'r') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

def setup_package():
    root = os.path.abspath(os.path.dirname(__file__))
    
    # Read about.py for package information
    with io.open(os.path.join(root, "about.py"), encoding="utf8") as f:
        about = {}
        exec(f.read(), about)
    
    # Read README.md
    with io.open(os.path.join(root, "README.md"), encoding="utf8") as f:
        readme = f.read()
    
    setup(
        name=about["__title__"],
        version=about["__version__"],
        author=about["__author__"],
        author_email=about["__email__"],
        description=about["__summary__"],
        long_description=readme,
        long_description_content_type="text/markdown",
        url=about["__uri__"],
        packages=find_packages(),
        install_requires=read_requirements(),
        python_requires='>=3.8',
        entry_points={
            'console_scripts': [
                'day2day=day2day.api.cli:main',
            ],
        },
        classifiers=[
            "Development Status :: 3 - Alpha",
            "Intended Audience :: Financial and Insurance Industry",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.8",
            "Programming Language :: Python :: 3.9",
            "Programming Language :: Python :: 3.10",
            "Programming Language :: Python :: 3.11",
            "Topic :: Office/Business :: Financial :: Investment",
        ],
    )

if __name__ == "__main__":
    setup_package()
