from setuptools import setup, find_packages

setup(
    name='ml_exercises',
    version='0.1.0',
    description='A repository for implementing several machine learning algorithms',
    author='Prabhat Nagarajan',
    packages=find_packages(),
    python_requires='>=3.6',
    install_requires=[
        # Add dependencies here as needed
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
