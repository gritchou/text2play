from setuptools import setup, find_packages

setup(
    name='text2play',
    version='0.0.1',
    author='Jean-François Grand',
    author_email='gritchou@gmail.com',
    description='Transfer style from paintings to game assets from a user prompt',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/gritchou/text2play',
    packages=find_packages(where='src'),
    package_dir={'': 'src'},
    include_package_data=True,
    install_requires=[
        'torch>=1.7',
        'torchvision>=0.8.0',
        'Pillow',
        'matplotlib',
        'numpy',
        'pandas',
        'sentence-transformers>=0.4.0'
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.7',
    entry_points={
        'console_scripts': [
            'text2play=src.main:main',
        ],
    },
)
