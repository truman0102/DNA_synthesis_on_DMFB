from setuptools import setup, find_packages

setup(
    name='dna_on_dmfb',
    version='0.1.0',
    author='Hongguang Wang',
    author_email='truman@pku.edu.cn',
    description='A large-scale DMFB environment for DNA synthesis',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    packages=find_packages(),
    keywords='DMFB reinforcement learning simulation',
    install_requires=[
        'gif',
        'gymnasium',
        'matplotlib',
        'numpy',
        'seaborn',
        'tqdm'
    ],
)
