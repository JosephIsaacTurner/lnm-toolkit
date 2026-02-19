from setuptools import setup, find_packages

setup(
    name='lnm',
    version='0.1',
    packages=find_packages(),
    author='Gemini',
    author_email='',
    description='A toolkit for lesion-to-network mapping analysis.',
    install_requires=[
        'numpy',
        'pandas',
        'nilearn',
        'scikit-learn',
    ],
)
