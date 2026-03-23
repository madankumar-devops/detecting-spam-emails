"""
Setup file for Detecting Spam Emails project.
KKR Gen AI Innovations | https://kkrgenaiinnovations.com/
"""
from setuptools import setup, find_packages

setup(
    name='detecting-spam-emails',
    version='1.0.0',
    description='Spam Email Detection using NLP and Machine Learning',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='KKR Gen AI Innovations',
    author_email='info@kkrgenaiinnovations.com',
    url='https://kkrgenaiinnovations.com/',
    packages=find_packages(),
    python_requires='>=3.8',
    install_requires=[
        'numpy>=1.24.0',
        'pandas>=2.0.0',
        'nltk>=3.8.0',
        'scikit-learn>=1.3.0',
        'matplotlib>=3.7.0',
        'seaborn>=0.12.0',
        'joblib>=1.3.0',
        'tqdm>=4.66.0',
    ],
    extras_require={
        'deep_learning': ['tensorflow>=2.13.0'],
        'visualization': ['wordcloud>=1.9.0'],
        'notebook': ['jupyter>=1.0.0'],
    },
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
)
