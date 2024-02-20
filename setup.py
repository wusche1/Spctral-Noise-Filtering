from setuptools import setup, find_packages

setup(
    name='SpectralMechanicsAnalysis',  # Replace with your own package name
    version='0.1.0',  # The initial release version
    author='Julian Schulz',  # Your name or your organization/company name
    author_email='mail@julianschulz.eu',  # Your email or your organization's email
    description='A set of analysis tools for cellular mechanics via powerspectra.',  # A short description of your package
    long_description=open('README.md').read(),  # A long description from your README.md
    long_description_content_type='text/markdown',  # Specify the content type of the long description
    url='https://github.com/wusche1/SpectralMechanicsAnalysis',  # The URL of your package's GitHub repo
    packages=find_packages(exclude=['demos', 'data']),  # Automatically find your package and any subpackages
    install_requires=[
        'numpy==1.24.2',
        'os',
        'concurrent',
        'tqdm==4.66.1',
        'scipy==1.10.1',
        'copy',
        'matplotlib==3.7.1',
        'collections',
        'sys',
    ],
    classifiers=[
        'Development Status :: 3 - Alpha',  # Choose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
        'Intended Audience :: Academia',  # Define the audience of your package
        'License :: OSI Approved :: MIT License',  # Choose the license as you wish (should match "LICENSE" file)
        'Programming Language :: Python :: 3',  # Specify which pyhton versions that you want to support
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    python_requires='>=3.7',  # Specify the Python versions you support here
)
