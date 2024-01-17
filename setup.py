from setuptools import setup, find_packages
from typing import List



HYPEN_E = '-e .'

def get_requirements(file_path:str)->List[str]:
    my_requirements = []
    with open(file_path) as file_obj:
        my_requirements = file_obj.readlines()
        my_requirements = [req.replace('\n', "") for req in my_requirements]

        if HYPEN_E in my_requirements:
            my_requirements.remove(HYPEN_E)
    
    return my_requirements

setup(
    name= 'HeartDiseasesClassification',
    version='0.0.1',
    author='augustin',
    author_email='augustin7766@gmail.com',
    packages=find_packages(),
    install_requires = get_requirements('requirements.txt')
)
