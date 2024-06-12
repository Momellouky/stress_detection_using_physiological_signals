from setuptools import find_packages, setup
from typing import List

HYPHEN_E_DOT = '-e .'

def get_requirements(file_path:str) -> List[str] : 
    '''
        Returns a list of requirements
    '''
    
    requirements = []
    with open(file_path, 'r') as file : 
        requirements = file.readlines()
        requirements = [req.replace('\n', '') for req in requirements]
        if HYPHEN_E_DOT in requirements : 
            requirements.remove(HYPHEN_E_DOT)
            
    print(f'requirements : {requirements}')
        
    return requirements

setup (
    name='Stress_detection_using_ml_and_physio_signals', 
    version='0.0.1',
    author='M. MELLOUKY', 
    author_email='contact.mellouky@gmail.com', 
    packages=find_packages(), 
    install_requires=get_requirements('requirements.txt')
)