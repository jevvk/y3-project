# -*- coding: utf-8 -*-
with open('README.rst') as f:
    readme = f.read()

with open('LICENSE') as f:
    license = f.read()

setup(
    name='y3p',
    version='0.0.1',
    description='Third year project',
    long_description=readme,
    author='Emilian Simion',
    author_email='emilian.simion@student.manchester.ac.uk',
    url='https://github.com/linoimi',
    license=license,
    packages=find_packages(exclude=('tests', 'docs'))
)
