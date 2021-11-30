#!/usr/bin/env python3

from setuptools import setup, find_packages # type: ignore
import upf
import setuptools.command.install # type: ignore


upf_tamer_commit = '29dfa4b3c43ac6da0b97ca89e94450fce19b5681'
upf_pyperplan_commit = 'ddaf0cee57343638a8af0f1454e2a48dc5aa210f'

long_description=\
"""============================================================
 UPF: A library that unifies planning frameworks
 ============================================================
    Insert long description here
"""

setup(name='upf',
      version=upf.__version__,
      description='Unified planning framework',
      author='AIPlan4EU Organization',
      author_email='aiplan4eu@fbk.eu',
      url='https://aiplan4eu.fbk.eu/',
      packages=find_packages(),
      include_package_data=True,
      install_requires=['tarski @ git+https://github.com/aig-upf/tarski.git@ebfda1c13ac908904d5b74587971cc7149e73d85#egg=tarski[arithmetic]'],
      extras_require={
            'devs': [f'upf_tamer@git+https://github.com/aiplan4eu/tamer-upf.git@{upf_tamer_commit}',
                    f'upf_pyperplan@git+https://github.com/aiplan4eu/pyperplan-upf.git@{upf_pyperplan_commit}']
        },
      license='APACHE'
     )
