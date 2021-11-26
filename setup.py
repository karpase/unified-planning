#!/usr/bin/env python3

from setuptools import setup, find_packages # type: ignore
import upf


upf_tamer_commit = 'acac3e41f6c47a9971771107a6a9e3f5d39e8131'
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
      author='UPF Team',
      author_email='info@upf.com',
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
