#! /usr/bin/env python


descr = """XXX"""

import os
from setuptools import setup, find_packages
from cpr.__init__ import __version__

DISTNAME = "cpr"
DESCRIPTION = descr
MAINTAINER = 'Franz Liem'
MAINTAINER_EMAIL = 'franziskus.liem@uzh.ch'
LICENSE = 'Apache2.0'
DOWNLOAD_URL = 'xxx'
VERSION = __version__

PACKAGES = find_packages()

if __name__ == "__main__":

    if os.path.exists('MANIFEST'):
        os.remove('MANIFEST')

    setup(name=DISTNAME,
          maintainer=MAINTAINER,
          maintainer_email=MAINTAINER_EMAIL,
          description=DESCRIPTION,
          license=LICENSE,
          version=VERSION,
          url=DOWNLOAD_URL,
          download_url=DOWNLOAD_URL,
          packages=PACKAGES,
          scripts=["scripts/run_cpr.py"],
          )
