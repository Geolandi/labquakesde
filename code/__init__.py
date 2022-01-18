#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
PACKAGE_PARENTS = ['.','..']
SCRIPT_DIR = os.path.dirname(os.path.realpath(
	os.path.join(os.getcwd(),
	os.path.expanduser(__file__))))
for P in PACKAGE_PARENTS:
	sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, P)))
