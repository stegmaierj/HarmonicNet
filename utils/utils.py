# -*- coding: utf-8 -*-
"""
# HarmonicNet.
# Copyright (C) 2021 D. Eschweiler, M. Rethwisch, S.Koppers, J. Stegmaier
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the Liceense at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Please refer to the documentation for more information about the software
# as well as for installation instructions.
#
# If you use this application for your work, please cite the following 
# publication:
#
# D. Eschweiler, M. Rethwisch, S.Koppers, J. Stegmaier, 
# "Spherical Harmonics for Shape-Constrained 3D Cell Segmentation", ISBI, 2021.
#
"""

import time


def print_timestamp(msg, args=None):
    
    print('[{0:02.0f}:{1:02.0f}:{2:02.0f}] '.format(time.localtime().tm_hour,time.localtime().tm_min,time.localtime().tm_sec) +\
          msg.format(*args if not args is None else ''))

