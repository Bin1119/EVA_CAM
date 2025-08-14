#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
EVA_CAM package initialization
"""

from .eva_cam_controller import EvaCamController
from .custom_motion_template import (
    MotionPattern, LinearMotionPattern, RotationalMotionPattern, 
    CircularMotionPattern, JointMotionPattern, CustomMotionBuilder
)

__version__ = "1.0.0"
__author__ = "EVA_CAM Development Team"
__description__ = "Integrated xArm and AlpLib Control System"

__all__ = [
    'EvaCamController',
    'MotionPattern', 'LinearMotionPattern', 'RotationalMotionPattern',
    'CircularMotionPattern', 'JointMotionPattern', 'CustomMotionBuilder'
]