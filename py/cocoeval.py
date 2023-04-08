# -*- coding: utf-8 -*-

"""
@date: 2023/3/27 下午10:11
@file: eval.py
@author: zj
@description: 
"""

annType = ['segm','bbox','keypoints']
annType = annType[1]      #specify type here
prefix = 'person_keypoints' if annType=='keypoints' else 'instances'
print 'Running demo for *%s* results.'%(annType)