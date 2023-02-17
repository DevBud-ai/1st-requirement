# -*- coding: utf-8 -*-
"""
Created on Fri Feb 10 03:52:29 2023

@author: ASUS
"""
import detect_compo.ip_region_proposal as ip
# switch of the classification func
key_params = {'min-grad':10, 'ffl-block':5, 'min-ele-area':300,
              'merge-contained-ele':True, 'merge-line-to-paragraph':False, 'remove-bar':True}

input_path_img="images/sample.jpg"
output_root="example_output"
# yy=ip.compo_detection(input_path_img, output_root, key_params,
#                    classifier=classifier, resize_by_height=1200, show=False)
# switch of the classification func
classifier = None
is_clf=True
if is_clf:
    classifier = {}
    from cnn.CNN import CNN
    # classifier['Image'] = CNN('Image')
    classifier['Elements'] = CNN('Elements')
    # classifier['Noise'] = CNN('Noise')
yy=ip.compo_detection(input_path_img, output_root, key_params,
                   classifier=classifier, resize_by_height=resized_height, show=False)