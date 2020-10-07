# -*- coding: utf-8 -*-
"""
Created on Wed Oct  7 22:32:38 2020

@author: Ajay
"""

import requests

url = 'http://localhost:5000/predict_api'
r = requests.post(url,json={'experience':2, 'test_score':9, 'interview_score':6})

print(r.json())