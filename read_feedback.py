import os
import json


feedback_path = ''
with open(feedback_path, 'r') as file:
    feedbacks = json.load(file)
    file.close()

overall_score = feedbacks['结果评分']

if feedbacks['搜索结果'][0]['Similarity'] == 1:
    pass