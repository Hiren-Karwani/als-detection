from app import app
import pandas as pd

# create test client
c = app.test_client()

# write a small CSV file
with open('test.csv','w') as f:
    f.write('label\n0\n1\n')

resp = c.post('/predict_tabular', data={'csv': (open('test.csv','rb'), 'test.csv')})
print('tabular status', resp.status_code, resp.json)

# test audio with no files
resp = c.post('/predict_audio', data={})
print('audio empty', resp.status_code, resp.json)
