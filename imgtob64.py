import base64
import os
lst = []
for i in os.listdir('Frames'):
    with open(f"Frames/{i}", "rb") as img_file:
        my_string = base64.b64encode(img_file.read())
    lst.append(my_string.decode('UTF-8'))
with open('log.txt','w') as f:
    f.write(str(lst))