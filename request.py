import requests


url = "http://localhost:8000/uploadfile/"
file_path = r'C:\Users\Admin\CAD_parts\Bearings\STL\Bearings_00ed2536-3d80-4f07-8851-4f49f1606498.stl'

with open(file_path, "rb") as file:
    response = requests.post(url, files={"file": file})

print(response.status_code)
print(response.json())