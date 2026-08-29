import requests

url = "http://localhost:8001/api/knowledge/upload?async_=true"

# file 为文件对象；strategy 普通表单字段
files = {
    "file": open("data/raw/test.txt", "rb")
}
data = {
    "strategy": "recursive"
}

resp = requests.post(url, files=files, data=data)

print(f"status_code: {resp.status_code}")
print(resp.text)

files["file"].close()