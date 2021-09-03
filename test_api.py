import requests
import json
import base64
import os
from glob import glob
from tqdm import tqdm

URL = "http://192.168.120.103:2210/inference"
# URL = "http://localhost:2210/inference"
# URL = "https://2cb3-42-119-189-122.ap.ngrok.io/inference"


def test_api2(data):
    # import pudb; pudb.set_trace()
    r = requests.post(url=URL, data=data)
    # print(r)
    # print(r.json())


def test_api3():
    for image_path in tqdm(glob("/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/Nobi-raw-image/*.jpg")*100):
        byte_content = base64.b64encode(open(image_path.strip(), "rb").read())
        img = byte_content.decode('utf-8')
        data = json.dumps({"image": img})
        # with open(os.path.join("EMoi", os.path.basename(image_path).replace(".jpg", ".json")), mode="w") as f:
        #     f.write(data)
        test_api2(data)


if __name__ == "__main__":
    import pudb; pudb.set_trace()
    r = requests.get("http://192.168.120.103:8888/00a6fe53dd8a14f15789139961bd93c9_0.jpg")
    with open("EMoi/image.jpg", mode="wb") as f:
        f.write(r.content);
    print(r)