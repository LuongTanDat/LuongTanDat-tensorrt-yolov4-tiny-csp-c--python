import requests
import json
import base64
import os

URL = "http://localhost:2210/inference"
# URL = "https://2cb3-42-119-189-122.ap.ngrok.io/inference"

def test_api2(data):
    # import pudb; pudb.set_trace()
    r = requests.post(url=URL, data=data)
    print(r)
    print(r.json())


if __name__ == "__main__":
    while True:
        # /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/Nobi-raw-image/c58ded9ef3db4134a5a3c11c17165d83.jpg
        # /mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/Nobi-raw-image/00a6fe53dd8a14f15789139961bd93c9_0.jpg
        image_path = input("Input image path:  ")
        byte_content = base64.b64encode(open(image_path.strip(), "rb").read())
        img = byte_content.decode('utf-8')
        data = json.dumps({"image": img})
        with open(os.path.join("EMoi", os.path.basename(image_path).replace(".jpg", ".json")), mode="w") as f:
            f.write(data)

        test_api2(data)