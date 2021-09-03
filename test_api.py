import requests
import json
import base64

URL = "http://localhost:2210/inference"
image_path = "/mnt/2B59B0F32ED5FBD7/Projects/KIKAI/samples/Nobi-raw-image/c58ded9ef3db4134a5a3c11c17165d83.jpg"
byte_content = base64.b64encode(open(image_path, "rb").read())
img = byte_content.decode('utf-8')

def test_api2():
    import pudb; pudb.set_trace()
    r = requests.post(url=URL, data=json.dumps({"image": img}), headers={'content-type': 'application/json'})

if __name__ == "__main__":
    test_api2()

