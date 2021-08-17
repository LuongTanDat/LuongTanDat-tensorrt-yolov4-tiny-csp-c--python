from tqdm import tqdm
from glob import glob
import numpy as np
import json
# import pudb
# pudb.set_trace()

writer = [open("/mnt/642C9F7E0555E58A/Nobi/nobi-hw-videocapture/tabular/0.txt", mode="w", encoding="utf-8"),
         open("/mnt/642C9F7E0555E58A/Nobi/nobi-hw-videocapture/tabular/1.txt", mode="w", encoding="utf-8"),
         open("/mnt/642C9F7E0555E58A/Nobi/nobi-hw-videocapture/tabular/2.txt", mode="w", encoding="utf-8"),
         open("/mnt/642C9F7E0555E58A/Nobi/nobi-hw-videocapture/tabular/3.txt", mode="w", encoding="utf-8"),
         open("/mnt/642C9F7E0555E58A/Nobi/nobi-hw-videocapture/tabular/4.txt", mode="w", encoding="utf-8")]

for json_file in tqdm(glob("/mnt/642C9F7E0555E58A/Nobi/nobi-hw-videocapture/EMoi/*.json")):
    with open(json_file) as f:
        data = json.load(f)
        if data is None:
            continue
        else:
            for obj in data:
                cls = obj["det"]["cls"]
                if cls > 4:
                    continue
                else:
                    x_box = obj["det"]["bbox"][0]
                    y_box = obj["det"]["bbox"][1]
                    w_box = obj["det"]["bbox"][2]
                    h_box = obj["det"]["bbox"][3]
                    feat = np.array(obj["det"]["feat"])
                    xp = (np.array(obj["pose"]["x"]) - x_box) / w_box
                    yp = (np.array(obj["pose"]["y"]) - y_box) / h_box
                    sp = np.array(obj["pose"]["score"])
                    output = np.concatenate([np.vstack([xp, yp, sp]).T.reshape(-1), feat])
                    for _out in output:
                        writer[cls].write("\t".join([str(i) for i in output]))
                        writer[cls].write("\n")
                        # print("\t".join([str(i) for i in output]))

for w in writer:
    w.close()
