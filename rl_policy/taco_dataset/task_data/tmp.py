from glob import glob
import os, json


for each in glob("*.json"):
    jf = json.load(open(each))
    if len(jf) > 3 and len(jf) < 6:
        print(each.split(".")[0])