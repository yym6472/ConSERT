# encoding=utf8

import glob
import os
import sys
import json

task=sys.argv[1]

res_list = []
for in_folder in glob.glob(f"output/*{task}_*"):
    if len(glob.glob(os.path.join(in_folder, "senteval_results*.json")))==0:
        print(in_folder)
        continue
        
    for res_file in glob.glob(os.path.join(in_folder, "senteval_results*.json")):
        with open(res_file) as fin:
            dat = json.load(fin)
            score = dat[task.upper()]["acc"]
            res_list.append((score, res_file))

res_list.sort(reverse=False)
for ln in res_list:
    print(ln)
        


