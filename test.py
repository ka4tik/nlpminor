import csv
a={}
a["kartik"]=10
w = csv.writer(open("output.csv", "w"))
for key, val in a.items():
    w.writerow([key, val])

import csv
dict = {}
for key, val in csv.reader(open("input.csv")):
    dict[key] = val
