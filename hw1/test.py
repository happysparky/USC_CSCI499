import json

f = open("lang_to_sem_data.json")
f = json.load(f)

for i in f["train"][0]:
    print(i)
