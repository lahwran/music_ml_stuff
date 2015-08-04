import json
import os
import localdata

extensions = set(["mp3", "au", "wav", "m4a", "aif"])
all_files = []
for input_dir in localdata.music_directories:
    for root, dirs, files in os.walk(input_dir, followlinks=True):
        for file in files:
            if file.rpartition(".")[-1].lower() in extensions:
                all_files.append(os.path.join(root, file))

buckets = [[] for x in range(4)]
for index, file in enumerate(all_files):
    buckets[index % len(buckets)].append(file)

for index, bucket in enumerate(buckets):
    with open("bucket_%d.json" % index, "w") as writer:
        json.dump(bucket, writer)
