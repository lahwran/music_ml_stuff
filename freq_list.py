
import time
import datetime
import argparse
import json
import audio2numpy
import random
import os

import resource
gb = 1024*1024*1024
oldsoftlimit, oldhardlimit  = resource.getrlimit(resource.RLIMIT_DATA)
print oldsoftlimit, oldhardlimit
resource.setrlimit(resource.RLIMIT_DATA, (1 * gb, min(2 * gb, oldhardlimit)))
print resource.getrlimit(resource.RLIMIT_DATA)

parser = argparse.ArgumentParser()
parser.add_argument("jsonfile")
files = [x for x in json.load(open(parser.parse_args().jsonfile, "r"))
            if "peter" in x]
max = float(len(files))
time_start = time.time()
exceptions = 0
random.shuffle(files)
for index, filename in enumerate(files):
    if exceptions > 100:
        break
    try:
        audio2numpy.convert_to_png_freq(filename)
    except audio2numpy.SkipHack:
        continue
    except KeyboardInterrupt:
        break
    except:
        import traceback
        traceback.print_exc()
        exceptions += 1
        continue
    count = index + 1
    progress = int((float(count) / max) * 50)
    remaining_scaled = 50 - progress
    time_spent = time.time() - time_start
    avg_time = time_spent / count
    remaining = max - count
    remaining_time = datetime.timedelta(seconds=remaining * avg_time)
    print
    print "=" * progress + " " * remaining_scaled, remaining_time, remaining, avg_time, time_spent
