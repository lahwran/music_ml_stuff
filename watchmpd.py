import os
import mpd
import sys
import datetime
import gzip
import time
import json
from localdata import mpd_password as password

x = os.path.expanduser("~/.musicdata")
try:
    os.mkdir(x)
except:
    # open up your mouth and EAT IT
    pass

client = mpd.MPDClient()

client.connect("localhost", 6600)
client.password(password)

def pall(topics):
    output = gzip.open(os.path.join(x, "log"), "a")
    currentsong = client.currentsong().get("file")

    now = datetime.datetime.now().isoformat("T")

    try:
        output.write(json.dumps({
            "c": currentsong,
            "t": topics,
            "d": now,
            "st": client.status(),
            "ss": client.stats(),
        }) + "\n")
    finally:
        output.close()

def main():
    init_reason = "_init"
    while True:
        try:
            pall([init_reason])
            while True:
                pall(client.idle())
        except (mpd.MPDError, IOError):
            try:
                client.disconnect()
            except:
                import traceback
                traceback.print_exc()
            for x in range(10):
                try:
                    client.connect("localhost", 6600)
                    client.password(password)
                    break
                except:
                    import traceback
                    traceback.print_exc()
            init_reason = "_reconnect"
        time.sleep(0.25)

if __name__ == "__main__":
    main()
