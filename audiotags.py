import mutagen.easyid3
import os
import musicdata
import json
import sys

basedir = os.path.expanduser("~/mpd")
jsonfile = musicdata.datadir.join("tags")

if jsonfile.check(file=True):
    tags = json.loads(jsonfile.read_binary())
else:
    tags = {}

def _findall(d):
    for root, _, filenames in os.walk(d, followlinks=True):
        for filename in filenames:
            yield os.path.join(root, filename)

def get_file_tags(relpath):
    return tags[relpath]

def save_json_file():
    jsonfile.write_binary(json.dumps(tags))

def getbytag(tag):
    return set([key for key, value in tags.items() if "@" + tag in value])

def main():
    files = _findall(basedir)

    for filename in files:
        relpath = os.path.relpath(filename, basedir)
        if not filename.lower().endswith(".mp3"):
            continue
        sys.stdout.write(".")
        sys.stdout.flush()
        id3 = mutagen.id3.ID3(filename)
        texts = (" ".join(c.text) for c in id3.getall("COMM:"))
        relevant = (text for text in texts if "@" in text)
        text = " ".join(relevant)
        file_tags= [tag for tag in text.split() if "@" in tag]
        tags[relpath] = file_tags

    save_json_file()


if __name__ == "__main__":
    print "Updating file tags",
    try:
        main()
    except:
        print "error:"
        raise
    else:
        print "done"
