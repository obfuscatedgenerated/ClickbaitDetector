import generate as gen
import json
from pyyoutube import Api
import numpy as np
 
f = open("tokens.json", "r")
data = json.load(f)
ytkey = data["youtube"]
f.close()

api = Api(api_key=ytkey)

id = input("Input a channel ID/Name: ")

# Get the channel's uploads playlist ID, usually the same as the channel, but best to be sure
#chanreq = requests.get("https://www.googleapis.com/youtube/v3/channels?part=snippet%2CcontentDetails%2Cstatistics&id="+id+"&fields=items(contentDetails%2Cid%2Csnippet(country%2Cdescription%2Ctitle)%2Cstatistics%2Cstatus)%2CnextPageToken%2CpageInfo%2CprevPageToken%2CtokenPagination&key="+ytkey)
#if chanreq.status_code != 200:
#    raise Exception("Error: API request failed with status code: " + str(chanreq.status_code))
#    exit()
#else:
#    chandat = chanreq.json()
#    chanuploads = chandat["items"][0]["contentDetails"]["relatedPlaylists"]["uploads"]
#    print(chandat)

# Get the channel's uploads playlist ID, usually the same as the channel, but best to be sure
# Use ID or username
try:
    if id.startswith("UC"):
        chan = api.get_channel_info(channel_id=id).items[0].to_dict()
    else:
        chan = api.get_channel_info(channel_name=id).items[0].to_dict()
except TypeError:
    print("Error: Could not find channel")
    exit()
name = chan["snippet"]["title"]
chanuploads = chan["contentDetails"]["relatedPlaylists"]["uploads"]
print("Uploads Playlist ID: ",chanuploads)

# Get the playlist
plist = api.get_playlist_items(playlist_id=chanuploads, count=None).items
print("Playlist: ",plist)

titles = []

for i in plist:
    titles.append(i.to_dict()["snippet"]["title"])

#print("Titles: ",titles)

scores = gen.predict(titles)

#print("Clickbait probabilities: ",scores)

avg = np.mean(scores)
peakidx = np.argmax(scores)
peak = scores[peakidx][0]
peakname = titles[peakidx]
sums = np.sum(scores)

print("\n\n\n")
print("==============================")
print("Channel ID:",id)
print("Channel Name:",name)
print("Average clickbait probability:",avg)
print("Peak clickbait probability:",peak,"on video:",peakname)
print("Total clickbait probability:",sums)
print("==============================")