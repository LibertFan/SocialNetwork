import json
import random
random.seed = 100

f = open("sample.json",'r')
sample_json = json.load(f)
f.close()

def get_colors(nodes):
    color_list = []
    for node in nodes:
        if node['color'] not in color_list:
            color_list.append(node['color'])
    return color_list

def get_location_bounds(nodes):
    x_min, y_min, x_max, y_max = 0,0,0,0
    # location_bounds = [x_min,y_min,x_max,y_max]

    for node in nodes:
        if node['x'] < x_min:
            x_min = node['x']
        if node['x'] > x_max:
            x_max = node['x']
        if node['y'] < y_min:
            y_min = node['y']
        if node['y'] > y_max:
            y_max = node['y']

    return [x_min, y_min, x_max, y_max]

sample_nodes = sample_json['nodes']
color_list = get_colors(sample_nodes)
locations = get_location_bounds(sample_nodes)
# print(color_list)
# print(locations)

f = open("../social_info.txt",'r')
social_content = f.read().strip().split('\n')
f.close()

edges = []
user_followers_num = {}
for line in social_content:
    follower,followee = line.split("\t")
    d = {}
    d['sourceID'] = followee
    d['attributes'] = {}
    d['targetID'] = follower
    d['size'] = 1
    edges.append(d)
    if followee in user_followers_num.keys():
        user_followers_num[followee] += 1
    else:
        user_followers_num[followee] = 1

nodes = []
for user in user_followers_num.keys():
    d = {}
    d['color'] = random.choice(color_list)
    d['label'] = user
    d['attributes'] = {}
    d['x'] = random.uniform(-2000,2000)
    d['y'] = random.uniform(-2000,2000)
    d['id'] = user
    d['size'] = user_followers_num[user]
    nodes.append(d)

output_content = {"nodes": nodes,
                  "edges": edges}

fout = open('douban_user.json','w')
json.dump(output_content, fout)