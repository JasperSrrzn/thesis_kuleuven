import json

facts = open('queryfile.pl').readlines()

dict_neighbors={}
for fact in facts:
    first = fact.find('(')
    second  = fact.find(',',first+1)
    third = fact.find(',',second+1)
    enta = fact[first+1:second]
    entb = fact[second+1:third]
    a = int(enta[enta.find('rel')+4:])
    b = int(entb[entb.find('rel')+4:])
    if a in dict_neighbors.keys():
        dict_neighbors[a].append(b)
    else:
        dict_neighbors[a] = [b]
    if b in dict_neighbors.keys():
        dict_neighbors[b].append(a)
    else:
        dict_neighbors[b] = [a]

json = json.dumps(dict_neighbors)
f = open("dict_neighbors.json","w")
f.write(json)
f.close()
