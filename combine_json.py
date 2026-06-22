import json

with open("part1.json" , "r") as f:
    part1 = json.load(f)

with open("part2.json" , "r") as f:
    part2 = json.load(f)

with open("part3.json" , "r") as f:
    part3 = json.load(f)

combined= part1+part2+part3

with open("combined.json" , "w") as f:
    json.dump(combined , f , indent=4)

print(f"Total chunks: {len(combined)}")