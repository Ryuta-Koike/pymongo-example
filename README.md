# pymongo-example

- install pymongo then check it
```
mongo [con_db_name]

show dbs;

use [dbname];
show collections;

db.[Collection].find().limit(10);
```

- set login info in ./conf.ini
```
[mongo]
id=your_user
password=**
```

- commands
```
#  forEach(printjson) は見やすく整形するメソッド
db.system.profile.find().forEach(printjson)

# sum 
#  ex: data = teamA, namea/ teamA, nameB / teamB, namec
#      result -> teamA: 2, teamB: 1
db.[collection].aggregate(
 {$project:{team:1}},
 {$group:{_id: "$team", count:{$sum:1}}},
 {$sort:{"count":-1}}
 )
```
# prepare dictionaries from dict files

- put dict file and set header line.
```
sed -i "1iheadword\tlabel\tdetail" *****.dic
```
- save to mongoDB
```
python init_collections_from_file.py -d"\t" -f **.dic --HEADER Y

[sample]
init_politely_dic.sh
```

# use as a submodule

- when you use this repo (pymongo) in your repoA, try followings
```
[add submodule]
git submodule add [pymong-repo.git]
git submodule update --init

[check]
cat /repoA/.gitmodules
git status

[use: commit in your repo]
git commit -m "add pymongo as a submodule"

[use: pull in your repo]
git submodule foreach git pull origin master
git status
```