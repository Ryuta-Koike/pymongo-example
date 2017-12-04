# Usage:

collection_name=$1

databases=(
  dictionaries
)

for i in ${databases[@]}; do
  mongodump -d $i -c $collection_name
done
