triplet=$1
ps aux | grep python | grep "$triplet" | awk '{print $2}' | xargs kill
