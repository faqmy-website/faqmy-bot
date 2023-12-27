set -x

FILES="documents/*.txt"
for f in ${FILES}
do
 echo "Processing $f file"
 curl -F "file=@${f}" http://127.0.0.1:8000/i/one/upload
done
