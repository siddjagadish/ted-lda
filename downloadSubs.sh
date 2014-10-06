#Go into subtitles folder
mkdir subtitles
cd subtitles
for VID_URL in $(cat linkList.txt);
do
    python ../ted-talks-download/src/TEDSubs.py -s $VID_URL
done
