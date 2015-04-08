# Searching for Ads using audfprint

Dan Ellis `dpwe@ee.columbia.edu` 2015-03-31

This document describes how to use the Python audio fingerprinting package [audfprint](https://github.com/dpwe/audfprint) to locate instances of known TV ads (or other exact sound clips) within an archive of recorded TV shows (or other longer recordings).  

It assumes we are running on a Linux host, with the relevant data files already accessible, in this example under the directory `/2/data/`.

* This version of audfprint relies on `ffmpeg` to read audio files.  You should check that you have a working version by typing `ffmpeg -version` at the command prompt.  If the command is not found, you should install it with something like `sudo apt-get install ffmpeg` (for Ubuntu) or `brew install ffmpeg` (for Macs with [homebrew](http://brew.sh) installed).

* Download and unpack [audfprint](https://github.com/dpwe/audfprint).  We'll assume we're running inside the source directory which contains `audfprint.py` etc.

```shell
$ wget https://github.com/dpwe/audfprint/archive/master.zip
$ unzip master.zip
$ cd audfprint-master/
```

* We will make plain text list files containing paths to all the reference advert files, and the TV shows to be searched.  These files can be audio or video; as long as `ffmpeg` can read them, we should be OK.

```shell
$ ls -1 /2/data/all_PHL_political_ads/*.mp4 > ads.list
$ ls -1 /2/data/20140924_daily_news_programming/*.mp4 > shows.list
```

* Build the reference database of known ads.  This creates `ads.db` in the current directory.  Since the total time of the reference ads is usually much smaller than the shows we're searching, this generally takes an insignificant proportion of the total time (e.g. 3 min per 100 ads).

```shell
$ ./audfprint.py new \
    --dbase ads.db \
    --density 100 \
    --samplerate 11025 \
    --shifts 4 \
    --list ads.list
```

* Precompute the landmark features of each show.  This writes a data file corresponding to each input file under `precomp/` in the current directory.  We run with `--ncores 4` to run on 4 cores in parallel.  This step takes around two-thirds of the total time, maybe 30 mins per 100 shows.  

```shell
$ ./audfprint.py precompute \
    --samplerate 11025 \
    --density 100 \
    --shifts 1 \
    --precompdir precomp \
    --ncores 4 \
    --list shows.list
```

* Do the matching of the precomputed landmarks against the reference database.  This step takes the remaining one-third of the time, or around 15 mins per 100 shows.

```shell
$ find precomp/ -name "*.afpt" > precomp.list
$ ./audfprint.py match \
    --dbase ads.db \
    --match-win 2 \
    --min-count 200 \
    --max-matches 100 \
    --sortbytime \
    --opfile matches.out \
    --ncores 4 \
    --list precomp.list
```

* Filter the output file written by the matching to extract the 4 fields we're interested in: the show name, the time of the start of the match (in seconds since the beginning of the show), the name of the matching ad, and the "match score" (count of common landmarks):

```shell
$ grep Matched matches.out \
    | sed -e "s@precomp@@" -e "s@/2/data/@@g" \
          -e "s/\.mp4//" -e "s/\.afpt//" \
    | awk '{print $2 "\t" (-1*$11) "\t" $9 "\t" $14}' \
    > HITS
```

* `HITS` now contains our list of hits, which should look something like the excerpt below.  Each line consists of: `SHOW_NAME \t MATCH_START_SEC \t MATCH_FILE \t MATCH_SCORE`.  Thanks to the `--min-count` option to `audfprint match`, we only include matches with scores of 200 or above (indicating a virtually certain match).

```
20140924_daily_news_programming/WTXF_20140924_110000_Good_Day_Philadelphia      1265.63 all_PHL_political_ads/PA_2014_Gov_Tom_Corbett_for_Governor_Tom_Wolf_(con)_pro_15_con_15 731
20140924_daily_news_programming/WTXF_20140924_110000_Good_Day_Philadelphia      2735.05 all_PHL_political_ads/PA_2014_Gov_Tom_Wolf_for_Governor_Tom_Wolf_(pro)_pro_23_con_7_#2  1030
20140924_daily_news_programming/WTXF_20140924_110000_Good_Day_Philadelphia      3310.03 all_PHL_political_ads/NJ_2014_CD3_Tom_MacArthur_for_Congress_Aimee_Belgard_(con)_pro_0_con_15_#1        466
20140924_daily_news_programming/WTXF_20140924_110000_Good_Day_Philadelphia      3445.19 all_PHL_political_ads/NJ_2014_CD3_Tom_MacArthur_for_Congress_Aimee_Belgard_(con)_pro_0_con_15_#2        387
...
```

That's it!



