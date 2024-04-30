#!/usr/bin/env bash
# coding: utf-8

# sample script to process quicklooks after receiving data

#stop on error
set -e

echo $HOSTNAME

days=$1 # how many days to look back? 
sett="${@:2}" #list of settings files to process



for SETTINGS in $sett 
do
   echo $SETTINGS

   # metaEvents
   echo "python3 -m VISSSlib scripts.loopCreateEvents /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1"
   python3 -m VISSSlib scripts.loopCreateEvents /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1
done


for SETTINGS in $sett 
do
   # level 0 quicklooks
   echo "python3 -m VISSSlib scripts.loopLevel0Quicklook /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1"
   python3 -m VISSSlib scripts.loopLevel0Quicklook /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1
done


for SETTINGS in $sett 
do
   # metaFrames
   echo "python3 -m VISSSlib scripts.loopCreateMetaFrames /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1"
   python3 -m VISSSlib scripts.loopCreateMetaFrames /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1
   #metaFrames Quicklook
   echo "python3 -m VISSSlib scripts.loopMetaFramesQuicklooks /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1"
   python3 -m VISSSlib scripts.loopMetaFramesQuicklooks /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1
done

for SETTINGS in $sett 
do
    # level1detect quicklooks, level1 detect on cluster!
    echo "python3 -m VISSSlib scripts.loopLevel1detectQuicklooks  /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1"
    python3 -m VISSSlib scripts.loopLevel1detectQuicklooks  /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1
done

for SETTINGS in $sett 
do
    # level1match quicklooks, level1 match on cluster!
    echo "python3 -m VISSSlib scripts.loopLevel1matchQuicklooks  /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1"
    python3 -m VISSSlib scripts.loopLevel1matchQuicklooks  /projekt1//ag_maahn/VISSS_config/$SETTINGS $days 1
done


for SETTINGS in $sett 
do
    echo "python3 -m VISSSlib scripts.reportLastFiles   /projekt1//ag_maahn/VISSS_config/$SETTINGS"
    python3 -m VISSSlib scripts.reportLastFiles   /projekt1//ag_maahn/VISSS_config/$SETTINGS
done
