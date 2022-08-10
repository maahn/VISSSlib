# -*- coding: utf-8 -*-


import os
import sys
import datetime

from copy import deepcopy

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from IPython.display import display, Image, clear_output
import ipywidgets as widgets

try:
    import cv2
except ImportError:
    warnings.warn("opencv not available!")

from . import __version__
from . import *

def imshow(img):
    import cv2
    import IPython
    _,ret = cv2.imencode('.jpg', img) 
    i = IPython.display.Image(data=ret)
    IPython.display.display(i)


class stereoView(object):
    def __init__(self, case, config, version=__version__):
        
        self.case = case
        self.config = config
        self.version = version
        self.cameras = [config.leader, config.follower]

        self.open()
        self.rr = 0

    def open(self):

        fL = files.FindFiles(self.case, self.config.leader, self.config, self.version)
        assert(len(fL.listFiles("level0")) == 1), f"Please select case so that this {fL.fnamesPattern.level0} results only in one file"
        fL1 = files.Filenames(fL.listFiles("level0")[0], self.config)

        #  open all the files
        fnamesLv0 = {}
        self.meta = {}
        self.lv1detect = {}
        self.videos = {}
        self.idDiffs = {}

        fnamesLv0[self.config.leader] = fL.listFiles("level0")[0]
        fnames0F = fL1.filenamesOtherCamera(graceInterval=-1, level="level0")
        if len(fnames0F)>1:
            print("Cannot handle camera restarts yet, taking only first file, omitting", fnames0F[1:])
            fnames0F = fnames0F[:1]
        fnamesLv0[self.config.follower] = fnames0F[0]

        self.meta[self.config.leader] = tools.open_mfmetaFrames(fL.listFiles("metaFrames"), self.config)
        self.lv1detect[self.config.leader] = tools.open_mflevel1detect(fL.listFiles("level1detect"), self.config)
        
        fnamesMF = fL1.filenamesOtherCamera(graceInterval=-1, level="metaFrames")
        if len(fnamesMF)>1:
            print("Cannot handle camera restarts yet, taking only first file, omitting", fnamesMF[1:])
            fnamesMF = fnamesMF[:1]
        fnames1F = fL1.filenamesOtherCamera(graceInterval=-1, level="level1detect")
        if len(fnames1F)>1:
            print("Cannot handle camera restarts yet, taking only first file, omitting", fnames[1:])
            fnames1F = fnames1F[:1]

        self.meta[self.config.follower] = tools.open_mfmetaFrames(fnamesMF, self.config)
        self.lv1detect[self.config.follower] = tools.open_mflevel1detect(fnames1F, self.config)
        
        self.lv1match = tools.open_mflevel1match(fL.listFiles("level1match"), self.config)
        
        #get capture ID diffs
        self.idDiff = tools.estimateCaptureIdDiffCore(*self.meta.values(), "capture_time")
        idDiff2 = tools.estimateCaptureIdDiffCore(*self.lv1detect.values(), "fpid")
        self.idDiffs[self.config.leader] = 0
        self.idDiffs[self.config.follower] = self.idDiff

        assert  self.idDiff == idDiff2, "estimateCaptureIdDiff did not come to same result for metaFrames and lv1detect"
        self.uniqueCaptureIds = xr.DataArray(
            np.sort(
                np.unique(
                    np.concatenate(
                        (
                            self.meta[self.config.leader].capture_id, 
                            self.meta[self.config.follower].capture_id-self.idDiff
                            )
                        )
                    )
                ),
            dims=["merged_record_id"]
            )
        self.captureTimes = {}
        for camera in self.cameras:
            self.videos[camera] = av.VideoReaderMeta(
                fnamesLv0[camera], 
                self.meta[camera], 
                lv1detect=self.lv1detect[camera], 
                lv1match=self.lv1match.sel(camera=camera), 
                config=self.config, 
                saveMode=False
            )
            self.captureTimes[camera] = xr.DataArray(
                self.meta[camera].capture_time, 
                coords=[self.meta[camera].capture_id], 
                dims=["capture_id"]
                )

    def get(self, rr, markParticles=True):
        self.rr = rr
        thisID = self.uniqueCaptureIds[rr].values

        frame = list()
        metaFrames = []
        lv1detects = []
        lv1matches = []
        
        for camera in self.cameras:
            if (thisID + self.idDiffs[camera]) in self.meta[camera].capture_id:
                captureTime = self.captureTimes[camera].sel(capture_id=thisID+ self.idDiffs[camera]).values
#                 captureTime = 0
                # print(f"found record {rr} in {camera} data at {captureTime}")
    
    
                res, self.frame1, meta1, meta2, meta3 = self.videos[camera].getFrameByCaptureTimeWithParticles(captureTime, markParticles=markParticles, highlightPid="meta")
                if self.frame1 is not None:
                    frame.append(self.frame1)
                else:
                    frame.append(
                        np.zeros((
                            self.config.frame_height+self.config.height_offset,
                            self.config.frame_width,
                            3
                        ), dtype=int) + 130
                    )
            else:
                # print(f"did not find record {rr} in {camera} data")
                frame.append(
                    np.zeros((
                        self.config.frame_height+self.config.height_offset,
                        self.config.frame_width,
                        3
                    ), dtype=int) 
                )
                meta1 =  meta2 = meta3 = None
            frame.append(np.zeros((self.config.frame_height+self.config.height_offset, 10, 3), dtype=int))
            metaFrames.append(meta1)
            lv1detects.append(meta2)
            lv1matches.append(meta3)

        try:
            lv1matches = xr.concat(lv1matches, dim="camera")
        except TypeError:
            lv1matches = None

        frame = np.concatenate(frame, axis=1)
        return frame, metaFrames, lv1detects, lv1matches
    
    def next(self):
        newrr = self.rr+1
        if newrr >= len(self.uniqueCaptureIds):
            print("end of movie file")
            return None, None, None, None
        return self.get(self.rr+1)
    
    def previous(self):
        newrr = self.rr-1
        if newrr < 0:
            print("beginning of movie file")
            return None, None, None, None
        return self.get(newrr)

    def nextCommon(self):
        while True:
            frame, metaFrames, lv1detects, lv1match = self.next()
            if np.sum([m is None for m in metaFrames])==1:
                continue
            else:
                break

        return frame, metaFrames, lv1detects, lv1match

    def previousCommon(self):
        while True:
            frame, metaFrames, lv1detects, lv1match = self.previous()
            if np.sum([m is None for m in metaFrames])==1:
                continue
            else:
                break
        return frame, metaFrames, lv1detects, lv1match
   

    def close(self, ):
        for camera in self.cameras:
            self.meta[camera].close()
            self.lv1detect[camera].close()
            self.videos[camera].release()


class stereoGUI():
    def __init__(self, case, config):
    
        self.sv = analysis.stereoView(case, config)
        
        return 
        
    def updateHandles(self, frame, metaFrames, lv1detects, lv1matches):

        self.metaFrames = metaFrames
        self.lv1detects = lv1detects
        self.lv1matches = lv1matches
        
        cc = 0
        mII = []
        if (self.lv1detects is not None) and (self.lv1detects[cc] is not None):
            
            for pid in self.lv1detects[cc].pid.values:
                mm = np.where(self.sv.lv1match.isel(camera=cc).pid == pid)[0]
                if len(mm) > 0:
                    mII.append(mm)
        if len(mII)>0:
            lv1match = self.sv.lv1match.isel(fpair_id=np.concatenate(mII))
        else:
            lv1match = None


        _, frame = cv2.imencode('.jpeg', frame)
        try:
            self.display_handle.update(Image(data=frame.tobytes())) 
        except AttributeError:
            self.display_handle = display(Image(data=frame.tobytes()),display_id=True)
            
        self.setNN(self.sv.rr)
        with self.out:
            self.out.clear_output()
            if metaFrames[0] is None:
                c0, i0 = "n/a", "n/a"
            else:
                c0, i0 = metaFrames[0].capture_time.values, metaFrames[0].capture_id.values
            if metaFrames[1] is None:
                c1, i1 = "n/a", "n/a"
            else:
                c1, i1 = metaFrames[1].capture_time.values, metaFrames[1].capture_id.values
            
            print("leader:", c0,i0, "follower:", c1,i1)
            if lv1match is not None:
                Zdiff = lv1match.position.isel(position_elements=[2,3]).diff("position_elements")
                print("%.7f score"%(lv1match.matchScore.values),"%.i ms"%(lv1match.capture_time.diff("camera").values[0].astype(int)/1e6 ) ,"%.2f y"%(lv1match.roi.diff("camera").sel(ROI_elements="y").values[0]),"%.2f h"%(lv1match.roi.diff("camera").sel(ROI_elements="h").values[0]), "%.2f Z"%Zdiff)
                print("#"*100)
            if (lv1detects[0] is not None) and (lv1detects[1] is not None):
                print(f"{'pid'.ljust(20)}: D {'X'.ljust(23)}, L {str(lv1detects[0]['pid'].values).ljust(23)}, F {str(lv1detects[1]['pid'].values).ljust(23)}") 
                for i in lv1detects[0].data_vars:
                    if i in ["touchesBorder", "pixPercentiles", "nThread", "record_id", ]: continue
                    print(f"{i.ljust(20)}: D {str(lv1detects[0][i].values - lv1detects[1][i].values).ljust(23)}, L {str(lv1detects[0][i].values).ljust(23)}, F {str(lv1detects[1][i].values).ljust(23)}") 


    def getNN(self):
        nn = int(self.texts[0].get_interact_value())
        return nn
    
    def setNN(self, nn):
        self.texts[0].value = str(nn)

    def createGUI(self, pid=0):
            
        self.out = widgets.Output()

        layout = widgets.Layout(width='auto', height='30px') #set width and height

        buttonNext = widgets.Button(description='>>', layout=layout)
        buttonPrev = widgets.Button(description='<<', layout=layout)
        buttonNext.on_click(lambda x: self.updateHandles(*self.sv.nextCommon()))
        buttonPrev.on_click(lambda x: self.updateHandles(*self.sv.previousCommon()))

        buttonNextFrame = widgets.Button(description='>', layout=layout)
        buttonPrevFrame = widgets.Button(description='<', layout=layout)
        buttonNextFrame.on_click(lambda x: self.updateHandles(*self.sv.next()))
        buttonPrevFrame.on_click(lambda x: self.updateHandles(*self.sv.previous()))

        
        buttons = [buttonPrev,buttonNext,buttonPrevFrame,buttonNextFrame]
        self.texts = []

        self.texts.append(widgets.Text(
            value=str(pid),
            description=f"{len(self.sv.uniqueCaptureIds)} tot. ids",
            disabled=False,
            width='auto',
        )
                        )
        
        load = widgets.Button(description='Load', layout=layout)
        load.on_click(lambda x: self.updateHandles(*self.sv.get(self.getNN())))
        self.texts.append(load)



        display_handle=None
        self.updateHandles(*self.sv.get(0))

        self.statusP = widgets.HTML(
            value="-",
        )

        # displaying button and its output together
        buttonsH = widgets.HBox(buttons)
        statusH = widgets.HBox(self.texts)

        return widgets.VBox([statusH,buttonsH, self.out])


