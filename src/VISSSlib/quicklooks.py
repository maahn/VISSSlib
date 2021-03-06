# -*- coding: utf-8 -*-

import uuid

import numpy as np
import os
import sys
import datetime

from copy import deepcopy

import pandas as pd
import xarray as xr
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import matplotlib as mpl

from image_packer import packer


import warnings
try:
    import cv2
except ImportError:
    warnings.warn("opencv not available!")

from .tools import nicerNames, readSettings
from .av import VideoReaderMeta

from . import __version__
from . import files


def crop(image):
    """
    crop black image parts
    """
    y_nonzero, x_nonzero = np.nonzero(image)
    return image[np.min(y_nonzero):np.max(y_nonzero)+1, np.min(x_nonzero):np.max(x_nonzero)+1]


def createLevel1detectQuicklook(timestamp, camera, config, 
                       version = __version__,
                       container_width=200,
                       container_height_max=300,
                       nTiles=60,
                       nRows=4,
                       extra=1,
                       readParticlesFromFiles=True,
                       skipExisting=True,
                       ):

    minBlur = config["minBlur4Plotting"]
    minSize = config["minDmax4Plotting"]

    if type(config) is str:
        config = readSettings(config)

    total_width = (container_width + extra) * nTiles // nRows
    max_height = (20 + container_height_max) * nRows + 60

    # let use a matplotlib font becuase we can be sure it is there
    mpl_data_dir = os.path.dirname(mpl.matplotlib_fname())
    mpl_ttf_dir = os.path.join(mpl_data_dir, 'fonts', 'ttf')
    font = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 35)
    fontL = ImageFont.truetype(f"{mpl_ttf_dir}/DejaVuSans.ttf", 16)

    ff = files.FindFiles(timestamp, camera, config, version)

    site = config["site"]

    particlesPloted = 0

    if os.path.isfile(ff.quicklook.level1detect) and skipExisting:
        print("SKIPPING - file exists", ff.quicklook.level1detect)
        return None, None

    if site != "mosaic":
        if (len(ff.listFiles("level0")) == 0) and (len(ff.listFiles("level0status")) == 0):
            print("NO DATA YET (TRANSFERRED?)", ff.quicklook.level1detect)
            return None, None

    if (len(ff.listFilesExt("level1detect")) == 0) and (len(ff.listFiles("level0")) > 0):
        print("NO DATA YET ", ff.quicklook.level1detect)
        return None, None

    if not ff.isCompleteL1detect:
        print("NOT COMPLETE YET %i/%i %s" %
              (len(ff.listFilesExt("level1detect")), len(ff.listFiles("level0")), ff.quicklook.level1detect))
#         if (len(ff.listFilesExt("level1detect")) == len(ff.listFiles("level0"))):
#             afshgsa
        return None, None

#     else:
    print("RUNNING ", ff.quicklook.level1detect)

    ff.createQuicklookDirs()

    dats2 = []

    for fname1 in ff.listFiles("metaFrames"):
        fname2 = fname1.replace('metaFrames', 'level1detect')
        try:
            dat2 = xr.open_dataset(fname2)
        except FileNotFoundError:
            if os.path.isfile(f"{fname2}.nodata"):
                continue
            elif os.path.isfile(f"{fname2}.broken.txt"):
                continue
            elif os.path.isfile(f"{fname2}.notenoughframes"):
                continue
            else:
                raise FileNotFoundError(fname2)

        dat2 = dat2[["Dmax", "blur", "touchesBorder",
                     "record_time", "record_id", "roi"]]
        dat2 = dat2.where((dat2.blur > minBlur) & (dat2.Dmax > minSize) & (
            ~(dat2.touchesBorder.any('side')))).dropna('pid')

        if len(dat2.pid) == 0:
            continue
        dat2 = dat2[["record_time", "record_id", "roi"]]
        dat2 = dat2.sel(ROI_elements=["w", "h"])

        dat2 = dat2.expand_dims(dict(file=[fname1]))
    #     dat2 = dat2.set_coords(dict(file = fname2))
        dat2 = dat2.stack(fpid=("file", "pid"))

        dats2.append(dat2)

    print("opened")
    new_im = Image.new('RGB', (total_width, max_height), color=(255, 255, 255))

    if len(dats2) == 0:

        draw = ImageDraw.Draw(new_im)
        draw.text((total_width//3, max_height//3),
                  'no raw data', (0, 0, 0), font=font)

        nParticles = 0

    else:
        limDat = xr.concat(dats2, dim='fpid')
    #         limDat = dats2
        print("merged")

        if len(limDat.fpid) == 0:
            print("TOO FEW DATA ", ff.quicklook.level1detect)

            draw = ImageDraw.Draw(new_im)
            draw.text((total_width//3, max_height//3),
                      'no data recorded', (0, 0, 0), font=font)

            nParticles = 0

        else:
            print('Total number of particles for plotting %i' %
                  len(limDat.fpid))

            timeSteps = np.percentile(
                limDat.record_time, np.linspace(0, 100, nTiles+1))

            mosaics = []

            videos = {}
            for tt, (t1, t2) in enumerate(zip(timeSteps[:-1], timeSteps[1:])):

                if tt == len(timeSteps) - 2:
                    whereCond = (limDat.record_time >= t1)
                else:
                    whereCond = ((limDat.record_time >= t1) &
                                 (limDat.record_time < t2))

                thisDat = limDat.where(whereCond).dropna('fpid')
                totalArea = 0

                # select pids randomly, figure out how much we need, and sort them again
                pids = deepcopy(thisDat.fpid.values)
                nPids = len(pids)
                np.random.seed(tt)
                np.random.shuffle(pids)

                containerSize = (container_width*container_height_max)

                if nPids < 5:  # fo very few particles I want to see them all!
                    nParticlesNeeded = len(pids)
                else:
                    try:
                        nParticlesNeeded = np.where(thisDat.sel(fpid=pids).roi.sel(ROI_elements=[
                                                    "w", "h"]).prod("ROI_elements").cumsum("fpid")/containerSize > 1)[0][0] + 1
                    except IndexError:
                        nParticlesNeeded = nPids

                pids = np.sort(pids[:nParticlesNeeded])
                print(tt, "/", nTiles, t1, t2, nParticlesNeeded, 'of', nPids)
                particlesPloted += nParticlesNeeded
                ims = []

                for fname, pid in pids:

                    basenameImg = fname.split('/')[-1]

                    if not readParticlesFromFiles:

                        basename = '_'.join(fname.split(
                            '/')[-1].split('.')[-2].split('_')[3:])
                        thisfname_lv0 = fname_lv0.format(root=root, computer=computer, visssGen=visssGen, camera=camera, timestamp=timestamp,
                                                         site=site, year=year, month=month, day=day, nThread='{thread}', basename=basename, movieExtension=movieExtension,)

                        if thisfname_lv0 not in videos.keys():
                            for k in videos.keys():
                                videos[k].release()

                            videos[thisfname_lv0] = VideoReaderMeta(
                                thisfname_lv0, fname)
                #             print('opened %s'%thisfname_lv0)

                    particle = thisDat.sel(fpid=(fname, pid))
                    kk = int(particle.record_id.values)
                    if not readParticlesFromFiles:
                        _, frame1, _ = videos[thisfname_lv0].getFrameByIndex(
                            kk)

                        if frame1 is None:
                            continue

                        x, y, w, h = particle.roi.values.astype(int)
                        if len(frame1.shape) == 3:
                            frame1 = frame1[:, :, 0]
                        im = frame1[y+height_offset:y+height_offset+h, x:x+w]
                    else:

                        fn = files.FilenamesFromLevel(fname, config)

                        pidStr = '%07i' % pid
                        imName = '%s.png' % (pidStr)
                        imfname = '%s/%s' % (
                            fn.imagepath.imagesL1detect.format(ppid=pidStr[:4]), imName)
                        try:
                            im = np.array(Image.open(imfname))
                        except FileNotFoundError:
                            print("NOT FOUND ", imfname)
                            continue

                    im = np.pad(im, [(0, 1), (0, 1)])
                    fid = np.where(fname == np.array(ff.listFiles("metaFrames")))[0][0]

                    text = np.zeros((100, 100))
                    text = cv2.putText(text, '%i.%i' % (fid, pid),
                                       (0, 50), cv2.FONT_HERSHEY_PLAIN, .75, 255, 1,)

                    text = crop(text)

                    y1, x1 = im.shape
                    y2, x2 = text.shape

                    # only add label if large enough
                    if x1 >= x2:

                        y3 = y1+y2
                        x3 = max(x1, x2)
                        imT = np.zeros((y3, x3), dtype=np.uint8)
                        imT[:y1, :x1] = im
                        imT[y1:, :x2] = text
                    else:
                        imT = im
                    ims.append(imT)
                    totalArea += np.prod(imT.shape)

                # make tile
                images = [Image.fromarray(im) for im in ims]
                if len(images) == 0:
                    continue
                mosaic = Packer_patched(images).pack(
                    container_width=container_width, container_height_max=container_height_max)
                mosaic = np.array(mosaic)

                if container_width > mosaic.shape[1]:
                    mosaic = np.pad(
                        mosaic, [(0, 0), (0, container_width-mosaic.shape[1]), (0, 0)])

                # sometimes container is too large...
                mosaic = mosaic[:container_height_max, :container_width]

                label = Image.fromarray(
                    np.ones((20, mosaic.shape[1], 3), dtype=np.uint8) * 255)
                drawL = ImageDraw.Draw(label)
                drawL.text((0, 0), '%s-%s' % (str(t1).split('.')[0].split('T')[1], str(
                    t2).split('.')[0].split('T')[1]), (0, 0, 0), font=fontL)

                mosaic = Image.fromarray(np.vstack((label, mosaic)))
    #             display(mosaic)
                mosaics.append(mosaic)

            nCols = nTiles//nRows

            widths, heights = zip(*(i.size for i in mosaics))

            for nRow in range(nRows):
                x_offset = 0

                for im in mosaics[nCols*(nRow):nCols*(nRow+1)]:
                    new_im.paste(im, (x_offset, max(heights)*nRow + 50))
                    x_offset += im.size[0] + extra

                # x_offset = 0
    # for im in mosaics[len(mosaics)//nRows:]:
    #   new_im.paste(im, (x_offset,max(heights) +50))
    #   x_offset += im.size[0] + extra

            nParticles = len(limDat.fpid)

    tenmm = 1e6/config["resolution"]/100

    title = '%s-%s-%s %s %s, size threshold for plotting: %i px (%.2f mm), %i of %i larger detections plotted, 10 mm = %.1f px =' % (
        ff.year, ff.month, ff.day, nicerNames(camera), config["name"], minSize, minSize * config["resolution"] * 1e-6 * 1000, particlesPloted, nParticles, tenmm)

    # new_im = cv2.putText(np.array(new_im), title,
    #                      (0, 45), FONT, FONT_SCALE, (0, 0, 0), FONT_THICKNESS,)
    # (label_width, label_height), baseline = cv2.getTextSize(
    #     title, FONT, FONT_SCALE, FONT_THICKNESS)

    draw = ImageDraw.Draw(new_im)
    draw.text((0, 0), title, (0, 0, 0), font=font)
    width, height = draw.textsize(title, font=font)

    draw.line((width + 15, 30, width +
               15+round(tenmm), 30), fill=0, width=5)

    new_im.save(ff.quicklook.level1detect)

    return ff.quicklook.level1detect, new_im


class Packer_patched(packer.Packer):
    """
    patched image_packer routine that works without files
    """

    def __init__(self, images):
        # Ensure plugins are fully loaded so that Image.EXTENSION is populated.
        Image.init()

        self._uid_to_filepath = dict()
        self._pieces = list()
        self._has_alpha = False

        for im in images:
            width = im.width
            height = im.height
            uid = uuid.uuid4()
            self._uid_to_filepath[uid] = deepcopy(im)
            self._pieces.append(packer.blf.Piece(
                uid=uid, size=packer.blf.Size(width, height)))
            if im.mode in ('RGBA', 'LA') or (im.mode == 'P' and 'transparency' in im.info):
                self._has_alpha = True

    def pack(self, container_width, options=None, container_height_max=100):
        '''Packs multiple images of different sizes or formats into one image.
        Args:
            container_width (int):
            options (dict):
        '''
        if options is None:
            options = self._DEFAULT_OPTIONS
        else:
            options = {
                key: options[key] if key in options else self._DEFAULT_OPTIONS[key]
                for key in self._DEFAULT_OPTIONS.keys()
            }

        margin_ = options['margin']
        assert isinstance(margin_, tuple) and len(margin_) == 4

        if options['enable_vertical_flip']:
            margin = packer.blf.Thickness(top=margin_[2], right=margin_[
                                          1], bottom=margin_[0], left=margin_[3])
        else:
            margin = packer.blf.Thickness(top=margin_[0], right=margin_[
                                          1], bottom=margin_[2], left=margin_[3])

        blf_options = {
            'margin': margin,
            'collapse_margin': options['collapse_margin'],
            'enable_auto_size': options['enable_auto_size'],
            'force_pow2': options['force_pow2']
        }

        container_width, container_height, regions = packer.blf_solver.solve(
            pieces=self._pieces,
            container_width=container_width,
            options=blf_options
        )

        compImage = self._save_image(
            container_width=container_width,
            container_height=container_height_max,
            regions=regions,
            options=options
        )
        return compImage

    def _save_image(
        self,
        container_width,
        container_height,
        regions,
        options
    ):
        bg_color_ = options['bg_color']
        assert isinstance(bg_color_, tuple) and (3 <= len(bg_color_) <= 4)
        bg_color = tuple(int(channel * 255.0) for channel in bg_color_)
        if len(bg_color) == 3:
            bg_color += (255,)

        if self._has_alpha:
            blank_image = Image.new(
                mode='RGBA',
                size=(container_width, container_height),
                color=bg_color
            )
        else:
            blank_image = Image.new(
                mode='RGB',
                size=(container_width, container_height),
                color=bg_color[0:3]
            )

        enable_vertical_flip = options['enable_vertical_flip']

        for region in regions:
            x = region.left
            if enable_vertical_flip:
                y = region.bottom
            else:
                y = container_height - region.top

            im = self._uid_to_filepath.get(region.uid)

            blank_image.paste(im=im, box=(x, y))

        return blank_image


def createMetaCoefQuicklook(case, config, version=__version__, skipExisting=True):
    '''
    Quicklooks of the coefficients obtained for matching in level3
    '''

    version = deepcopy(version)
    versionOld = version[:8]

    if type(config) is str:
        config = readSettings(config)

    leader = config["leader"]
    follower = config["follower"]

    fn = files.FindFiles(case, leader, config, version)
    ff = files.FindFiles(case, follower, config, version)
    fnOld = files.FindFiles(case, leader, config, versionOld)
    ffOld = files.FindFiles(case, follower, config, versionOld)

    outFile = fn.quicklook.matchCoefficients

    if os.path.isfile(outFile) and skipExisting:
        print(f"skip {case} ")
        return None, None

    coefFiles = fn.listFiles("metaMatchCoefficients")
    if len(coefFiles) == 0:
        print(f"no data {case} {fn.fnamesPattern.metaMatchCoefficients}")
        return None, None

    print(f"running {case}")

    dat3 = xr.open_mfdataset(coefFiles, concat_dim="file_starttime")
    print("VERSION HACK")

    fig, (bx1, bx2, bx3, bx4) = plt.subplots(
        figsize=(10, 10), nrows=4, sharex=True)

    dat3Stats = dat3.isel(iteration=1)

    bx1.plot(dat3Stats.file_firsttime,
             dat3Stats.usedSamples, marker=".", ls="None")
    bx1.set_yscale("log")
    bx1.set_ylabel("Used matches  [#]")

    bx2.fill_between(dat3Stats.file_firsttime, (dat3Stats.muH + dat3Stats.sigmaH),
                     (dat3Stats.muH - dat3Stats.sigmaH), facecolor='C1', alpha=0.4)
    bx2.plot(dat3Stats.file_firsttime, dat3Stats.muH,
             marker=".", ls="None", color="C1")
    bx2.set_ylabel("Height difference [px]")
    bx2.axhline(0, color="gray")

    bx3.fill_between(dat3Stats.file_firsttime, (dat3Stats.muY + dat3Stats.sigmaY),
                     (dat3Stats.muY - dat3Stats.sigmaY), facecolor='C2', alpha=0.4)
    bx3.plot(dat3Stats.file_firsttime, dat3Stats.muY,
             marker=".", ls="None", color="C2")
    bx3.set_ylabel("Y position difference [px]")

    bx4.fill_between(dat3Stats.file_firsttime, (dat3Stats.muT + dat3Stats.sigmaT)
                     * 1000, (dat3Stats.muT - dat3Stats.sigmaT)*1000, facecolor='C3', alpha=0.4)
    bx4.plot(dat3Stats.file_firsttime, dat3Stats.muT *
             1000, marker=".", ls="None", color="C3")
    bx4.set_ylabel("Time difference [ms]")
    # bx4.axhline(1/config["fps"], color= "gray")
    # bx4.axhline(-1/config["fps"], color= "gray")
    bx4.axhline(0, color="gray")


    tStart = dat3Stats.file_firsttime.to_pandas().index[0].replace(hour=0, minute=0, second=0)
    tEnd = dat3Stats.file_firsttime.to_pandas().index[-1].replace(hour=23, minute=59, second=59)
    
    eventDatL = xr.open_dataset(fnOld.listFiles("metaEvents")[0])
    for event in eventDatL.event:
        if str(event.values).startswith("start"):
            for bx in [bx1,bx2,bx3,bx4]:
                bx.axvline(event.file_starttime.values, color="r")
    lBlocked = (eventDatL.blocking.sel(blockingThreshold=50) > 0.1)
    lBlocked = lBlocked.file_starttime.where(lBlocked).values
    for bx in [bx1,bx2,bx3,bx4]:
        ylim = bx.get_ylim() 
        bx.fill_between(lBlocked, [ylim[0]]*len(lBlocked), [ylim[1]]*len(lBlocked), color="red", alpha=0.25, label="Leader")
        bx.set_ylim(ylim) 

    eventDatF = xr.open_dataset(ffOld.listFiles("metaEvents")[0])
    for event in eventDatF.event:
        if str(event.values).startswith("start"):
            for bx in [bx1,bx2,bx3,bx4]:
                bx.axvline(event.file_starttime.values, color="blue")
    fBlocked = (eventDatF.blocking.sel(blockingThreshold=50) > 0.1)
    fBlocked = fBlocked.file_starttime.where(fBlocked).values
    for bx in [bx1,bx2,bx3,bx4]:
        ylim = bx.get_ylim() 
        bx.fill_between(fBlocked, [ylim[0]]*len(fBlocked), [ylim[1]]*len(fBlocked), color="blue", alpha=0.25, label="Follower")
        bx.set_ylim(ylim) 

    bx1.legend() 
    bx1.grid(True)
    bx2.grid(True)
    bx3.grid(True)
    bx4.grid(True)


    bx4.set_xlim(
        tStart, 
        tEnd,
    )

    fig.suptitle(f"{fn.year}-{fn.month}-{fn.day}")
    fig.tight_layout()

    fn.createQuicklookDirs()
    print(outFile)
    fig.savefig(outFile)
    return outFile, fig


def metaFramesQuicklook(
                case, camera, config, version=__version__, skipExisting=True):

    """
    Anja Stallmach 2022

    """

    if type(config) is str:
        config = readSettings(config)
        

    global metaDats
    metaDats = []

    camera_new= camera.split('_')[0]+'-'+camera.split('_')[1]


    #get level 0 file names
    ff = files.FindFiles(case, camera, config, version)
    fOut = ff.quicklook.metaFrames


    if skipExisting and os.path.isfile(fOut):
        print(case, camera, "skip exisiting")
        return None, None


    print(case, camera, fOut)

    if len(ff.listFiles("level0")) == 0 and len(ff.listFiles("level0status")) == 0:
        print(case, "no data")

        return None, None

    try:
        level0status = ff.listFiles("level0status")[0]
    except IndexError:
        level0status = None

    print("reading events")
    if len(ff.listFiles("metaEvents")) == 0:
        print(f'event data not found')
        return None, None
    try:
        events = xr.open_dataset(ff.listFiles("metaEvents")[0])
    except :
        print(f'{ff.listFiles("metaEvents")[0]} broken')
        return None, None

    if not ff.isCompleteMetaFrames:
        print(f'meta frames not complete')
        return None, None

    #iterate to get meta data
    # if len(ff.listFiles("metaFrames")) > 0:
    #     print("reading metaFrames")
        # try: #faster?
        #     metaDats = xr.open_mfdataset(ff.listFiles("metaFrames"), combine="by_coords")
        # except ValueError:
        #     metaDats = xr.open_mfdataset(ff.listFiles("metaFrames"), combine="nested")

    # else:
    #     print(f'metaFrames not found')
    #     return None, None
    print("reading metaFrames")
    metaDats = []
    for fname1 in ff.listFiles("metaFrames"):
        metaDat = xr.open_dataset(fname1)
        keys = [ "capture_time"]
        if "queue_size" in metaDat.variables:
            keys.append("queue_size")
        else:
            keys.append("record_time")
        metaDat = metaDat[keys]
        metaDats.append(deepcopy(metaDat))
        metaDat.close()
    if len(metaDats) == 0:
        metaDats = None
    else:
        metaDats = xr.concat(metaDats, dim='capture_time')
        metaDats = metaDats.sortby('capture_time')

    ts =  events.file_starttime.where((events.event == 'sleep-trigger') | (events.event == 'stop-trigger'))

    print("plotting")
    #plotting
    fig, (ax1, ax2, ax3) = plt.subplots(3, figsize=(20,15), gridspec_kw={'hspace':0.3})
    # plt.rcParams['text.usetex'] = False
    # plt.rcParams['lines.linewidth'] = 1.5
    mid = (fig.subplotpars.right + fig.subplotpars.left)/2
    plt.suptitle('VISSS Status-Plot \n'+f'{ff.year}-{ff.month}-{ff.day}'+', '+config["name"]+'', fontsize=25, y=0.995, fontweight='bold', x=mid)
    fig.text(mid, 0.07, 'time', fontsize=20, ha='center')

    
    if metaDats is not None:
        if "queue_size" in keys:
            queue=metaDats.queue_size
            ax1.plot(metaDats.capture_time, queue)
            ax1.set_ylabel('frames in queue', fontsize=20)
        else:
            delay=(metaDats.record_time-metaDats.capture_time).astype(int)/1e9
            ax1.plot(metaDats.capture_time, delay)
            ax1.set_ylabel('record delay [s]', fontsize=20)

    ylim = ax1.get_ylim() 
    ax1.fill_between(ts, [ylim[0]]*len(ts), [ylim[1]]*len(ts), color="orange", alpha=0.5, label="idle")
    ax1.set_title(r"$\bf{" + str(camera_new) + "}$""\n""queue size", fontsize=20)
    ax1.set_ylim(ylim)
    ax1.set_xlim(np.datetime64(f'{ff.year}-{ff.month}-{ff.day}'+'T00:00'), np.datetime64(f'{ff.year}-{ff.month}-{ff.day}'+'T00:00')+np.timedelta64(1, 'D'))
    ax1.tick_params(axis='both', labelsize=15)
    ax1.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    ax1.grid()

    if metaDats is not None:
        frames=metaDats.capture_time.notnull().resample(capture_time="1T").sum()
        ax2.plot(frames.capture_time, frames)
    ylim = [0,config.fps*60*1.1]
    ax2.fill_between(ts, [ylim[0]]*len(ts), [ylim[1]]*len(ts), color="orange", alpha=0.5, label="idle")
    ax2.set_ylim(ylim)
    ax2.set_xlim(np.datetime64(f'{ff.year}-{ff.month}-{ff.day}'+'T00:00'), np.datetime64(f'{ff.year}-{ff.month}-{ff.day}'+'T00:00')+np.timedelta64(1, 'D'))
    ax2.set_title('total frames / min', fontsize=20)
    ax2.set_ylabel('frames', fontsize=20)
    ax2.tick_params(axis='both', labelsize=15)
    ax2.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
    ax2.grid()

    if len(events.blocking.isel(blockingThreshold=[2, 4, 6, 8, 10]).dropna("file_starttime")) != 0:
        events.blocking.isel(blockingThreshold=[2, 4, 6, 8, 10]).dropna("file_starttime").plot.line(x="file_starttime")
        ylim = [0,1]
        ax3.fill_between(ts, [ylim[0]]*len(ts), [ylim[1]]*len(ts), color="orange", alpha=0.5)
        ax3.set_ylim(ylim)
        ax3.set_xlim(np.datetime64(f'{ff.year}-{ff.month}-{ff.day}'+'T00:00'), np.datetime64(f'{ff.year}-{ff.month}-{ff.day}'+'T00:00')+np.timedelta64(1, 'D'))
        ax3.set_title('blocking', fontsize=20)
        ax3.set_ylabel('pixels below \n blocking threshold', fontsize=20)
        ax3.set_xlabel('')
        plt.setp(ax3.get_xticklabels(), rotation=False, ha="center", rotation_mode="anchor")
        ax3.tick_params(axis='both', labelsize=15)
        ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
        ax3.grid()

    else: 
        ax3.fill_between(ts, [ylim[0]]*len(ts), [ylim[1]]*len(ts), color="orange", alpha=0.5)
        ax3.set_ylim(0,1)
        ax3.set_xlim(np.datetime64(f'{ff.year}-{ff.month}-{ff.day}'+'T00:00'), np.datetime64(f'{ff.year}-{ff.month}-{ff.day}'+'T00:00')+np.timedelta64(1, 'D'))
        ax3.set_title('blocking', fontsize=20)
        ax3.set_ylabel('pixels below \n blocking threshold', fontsize=20)
        ax3.set_xlabel('')
        ax3.tick_params(axis='both', labelsize=15)
        ax3.xaxis.set_major_formatter(mpl.dates.DateFormatter("%H:%M"))
        ax3.grid()

    firstEvent = True
    for event in events.event:
        if str(event.values).startswith("start"):
            for bx in [ax1,ax2,ax3]:
                if firstEvent:
                    label="restarted"
                    firstEvent = False
                else:
                    label = None
                bx.axvline(event.file_starttime.values, color="red", ls=":", label=label)

    ax1.legend(fontsize=15, bbox_to_anchor=(1, 1.3))

    ff.createQuicklookDirs()
    fig.savefig(fOut)

    if metaDats is not None: metaDats.close()
    events.close()
    
    return fOut, fig