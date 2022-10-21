import sys
import os
import socket
import datetime
import multiprocessing
import subprocess
import collections
import logging
import shlex
from functools import partial

import numpy as np
import pandas as pd
import xarray as xr


from . import quicklooks
from . import files
from . import tools
from . import metadata
from . import matching
from . import __version__



def loopMetaFramesQuicklooks(settings, version=__version__, skipExisting=True, nDays=0):

    config = tools.readSettings(settings)
    instruments = config["instruments"]
    computers = config["computers"]    
    
    days = tools.getDateRange(nDays, config, endYesterday=False)


    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"
        print(case)
        for computer, camera in zip(computers, instruments):
            quicklooks.metaFramesQuicklook(case, camera, config, version=version, skipExisting=skipExisting)


def loopLevel1detectQuicklooks(settings, version=__version__, nDays = 0, skipExisting=True):

    config = tools.readSettings(settings)
    instruments = config["instruments"]
    computers = config["computers"]

    days = tools.getDateRange(nDays, config)

    for dd in days:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for computer, camera in zip(computers, instruments):
            #         print(case, computer, camera)

            fname, fig = quicklooks.createLevel1detectQuicklook(
                case, 
                camera, 
                config, 
                version=version, 
                skipExisting=skipExisting,
                )
            try:
                fig.close()
            except AttributeError:
                pass
    return


def loopLevel1matchQuicklooks(settings, version=__version__, nDays = 0, skipExisting=True, plotCompleteOnly=True):

    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config)

    for dd in days:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        #         print(case, computer, camera)

        fname, fig = quicklooks.createLevel1matchQuicklook(
            case, 
            config, 
            version=version, 
            skipExisting=skipExisting,
            plotCompleteOnly=plotCompleteOnly,
            )
        try:
            fig.close()
        except AttributeError:
            pass
    return



def loopMetaCoefQuicklooks(settings, version=__version__, skipExisting=True):

    config = tools.readSettings(settings)

    if config["end"] == "today":
        end = datetime.datetime.utcnow() - datetime.timedelta(days=1)
    else:
        end = config["end"]

    for dd in pd.date_range(
        start=config["start"],
        end=end,
        freq="1D",
        tz=None,
        normalize=True,
        name=None,
        inclusive=None
    ):

        # , periods=nDays´´

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        fname, fig = quicklooks.createMetaCoefQuicklook(
            case, config, version=version, skipExisting=skipExisting)
        try:
            fig.close()
        except AttributeError:
            pass


def loopCreateEvents(settings, skipExisting=True, nDays = 0):


    config = tools.readSettings(settings)


    days = tools.getDateRange(nDays, config, endYesterday=False)

    for dd in days:

        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for camera in config.instruments:
            metadata.createEvent(case, camera, config, skipExisting=True)


def loopCreateMetaFrames(settings, skipExisting=True, nDays = 0, cameras = "all"):

    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config, endYesterday=False)

    if cameras == "all":
        cameras = [config.follower, config.leader]

    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for camera in cameras:

            metadata.createMetaFrames(case, camera, config, skipExisting=skipExisting)


    return




def loopCreateLevel1detectWorker(fname, settings, skipExisting=True, stdout=subprocess.DEVNULL, nCPU=1):

    '''
    We need this worker function becuase  we want to do the detection by indiependent 
    processes in parallel
    '''
    config = tools.readSettings(settings)
    logging.config.dictConfig(tools.get_logging_config('detection_run.log'))
    log = logging.getLogger()

    errorlines = collections.deque([], 50)
    
            
    fn = files.Filenames(fname, config, __version__)
    camera = fn.camera
    
    fn.createDirs()
    if skipExisting and ( os.path.isfile('%s' % fn.fname.level1detect)):
        # log.info('output exists %s %s' % (fname, fn.fname.level1detect))
        return 0
    elif skipExisting and ( os.path.isfile('%s.nodata' % fn.fname.level1detect)):
        # log.info('output exists %s %s' % (fname, fn.fname.level1detect))
        return 0
    elif skipExisting and (os.path.isfile('%s.broken.txt' % fn.fname.level1detect)):
        log.error('output broken %s %s' % (fname, fn.fname.level1detect))
        return 1
    elif skipExisting and ( os.path.isfile(os.path.basename('%s.processing.txt' % fn.fname.level1detect))):
        log.info('output processing %s %s' % (fname, fn.fname.level1detect))
        return 0
    elif skipExisting and ( os.path.isfile('%s.nodata' % fn.fname.metaFrames)):
        log.info('metaFrames contains no data %s %s' % (fname, fn.fname.metaFrames))
        with open('%s.nodata' % fn.fname.level1detect, 'w') as f:
            f.write('metaFrames contains no data %s %s' % (fname, fn.fname.metaFrames))
        return 0
    else:
        pass


    tmpFile = os.path.basename('%s.processing.txt' % fn.fname.level1detect)
    with open(tmpFile, 'w') as f:
        f.write('%i %s' % (os.getpid(), socket.gethostname()))
        print("written", tmpFile, "in", os.getcwd())
        
    BIN = os.path.join(sys.exec_prefix, "bin", "python")

    if nCPU is None:
        command = f"{BIN} -m VISSSlib detection.detectParticles  {fname} {settings}"
    else:
        command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {BIN} -m VISSSlib detection.detectParticles  {fname} {settings}"

    log.info(command)
    # sleep to avoid py38 env not found error
    with subprocess.Popen(shlex.split(f'bash -c "{command}"'), stderr=subprocess.PIPE, stdout=stdout, universal_newlines=True) as proc:
        try:
            for line in proc.stdout:
                print(str(line),end='')
        except TypeError:
            pass
        for line in proc.stderr:
            print(str(line),end='')
            errorlines.append(line)
        proc.wait() # required, otherwise returncode can be none is process is too fast
        if proc.returncode != 0:
            with open('%s.broken.txt' % fn.fname.level1detect, 'w') as f:
                
                for errorline in list(errorlines):
                    f.write(errorline)
                f.write("\r")
                f.write(command)
            log.info(f"{fname} BROKEN {proc.returncode}")
        else:
            log.info(f"{fname} SUCCESS {proc.returncode}")
    try:
        os.remove(os.path.basename('%s.processing.txt' % fn.fname.level1detect))
    except:
        pass
    
    
    return 0




def loopCreateLevel1detect(settings, skipExisting=True, nDays = 0, cameras = "all", nCPU=None):
    config = tools.readSettings(settings)

    logging.config.dictConfig(tools.get_logging_config('detection_run.log'))
    log = logging.getLogger()


    days = tools.getDateRange(nDays, config)
    if cameras == "all":
        cameras = [config.follower, config.leader]

    if nCPU is None:
        nCPU = os.cpu_count()

    fnames = []

    for dd in days:
        # create list of files
        year  =str(dd.year)
        month  ="%02i"%(dd.month)
        day  ="%02i"%(dd.day)      
        case = f"{year}{month}{day}"
        for camera in cameras:
            # find files
            ff = files.FindFiles(case, camera, config)
            fname0s =  ff.listFiles("level0txt")
            fnames += filter( os.path.isfile,
                                    fname0s )

        # Sort list of files in directory by size 
        fnames = sorted( fnames,
                                key =  lambda x: os.stat(x).st_size)[::-1]
    if len(fnames) == 0:
        log.error('no files %s' % fnamesPattern)
        # print('no files %s' % fnamesPattern)

    else:
        
        print(f"found {len(fnames)} files, lets do it:")
        

        p = multiprocessing.Pool(nCPU)
        p.map(partial(loopCreateLevel1detectWorker, settings=settings, skipExisting=skipExisting), fnames)


        p.close()
        p.join()

    return p


def loopCreateLevel1match(settings, skipExisting=True, nDays = 0, version=__version__, useWorker=False, nCPU=None):

    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config)
    leader = config["leader"]
    follower = config["follower"]
    
    if useWorker:
        assert version == __version__
        if nCPU is None:
            nCPU = os.cpu_count()
        p = multiprocessing.Pool(nCPU)
        days = [str(day).split(" ")[0].replace("-","") for day in days]
        p.map(partial(loopCreateLevel1matchWorker, settings=settings, skipExisting=skipExisting), days)
        p.close()
        p.join()


    else:
        for dd in days:
            year = str(dd.year)
            month = "%02i" % dd.month
            day = "%02i" % dd.day
            case = f"{year}{month}{day}"

            matching.createLevel1match(case, config, skipExisting=skipExisting, version=version, )

        return

def loopCreateLevel1matchWorker(day, settings=None, skipExisting=True, nCPU=1, stdout=subprocess.DEVNULL):
    logging.config.dictConfig(tools.get_logging_config('match_run.log'))
    log = logging.getLogger()

    BIN = os.path.join(sys.exec_prefix, "bin", "python")

    if nCPU is None:
        command = f"{BIN} -m VISSSlib scripts.loopCreateLevel1match   {settings} {day}"
    else:
        command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {BIN} -m VISSSlib scripts.loopCreateLevel1match   {settings} {day}"

    log.info(command)
    errorlines = collections.deque([], 50)

    with subprocess.Popen(shlex.split(f'bash -c "{command}"'), stderr=subprocess.PIPE, stdout=stdout, universal_newlines=True) as proc:
        try:
            for line in proc.stdout:
                print(str(line),end='')
        except TypeError:
            pass
        for line in proc.stderr:
            print(str(line),end='')
            errorlines.append(line)
        proc.wait() # required, otherwise returncode can be none is process is too fast
        if proc.returncode != 0:

            log.info(f"{day} BROKEN {proc.returncode}")
        else:
            log.info(f"{day} SUCCESS {proc.returncode}")

        log.info(errorlines)

def loopCheckCompleteness(settings, nDays = 0, cameras = "all", checkDuplicates=True, checkMissing=True):

    config = tools.readSettings(settings)

    days = tools.getDateRange(nDays, config)

    if cameras == "all":
        cameras = [config.follower, config.leader]

    for dd in days:
        year = str(dd.year)
        month = "%02i" % dd.month
        day = "%02i" % dd.day
        case = f"{year}{month}{day}"

        for camera in cameras:
            # find files
            ff = files.FindFiles(case, camera, config)
            mf = ff.isCompleteMetaFrames
            l1d = ff.isCompleteL1detect
            l1m = ff.isCompleteL1match
            products = ["metaFrames", "level1detect", "level1match"]

            if camera == config.leader:
                allDone = [mf, l1d, l1m]
            else:
                allDone = [mf, l1d]
            if np.all(allDone):
                print(camera, case, "all done", np.all(allDone))
            else:
                print(camera, case, "MISSING", allDone)

                firstMiss = products[np.where(np.array(allDone) == False)[0][0]]
                recFiles = np.array(ff.listFiles("level0txt"))
                nRec = len(recFiles)
                procFiles = np.array(ff.listFilesExt(firstMiss))
                nProc = len(procFiles)
                print("# level0", nRec , "#", firstMiss, nProc)

                processedTimes = np.array([files.FilenamesFromLevel(f, config).datetime64 for f in ff.listFilesExt(firstMiss)])


                if checkDuplicates and nProc > nRec:
                    print("too many files processed, check these files:")
                    print("*"*50)
                    seen = set()
                    dupes = [x for x in processedTimes if x in seen or seen.add(x)]    
                    dupeFiles = []
                    for dupe in dupes:
                        dupeFiles.append(procFiles[(dupe == processedTimes)])
                    dupeFiles = np.concatenate(dupeFiles)
                    for dupeFile in dupeFiles: print(dupeFile)
                elif checkMissing:
                    print("files missing")
                    print("*"*50)

                    recTimes = [files.Filenames(f, config).datetime64 for f in ff.listFiles("level0txt")]
                    missingTimes = set(recTimes).difference( set(processedTimes))
                    for missingTime in missingTimes:
                        print(camera, firstMiss, missingTime)
    return
