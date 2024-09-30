import logging
import os
import sys
from functools import partial

import numpy as np
import pandas as pd
import taskqueue
import xarray as xr

from . import __version__, files, scripts, tools
from .tools import runCommandInQueue

log = logging.getLogger(__name__)


class DataProduct(object):
    def __init__(
        self,
        level,
        case,
        settings,
        fileQueue,
        camera,
        relatives="",
        addRelatives=True,
    ):
        """
        Class for processing VISSS data

        """
        log.info(f"create object for level {level}")
        self.level = level
        self.config = tools.readSettings(settings)
        self.settings = settings
        self.relatives = f"{relatives}.{level}"

        if camera == "leader":
            self.cameraFull = self.config.leader
        elif camera == "follower":
            self.cameraFull = self.config.follower
        else:
            raise ValueError(f"do not understand camera: {camera}")
        self.camera = camera
        self.case = case
        self.fileQueue = fileQueue

        self.tq = taskqueue.TaskQueue(f"fq://{self.fileQueue}")
        self.commands = []

        self.fn = files.FindFiles(self.case, self.cameraFull, self.config)
        self.parents = tools.DictNoDefault({})

        if self.level == "level0":
            self.parentNames = []
        elif level == "metaEvents":
            self.parentNames = [f"{camera}_level0"]
        elif level == "metaFrames":
            self.parentNames = [f"{camera}_level0"]
        elif level == "level1detect":
            self.parentNames = [
                f"{camera}_metaFrames",
                f"{camera}_metaEvents",
            ]
        elif level == "metaRotation":
            self.parentNames = [
                f"leader_level1detect",
                f"follower_level1detect",
            ]
        elif level == "level1match":
            self.parentNames = [f"{camera}_metaRotation"]
        elif level == "level1track":
            self.parentNames = [f"{camera}_level1match"]
        elif level == "level2detect":
            self.parentNames = [f"{camera}_level1detect"]
        elif level == "level2match":
            self.parentNames = [f"{camera}_level1match"]
        elif level == "level2track":
            self.parentNames = [f"{camera}_level1track"]
        else:
            raise ValueError(f"Do not understand {level}")

        if addRelatives:
            self.addRelatives()

    def addRelatives(self):
        for parentCam in self.parentNames:
            camera, parent = parentCam.split("_")
            self.parents[parentCam] = DataProduct(
                parent,
                self.case,
                self.settings,
                self.fileQueue,
                camera,
                relatives=f"{self.relatives}",
            )

            self.parents.update(self.parents[parentCam].parents)

    def generateAllCommands(self, skipExisting=True, withParents=True):
        if skipExisting and self.allComplete:
            log.info(f"{self.relatives} skip all existing")
            return []
        if self.parentsComplete:
            log.info(f"{self.relatives} generate commands")
            commands = self.generateCommands(skipExisting=skipExisting)
        else:
            log.warning(f"{self.case} {self.relatives} Parents not ready yet")
            commands = []
        if withParents:
            for parent in self.parents.keys():
                log.info(
                    f"{self.relatives} generate commands of parent {self.parents[parent].level}",
                )
                commands = commands + self.parents[parent].generateAllCommands(
                    skipExisting=skipExisting, withParents=False
                )
        self.commands = list(set(commands))
        if len(self.commands) == 0:
            log.warning(
                f"{self.level} {self.case} all done",
            )
        return self.commands

    def generateCommands(self, skipExisting=True, nCPU=1, bin=None):
        if self.level == "level0":
            return []
        elif self.level == "metaEvents":
            return self.commandTemplateDaily(
                "metadata.createEvent", skipExisting=skipExisting, nCPU=nCPU, bin=bin
            )
        elif self.level == "metaFrames":
            return self.commandTemplateDaily(
                "scripts.loopCreateMetaFrames",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level1detect":
            originLevel = "level0txt"
            call = "detection.detectParticles"
            return self.commandTemplateL1(
                originLevel, call, skipExisting=skipExisting, nCPU=nCPU, bin=bin
            )
        elif self.level == "metaRotation":
            return self.commandTemplateDaily(
                "matching.createMetaRotation",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level1match":
            originLevel = "level1detect"
            call = "matching.matchParticles"
            return self.commandTemplateL1(
                originLevel, call, skipExisting=skipExisting, nCPU=nCPU, bin=bin
            )
        elif self.level == "level1track":
            originLevel = "level1detect"
            call = "tracking.trackParticles"
            return self.commandTemplateL1(
                originLevel, call, skipExisting=skipExisting, nCPU=nCPU, bin=bin
            )
        elif self.level == "level2detect":
            return self.commandTemplateDaily(
                "distributions.createLevel2detect",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level2match":
            return self.commandTemplateDaily(
                "distributions.createLevel2match",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level2track":
            return self.commandTemplateDaily(
                "distributions.createLevel2track",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        else:
            raise ValueError(f"Do not understand {level}")

    def commandTemplateL1(self, originLevel, call, skipExisting=True, nCPU=1, bin=None):
        nCPU = 1
        skipExisitingInt = int(skipExisting)
        if bin is None:
            bin = os.path.join(sys.exec_prefix, "bin", "python")
        commands = []
        for fname in self.fn.listFiles(originLevel):
            if originLevel.startswith("level0"):
                f1 = files.Filenames(fname, self.config)
            else:
                f1 = files.FilenamesFromLevel(fname, self.config)
            outFile = f1.fname[self.level]
            if skipExisting:
                if os.path.isfile(outFile):
                    log.info(f"{self.relatives} skip exisiting {outFile}")
                    continue
                elif os.path.isfile(f"{outFile}.broken.txt"):
                    log.info(f"{self.relatives} skip broken {outFile}.broken.txt")
                    continue
                elif os.path.isfile(f"{outFile}.nodata.txt"):
                    log.info(f"{self.relatives} skip nodata {outFile}.nodata.txt")
                    continue
            command = f"{bin} -m VISSSlib {call}  {fname} {self.settings}"
            if nCPU is not None:
                command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {command}"
            commands.append((command, outFile))
        return commands

    def commandTemplateDaily(self, call, skipExisting=True, nCPU=1, bin=None):
        nCPU = 1
        skipExisitingInt = int(skipExisting)
        if bin is None:
            bin = os.path.join(sys.exec_prefix, "bin", "python")
        if (
            call.endswith("detect")
            or call.endswith("MetaFrames")
            or call.endswith("createEvent")
        ):
            case = f"{self.camera}+{self.case}"
        else:
            case = self.case

        outFile = self.fn.fnamesDaily[self.level]

        if skipExisting:
            if os.path.isfile(outFile):
                log.info(f"{self.relatives} skip exisiting {outFile}")
                return []
            elif os.path.isfile(f"{outFile}.broken.txt"):
                log.info(f"{self.relatives} skip broken {outFile}.broken.txt")
                return []
            elif os.path.isfile(f"{outFile}.nodata.txt"):
                log.info(f"{self.relatives} skip nodata {outFile}.nodata.txt")
                return []

        command = f"{bin} -m VISSSlib {call} {self.settings} {case} {skipExisitingInt}"
        if nCPU is not None:
            command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {command}"
        return [(command, outFile)]

    def submitCommands(
        self, skipExisting=True, checkForDuplicates=False, withParents=True
    ):
        if len(self.commands) == 0:
            self.generateAllCommands(skipExisting=skipExisting, withParents=withParents)

        if len(self.allCommands) == 0:
            log.error("nothing to submit")
            return

        if checkForDuplicates:
            running = [t.args for t in self.tq.tasks()]
            commands = []
            for command in self.commands:
                if [command] in running:
                    continue
                else:
                    commands.append(command)
        else:
            commands = self.commands

        log.warning(f"sending {len(commands)} commands to {self.fileQueue}")
        # region is SQS specific, green means cooperative threading

        self.tq.insert([partial(runCommandInQueue, c) for c in commands])
        log.warning(f"{self.tq.enqueued} tasks in Queue")
        return

    def deleteQueue(self):
        log.info(f"Deleting {self.tq.enqueued} tasks")
        [self.tq.delete(t) for t in self.tq.tasks()]
        return

    @property
    def isComplete(self):
        return self.fn.isComplete(self.level)

    @property
    def parentsComplete(self):
        parentsComplete = True
        for parent in self.parents.keys():
            log.info(
                f"{self.relatives} {parent} parentsComplete {self.parents[parent].isComplete}",
            )
            parentsComplete = parentsComplete and self.parents[parent].isComplete
        return parentsComplete

    @property
    def allComplete(self):
        return self.isComplete and self.parentsComplete

    @property
    def nFiles():
        return len(self.fn.listFilesExt(self.level))

    def listFilesExt():
        return self.fn.listFilesExt(self.level)

    def listFiles():
        return self.fn.listFiles(self.level)


class level2track(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("level2track", case, settings, fileQueue, camera)


class level2match(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("level2match", case, settings, fileQueue, camera)


class level2detect(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("level2detect", case, settings, fileQueue, camera)


class level1track(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("level1track", case, settings, fileQueue, camera)


class level1match(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("level1match", case, settings, fileQueue, camera)


class metaRotation(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("metaRotation", case, settings, fileQueue, camera)


class level1detect(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("level1detect", case, settings, fileQueue, camera)


class metaFrames(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("metaFrames", case, settings, fileQueue, camera)


class metaEvents(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("metaEvents", case, settings, fileQueue, camera)


class level0(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("level0", case, settings, fileQueue, camera)


class DataProductRange(object):
    def __init__(
        self,
        level,
        nDays,
        settings,
        fileQueue,
        camera="leader",
        addRelatives=True,
    ):
        self.settings = settings
        self.config = tools.readSettings(settings)
        self.days = tools.getDateRange(nDays, self.config, endYesterday=False)
        self.dailies = {}
        self.level = level
        self.camera = camera
        for dd in self.days:
            case = tools.timestamp2case(dd)
            self.dailies[dd] = DataProduct(
                level,
                case,
                settings,
                fileQueue,
                camera,
                addRelatives=addRelatives,
            )
        self.fileQueue = fileQueue
        self.tq = taskqueue.TaskQueue(f"fq://{self.fileQueue}")

    def generateAllCommands(self, skipExisting=True, withParents=True):
        self.allCommands = []
        for dd in self.days:
            self.allCommands += self.dailies[dd].generateAllCommands(
                skipExisting=skipExisting, withParents=withParents
            )
        return self.allCommands

    def submitCommands(
        self, skipExisting=True, checkForDuplicates=False, withParents=True
    ):
        self.generateAllCommands(skipExisting=skipExisting, withParents=withParents)
        if len(self.allCommands) == 0:
            log.error("nothing to submit")
            return

        if checkForDuplicates:
            running = [t.args for t in self.tq.tasks()]
            commands = []
            for command in self.allCommands:
                if [command] in running:
                    continue
                else:
                    commands.append(command)
        else:
            commands = self.allCommands
        if len(commands) == 0:
            log.error("nothing to submit after checking for duplicates")
            return

        log.warning(f"adding {len(commands)} commands to {self.fileQueue}")
        # region is SQS specific, green means cooperative threading
        self.tq.insert([partial(runCommandInQueue, c) for c in commands])
        log.warning(f"{self.tq.enqueued} tasks in Queue")
        return

    def deleteQueue(self):
        log.info(f"Deleting {self.tq.enqueued} tasks")
        [self.tq.delete(t) for t in self.tq.tasks()]
        return


def submitAll(
    nDays,
    settings,
    fileQueue,
    withLevel2detect=False,
    doMetaRot=True,
    submitJobs=True,
    skipExisting=True,
):
    if submitJobs:
        prod = DataProductRange("level2track", nDays, settings, fileQueue, "leader")
        prod.submitCommands(checkForDuplicates=True, skipExisting=skipExisting)

        prod = DataProductRange("level2match", nDays, settings, fileQueue, "leader")
        prod.submitCommands(
            checkForDuplicates=True, withParents=False, skipExisting=skipExisting
        )

        if withLevel2detect:
            prod = DataProductRange(
                "level2detect", nDays, settings, fileQueue, "leader"
            )
            prod.submitCommands(
                checkForDuplicates=True, withParents=False, skipExisting=skipExisting
            )
            prod = DataProductRange(
                "level2detect", nDays, settings, fileQueue, "follower"
            )
            prod.submitCommands(
                checkForDuplicates=True, withParents=False, skipExisting=skipExisting
            )
    if doMetaRot:
        scripts.loopCreateMetaRotation(settings, skipExisting=skipExisting, nDays=nDays)