import glob
import logging
import os
import random
import string
import sys
from functools import cached_property, partial

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
        relatives=None,
        addRelatives=True,
        fileObject=None,
        childrensRelatives=tools.DictNoDefault({}),
    ):
        """
        Class for processing VISSS data

        """
        log.info(f"create object for level {level} {camera}")
        self.level = level
        self.config = tools.readSettings(settings)
        self.settings = settings
        if relatives is not None:
            self.relatives = f"{relatives}.{level}"
        else:
            self.relatives = level
        self.childrensRelatives = childrensRelatives
        if camera == "leader":
            self.cameraFull = self.config.leader
        elif camera == "follower":
            self.cameraFull = self.config.follower
        else:
            raise ValueError(f"do not understand camera: {camera}")
        self.camera = camera
        self.case = case

        if fileQueue is None:
            fileQueue = f"/tmp/visss_{''.join(random.choice(string.ascii_uppercase) for _ in range(10))}"

        if type(fileQueue) is str:
            self.fileQueue = fileQueue
            self.tq = taskqueue.TaskQueue(f"fq://{self.fileQueue}")
        else:
            self.tq = fileQueue
            self.fileQueue = self.tq.path.path

        self.commands = []
        self.allCommands = []

        self.fn = files.FindFiles(str(self.case), self.cameraFull, self.config)

        self.parents = tools.DictNoDefault({})

        if self.level == "level0":
            self.parentNames = []
        elif self.level == "level0txt":
            self.parentNames = []
        elif level == "metaEvents":
            self.parentNames = [f"{camera}_level0txt"]
        elif level == "metaFrames":
            self.parentNames = [f"{camera}_level0txt"]
        elif level == "level1detect":
            self.parentNames = [
                # f"{camera}_metaFrames", # done by level1detect
                # f"{camera}_metaEvents", # done by metaRotation
            ]
        elif level == "metaRotation":
            assert camera == "leader"
            self.parentNames = [
                f"leader_level1detect",
                f"follower_level1detect",
                f"leader_metaEvents",  # metaEvents are aded to all the L2 products to force regenration when event file is updated (ie more data is transferred)
                f"follower_metaEvents",
            ]
        elif level == "level1match":
            assert camera == "leader"
            self.parentNames = [f"{camera}_metaRotation"]
        elif level == "level1track":
            assert camera == "leader"
            self.parentNames = [f"{camera}_level1match"]
        elif level == "level1shape":
            assert camera == "leader"
            self.parentNames = [f"{camera}_level1track"]
        elif level == "level2detect":
            self.parentNames = [f"{camera}_level1detect", f"{camera}_metaEvents"]
        elif level == "level2match":
            assert camera == "leader"
            self.parentNames = [
                f"{camera}_level1match",
                f"leader_metaEvents",  # metaEvents are aded to all the L2 products to force regenration when events file is updated (ie more data is transferred)
                f"follower_metaEvents",
            ]
        elif level == "level2track":
            assert camera == "leader"
            self.parentNames = [
                f"{camera}_level1track",
                f"leader_metaEvents",
                f"follower_metaEvents",
            ]
        elif level == "level3combinedRiming":
            assert camera == "leader"
            self.parentNames = [
                f"{camera}_level2track",
                f"leader_metaEvents",
                f"follower_metaEvents",
            ]
        elif level == "allDone":
            assert camera == "leader"
            self.parentNames = [
                f"leader_metaEvents",
                f"follower_metaEvents",
            ]
            if self.config.matchData:
                self.parentNames += [
                    "leader_level2track",
                    "leader_level2match",
                ]
            if self.config.processL2detect:
                self.parentNames += [
                    "leader_level2detect",
                    "follower_level2detect",
                ]
            if self.config.level3.combinedRiming.processRetrieval:
                self.parentNames += [
                    "leader_level3combinedRiming",
                ]

        else:
            raise ValueError(f"Do not understand {level}")
        if addRelatives:
            self.addRelatives()

    def addRelatives(self):
        for parentCam in self.parentNames:
            camera, parent = parentCam.split("_")
            # save time by not adding a product more than once
            if parentCam in self.childrensRelatives.keys():
                # print(f"{self.relatives}, found {parentCam} from other relative")
                self.parents[parentCam] = self.childrensRelatives[parentCam]
                continue
            self.parents[parentCam] = DataProduct(
                parent,
                self.case,
                self.settings,
                self.tq,
                camera,
                relatives=f"{self.relatives}",
                childrensRelatives=self.parents,
            )
            self.parents.update(self.parents[parentCam].parents)
            self.childrensRelatives.update(self.parents)

    def generateAllCommands(self, skipExisting=True, withParents=True):
        # cache for this function
        isComplete = self.isComplete

        if (
            skipExisting
            and isComplete
            and self.youngerThanParents
            and self.parentsComplete
        ):
            if withParents:
                log.warning(f"{self.case} {self.relatives}: everything processed")
            return []
        if isComplete and (not self.youngerThanParents):
            for name, younger in self._youngerThanParentsDict.items():
                if not younger:
                    log.warning(
                        f"{self.case} {self.relatives} redoing level, parent {name} is younger"
                    )
        if self.parentsComplete and self.parentsYoungerThanGrandparents:
            commands = self.generateCommands(skipExisting=skipExisting)
            if len(commands) > 0:
                log.warning(
                    f"{self.case} {self.relatives} generated commands for level {self.level} {self.camera}"
                )
        elif not self.parentsComplete:
            log.info(
                f"{self.case} {self.relatives} no commands generated yet, parents not complete yet"
            )
            commands = []
        else:
            log.info(
                f"{self.case} {self.relatives} no commands generated, grandparents older"
            )
            commands = []
        if withParents:
            for parent in self.parents.keys():
                # parents always with skipExisting = True to avoid chain reaction
                commands = commands + self.parents[parent].generateAllCommands(
                    skipExisting=True, withParents=False
                )
        self.commands = list(set(commands))
        if (len(self.commands) == 0) and (withParents):
            log.warning(
                f"{self.level} {self.camera} {self.case} no commands created",
            )
        return self.commands

    def generateCommands(self, skipExisting=True, nCPU=1, bin=None):
        if self.level == "level0":
            return []
        elif self.level == "level0txt":
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
                originLevel,
                call,
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
                extraOrigin="metaRotation",
            )
        elif self.level == "level1track":
            originLevel = "level1match"
            call = "tracking.trackParticles"
            return self.commandTemplateL1(
                originLevel, call, skipExisting=skipExisting, nCPU=nCPU, bin=bin
            )
        elif self.level == "level1shape":
            originLevel = "level1track"
            call = "particleshape.classifyParticles"
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
        elif self.level == "level3combinedRiming":
            return self.commandTemplateDaily(
                "level3.retrieveCombinedRiming",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "allDone":
            outFile = self.fn.fnamesDaily["allDone"]
            command = f"mkdir -p {os.path.dirname(outFile)} && touch {outFile}"
            return [(command, outFile)]
        else:
            raise ValueError(f"Do not understand {level}")

    def commandTemplateL1(
        self, originLevel, call, skipExisting=True, nCPU=1, bin=None, extraOrigin=None
    ):
        nCPU = 1
        skipExisitingInt = int(skipExisting)
        if bin is None:
            bin = os.path.join(sys.exec_prefix, "bin", "python")
        commands = []
        for pName in self.fn.listFilesExt(originLevel):
            if originLevel.startswith("level0"):
                f1 = files.Filenames(pName, self.config)
            else:
                f1 = files.FilenamesFromLevel(pName, self.config)
            outFile = f1.fname[self.level]
            exisiting = glob.glob(f"{outFile}*")

            if (len(exisiting) >= 1) and (extraOrigin is not None):
                extraOlder = os.path.getmtime(
                    self.fn.listFilesExt(extraOrigin)[0]
                ) < os.path.getmtime(exisiting[0])
            else:
                extraOlder = True

            if (
                skipExisting
                and (len(exisiting) >= 1)
                and (os.path.getmtime(pName) < os.path.getmtime(exisiting[0]))
                and extraOlder
            ):
                log.info(f"{self.relatives} skip exisiting {exisiting[0]}")
                continue

            if len(exisiting) > 1:
                for ex in exisiting:
                    os.remove(ex)
                    log.warning(f"too many files, removed {ex}")

            command = f"{bin} -m VISSSlib {call}  {pName} {self.settings}"
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
            or call.endswith("createLevel1detectQuicklook")
        ):
            case = f"{self.camera}+{self.case}"
        else:
            case = self.case

        outFile = self.fn.fnamesDaily[self.level]

        exisiting = glob.glob(f"{outFile}*")
        if skipExisting and (len(exisiting) >= 1) and (self.youngerThanParents):
            log.info(f"{self.relatives} skip exisiting {exisiting[0]}")
            return []

        command = f"{bin} -m VISSSlib {call} {self.settings} {case} {skipExisitingInt}"
        if nCPU is not None:
            command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {command}"
        return [(command, outFile)]

    def submitCommands(
        self,
        skipExisting=True,
        checkForDuplicates=False,
        withParents=True,
        runWorkers=False,
    ):
        if len(self.commands) == 0:
            self.generateAllCommands(skipExisting=skipExisting, withParents=withParents)

        if len(self.commands) == 0:
            log.error("nothing to submit")
            return

        if checkForDuplicates:
            running = [t.args[0] for t in self.tq.tasks()]
            commands = []
            for command in self.commands:
                if command[0][0] in running:
                    continue
                else:
                    commands.append(command)
        else:
            commands = self.commands

        log.warning(f"sending {len(commands)} commands to {self.fileQueue}")
        # region is SQS specific, green means cooperative threading

        self.tq.insert([partial(runCommandInQueue, c) for c in commands])
        log.warning(f"{self.tq.enqueued} tasks in Queue")

        if runWorkers:
            tools.workers(self.fileQueue)

        return

    def deleteQueue(self):
        log.info(f"Deleting {self.tq.enqueued} tasks")
        [self.tq.delete(t) for t in self.tq.tasks()]
        return

    @cached_property
    def isComplete(self):
        nMissing = self.fn.nMissing(self.level)
        if nMissing > 0:
            log.info(f"{self.case} {self.relatives} {nMissing} files are missing")
        return nMissing == 0

    @cached_property
    def _youngerThanParentsDict(self):
        youngerThanParentsDict = tools.DictNoDefault()
        for name, parent in self.parents.items():
            isYounger = parent.fileCreation < self.fileCreation
            if (self.level == "level1detect") and (parent.level == "metaEvents"):
                # special case: no need to do level1detect again due to updated metaEvents
                youngerThanParentsDict[name] = True
            else:
                youngerThanParentsDict[name] = isYounger
            if not youngerThanParentsDict[name]:
                log.info(
                    f"{self.relatives} is older "
                    f"({tools.timestamp2str(self.fileCreation)}) than parent "
                    f"{name} ({tools.timestamp2str(parent.fileCreation)})",
                )
        return youngerThanParentsDict

    @cached_property
    def youngerThanParents(self):
        youngerThanParents = np.all(list(self._youngerThanParentsDict.values()))
        return youngerThanParents

    @cached_property
    def parentsYoungerThanGrandparents(self):
        parentsYoungerThanGrandparents = True
        for name, parent in self.parents.items():
            parentsYoungerThanGrandparents = (
                parentsYoungerThanGrandparents and parent.youngerThanParents
            )
            log.info(
                f"{self.relatives} parent {name} is younger than its (grand)parents { parent.youngerThanParents}"
            )
        return parentsYoungerThanGrandparents

    def _fileCreation(self, files):
        if len(files) > 0:
            return np.max([os.path.getmtime(f) for f in files])
        else:
            return 0

    @cached_property
    def fileCreation(self):
        files = self.listFilesExt()
        return self._fileCreation(files)

    @cached_property
    def parentsComplete(self):
        parentsComplete = True
        for name, parent in self.parents.items():
            thisParentIsComplete = parent.isComplete
            log.info(
                f"{self.relatives} {name} parentsComplete {thisParentIsComplete}",
            )
            parentsComplete = parentsComplete and thisParentIsComplete
            if not parentsComplete:  # shortcut
                break
        return parentsComplete

    def report(self, withParents=True):
        nMissing = self.fn.nMissing(self.level)
        print(
            self.camera,
            self.level,
            "nMissing",
            nMissing,
            "newest file",
            tools.timestamp2str(self.fileCreation),
            "younger than parents",
            self.youngerThanParents,
        )
        if nMissing > 0:
            print(
                " " * 5,
                [(p, self.fn.nMissing(p.split("_")[1])) for p in self.parentNames],
            )
        if withParents:
            for name, parent in self.parents.items():
                parent.report(withParents=False)

    @cached_property
    def allComplete(self):
        return self.isCompleteand and self.youngerThanParents and self.parentsComplete

    @cached_property
    def nFiles(self):
        return len(self.fn.listFilesExt(self.level))

    def listFilesExt(self):
        return self.fn.listFilesExt(self.level)

    def listFiles(self):
        return self.fn.listFiles(self.level)

    def listBroken(self):
        return self.fn.listBroken(self.level)

    def listNoData(self):
        return self.fn.listNoData(self.level)

    def cleanUpBroken(self, withParents=False, withNoData=False):
        for fname in self.listBroken():
            assert fname.endswith("broken.txt")
            os.remove(fname)
            log.warning(f"{fname} removed")
        if withNoData:
            for fname in self.listNoData():
                assert fname.endswith("nodata")
                os.remove(fname)
                log.warning(f"{fname} removed")
        if withParents:
            for name, parent in self.parents.items():
                parent.cleanUpBroken(withParents=False, withNoData=withNoData)

    def cleanUpDuplicates(self, withParents=False):
        for fname in self.fn.reportDuplicates(self.level):
            os.remove(fname)
            log.warning(f"{fname} removed")
        if withParents:
            for name, parent in self.parents.items():
                parent.cleanUpDuplicates(withParents=False)


class allDone(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("allDone", case, settings, fileQueue, camera)


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


class level1shape(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        super().__init__("level1shape", case, settings, fileQueue, camera)


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

        if fileQueue is None:
            fileQueue = f"/tmp/visss_{''.join(random.choice(string.ascii_uppercase) for _ in range(10))}"

        if type(fileQueue) is str:
            self.fileQueue = fileQueue
            self.tq = taskqueue.TaskQueue(f"fq://{self.fileQueue}")
        else:
            self.tq = fileQueue
            self.fileQueue = self.tq.path.path

        for dd in self.days:
            case = tools.timestamp2case(dd)
            self.dailies[dd] = DataProduct(
                level,
                case,
                settings,
                self.tq,
                camera,
                addRelatives=addRelatives,
            )

    def generateAllCommands(self, skipExisting=True, withParents=True):
        self.allCommands = []
        for dd in self.days:
            self.allCommands += self.dailies[dd].generateAllCommands(
                skipExisting=skipExisting, withParents=withParents
            )
        return self.allCommands

    def submitCommands(
        self,
        skipExisting=True,
        checkForDuplicates=False,
        withParents=True,
        runWorkers=False,
    ):
        self.generateAllCommands(skipExisting=skipExisting, withParents=withParents)
        if len(self.allCommands) == 0:
            log.error("nothing to submit")
            return

        if checkForDuplicates:
            running = [t.args[0][0] for t in self.tq.tasks()]
            commands = []
            for command in self.allCommands:
                if command[0] in running:
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

        if runWorkers:
            tools.workers(self.fileQueue)

        return

    def deleteQueue(self):
        log.info(f"Deleting {self.tq.enqueued} tasks")
        [self.tq.delete(t) for t in self.tq.tasks()]
        return

    def cleanUpBroken(self, withParents=False, withNoData=False):
        for dd in self.days:
            self.dailies[dd].cleanUpBroken(
                withParents=withParents, withNoData=withNoData
            )

    def cleanUpDuplicates(self, withParents=False):
        for dd in self.days:
            self.dailies[dd].cleanUpDuplicates(withParents=withParents)


def submitAll(
    nDays,
    settings,
    fileQueue,
    doMetaRot=True,
    submitJobs=True,
    skipExisting=True,
    checkForDuplicates=True,
    runWorkers=False,
    cleanUpBroken=False,
    cleanUpDuplicates=False,
):
    if submitJobs:
        tq = taskqueue.TaskQueue(f"fq://{fileQueue}")
        log.warning(f"{tq.enqueued} tasks in Queue")

        prod = DataProductRange("allDone", nDays, settings, fileQueue, "leader")
        if cleanUpBroken:
            prod.cleanUpBroken(withParents=True, withNoData=False)
        if cleanUpDuplicates:
            prod.cleanUpDuplicates(withParents=True)
        prod.submitCommands(
            checkForDuplicates=checkForDuplicates,
            skipExisting=skipExisting,
            runWorkers=runWorkers,
        )

    else:
        prod = None

    if doMetaRot:
        scripts.loopCreateMetaRotation(settings, skipExisting=skipExisting, nDays=nDays)

    return prod
