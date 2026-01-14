import glob
import logging
import os
import random
import string
import sys
from functools import cached_property, partial

import numpy as np
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
        childrensRelatives=tools.DictNoDefault({}),
    ):
        """
        Initialize a DataProduct for processing VISSS data.

        Parameters
        ----------
        level : str
            Processing level (e.g., 'level0', 'level1detect', 'metaEvents')
        case : str
            Case identifier for the data
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str
            Camera identifier ('leader' or 'follower')
        relatives : str, optional
            Relative path specification
        addRelatives : bool, default True
            Whether to add relatives of the corresponding product
        childrensRelatives : dict, default {}
            Dictionary of child relatives

        Raises
        ------
        ValueError
            If camera is not 'leader' or 'follower'
        """
        import taskqueue

        """
        Class for processing VISSS data

        """
        log.info(f"created  {level} {camera} for {case} with {childrensRelatives}.")
        self.level = level
        self.config = tools.readSettings(settings)
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

        self.fn = files.FindFiles(str(self.case), self.cameraFull, self.config)
        self.path = self.fn.fnamesPatternExt[self.level]

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
                f"leader_metaEvents",  # metaEvents are added to all the L2 products to force regenration when event file is updated (ie more data is transferred)
                f"follower_metaEvents",
            ]
        elif level == "level1match":
            assert camera == "leader"
            self.parentNames = [f"{camera}_metaRotation"]
        elif level == "level1track":
            assert camera == "leader"
            self.parentNames = [f"{camera}_level1match"]
        # elif level == "level1shape":
        #     assert camera == "leader"
        #     self.parentNames = [f"{camera}_level1track"]
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
            if self.config.level1match.processL1match:
                self.parentNames += [
                    "leader_level2track",
                    "leader_level2match",
                ]
            if self.config.level2.processL2detect:
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

    def __repr__(self):
        """
        Return string representation of the DataProduct object.

        Returns
        -------
        str
            String representation of the object
        """
        reprstr = (
            f"<VISSSlib.products.DataProduct object {self.level} "
            f"using {self.camera} on {self.case}>"
        )
        return reprstr

    def addRelatives(self):
        """
        Add relative products to the current product.

        This method recursively adds parent products based on the parent names
        defined for the current level.
        """
        for parentCam in self.parentNames:
            # save time by not adding a product more than once
            if parentCam in self.childrensRelatives.keys():
                # print(f"{self.relatives}, found {parentCam} from other relative")
                self.parents[parentCam] = self.childrensRelatives[parentCam]
                assert self.case == self.childrensRelatives[parentCam].case
                continue
            camera, parent = parentCam.split("_")
            self.parents[parentCam] = DataProduct(
                parent,
                self.case,
                self.config,
                self.tq,
                camera,
                relatives=f"{self.relatives}",
                childrensRelatives=self.parents,
            )
            self.parents.update(self.parents[parentCam].parents)
            self.childrensRelatives.update(self.parents)

    def generateAllCommands(self, skipExisting=True, withParents=True):
        """
        Generate all commands for processing this product and its dependencies.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        withParents : bool, default True
            Whether to include parent commands

        Returns
        -------
        list
            List of commands to execute
        """
        # cache for this function
        isComplete = self.isComplete

        if not self.dataAvailable:
            log.warning(
                f"{self.case} {self.relatives}: no data found in {self.fn.fnamesPattern.level0txt}"
            )
            return []

        if (
            skipExisting
            and isComplete
            and self._youngerThanParents
            and self.parentsComplete
        ):
            if withParents:
                log.warning(f"{self.case} {self.relatives}: everything processed")
            return []
        if isComplete and (not self._youngerThanParents):
            for name, younger in self._youngerThanParentsDict.items():
                if not younger:
                    log.warning(
                        f"{self.case} {self.relatives} redoing level, parent {name} is younger"
                    )
        if self.parentsComplete and self._parentsYoungerThanGrandparents:
            commands = self.generateCommands(
                skipExisting=skipExisting,
            )
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
                    skipExisting=True,
                    withParents=False,
                )
        self.commands = list(set(commands))
        if (len(self.commands) == 0) and (withParents):
            log.warning(
                f"{self.level} {self.camera} {self.case} no commands created",
            )
        return self.commands

    def generateCommands(self, skipExisting=True, nCPU=1, bin=None):
        """
        Generate commands for processing this product.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        nCPU : int, default 1
            Number of CPU cores to use
        bin : str, optional
            Python binary path

        Returns
        -------
        list
            List of commands to execute

        Raises
        ------
        ValueError
            If the level is not recognized
        """
        if self.level == "level0":
            return []
        elif self.level == "level0txt":
            return []
        elif self.level == "metaEvents":
            return self._commandTemplateDaily(
                "metadata.createEvent", skipExisting=skipExisting, nCPU=nCPU, bin=bin
            )
        elif self.level == "metaFrames":
            return self._commandTemplateDaily(
                "scripts.loopCreateMetaFrames",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level1detect":
            originLevel = "level0txt"
            call = "detection.detectParticles"
            return self._commandTemplateL1(
                originLevel,
                call,
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "metaRotation":
            return self._commandTemplateDaily(
                "matching.createMetaRotation",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level1match":
            originLevel = "level1detect"
            call = "matching.matchParticles"
            return self._commandTemplateL1(
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
            return self._commandTemplateL1(
                originLevel, call, skipExisting=skipExisting, nCPU=nCPU, bin=bin
            )
        # elif self.level == "level1shape":
        #     originLevel = "level1track"
        #     call = "particleshape.classifyParticles"
        #     return self._commandTemplateL1(
        #         originLevel, call, skipExisting=skipExisting, nCPU=nCPU, bin=bin
        #     )
        elif self.level == "level2detect":
            return self._commandTemplateDaily(
                "distributions.createLevel2detect",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level2match":
            return self._commandTemplateDaily(
                "distributions.createLevel2match",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level2track":
            return self._commandTemplateDaily(
                "distributions.createLevel2track",
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
            )
        elif self.level == "level3combinedRiming":
            return self._commandTemplateDaily(
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
            raise ValueError(f"Do not understand {self.level}")

    def _commandTemplateL1(
        self,
        originLevel,
        call,
        skipExisting=True,
        nCPU=1,
        bin=None,
        extraOrigin=None,
    ):
        """
        Generate commands for L1 processing steps.

        Parameters
        ----------
        originLevel : str
            Origin level for processing
        call : str
            Function call to execute
        skipExisting : bool, default True
            Whether to skip existing files
        nCPU : int, default 1
            Number of CPU cores to use
        bin : str, optional
            Python binary path
        extraOrigin : str, optional
            Extra origin level for comparison

        Returns
        -------
        list
            List of commands to execute
        """
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

            command = f"{bin} -m VISSSlib {call}  {pName} {self.config.filename} {skipExisitingInt}"
            if nCPU is not None:
                command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {command}"
            commands.append((command, outFile))
        return commands

    def _commandTemplateDaily(self, call, skipExisting=True, nCPU=1, bin=None):
        """
        Generate commands for daily processing steps.

        Parameters
        ----------
        call : str
            Function call to execute
        skipExisting : bool, default True
            Whether to skip existing files
        nCPU : int, default 1
            Number of CPU cores to use
        bin : str, optional
            Python binary path

        Returns
        -------
        list
            List of commands to execute
        """
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
        if skipExisting and (len(exisiting) >= 1) and (self._youngerThanParents):
            log.info(f"{self.relatives} skip exisiting {exisiting[0]}")
            return []

        command = (
            f"{bin} -m VISSSlib {call} {self.config.filename} {case} {skipExisitingInt}"
        )
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
        """
        Submit commands to the task queue.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        checkForDuplicates : bool, default False
            Whether to check for duplicate commands
        withParents : bool, default True
            Whether to include parent commands
        runWorkers : bool, default False
            Whether to run workers immediately
        """
        if len(self.commands) == 0:
            self.generateAllCommands(
                skipExisting=skipExisting,
                withParents=withParents,
            )

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
            self.runWorkers()

        return

    def runWorkers(self, nJobs=os.cpu_count(), waitTime=1):
        """
        Run worker processes.

        Parameters
        ----------
        nJobs : int, default os.cpu_count()
            Number of jobs to run
        """
        tools.workers(self.fileQueue, nJobs=nJobs, waitTime=waitTime)

    def deleteQueue(self):
        """
        Delete all tasks from the queue.
        """
        log.info(f"Deleting {self.tq.enqueued} tasks")
        [self.tq.delete(t) for t in self.tq.tasks()]
        return

    @cached_property
    def isComplete(self):
        """
        Check if all required files for this level exist.

        Returns
        -------
        bool
            True if all files are complete, False otherwise
        """
        nMissing = self.fn.nMissing(self.level)
        if nMissing > 0:
            log.info(f"{self.case} {self.relatives} {nMissing} files are missing")
        return nMissing == 0

    @cached_property
    def _youngerThanParentsDict(self):
        """
        Check if this product is younger than its parents.

        Returns
        -------
        dict
            Dictionary mapping parent names to boolean values indicating
            whether this product is younger than each parent
        """
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
    def _youngerThanParents(self):
        """
        Check if this product is younger than all parents.

        Returns
        -------
        bool
            True if this product is younger than all parents, False otherwise
        """
        youngerThanParents = np.all(list(self._youngerThanParentsDict.values()))
        return youngerThanParents

    @cached_property
    def _parentsYoungerThanGrandparents(self):
        """
        Check if parents are younger than their grandparents.

        Returns
        -------
        bool
            True if all parents are younger than their grandparents, False otherwise
        """
        parentsYoungerThanGrandparents = True
        for name, parent in self.parents.items():
            parentsYoungerThanGrandparents = (
                parentsYoungerThanGrandparents and parent._youngerThanParents
            )
            log.info(
                f"{self.relatives} parent {name} is younger than its (grand)parents { parent._youngerThanParents}"
            )
        return parentsYoungerThanGrandparents

    def _fileCreation(self, files):
        """
        Get the creation time of the most recent file.

        Parameters
        ----------
        files : list
            List of file paths

        Returns
        -------
        float
            Maximum modification time of the files
        """
        if len(files) > 0:
            return np.max([os.path.getmtime(f) for f in files])
        else:
            return 0

    @cached_property
    def fileCreation(self):
        """
        Get the creation time of this product.

        Returns
        -------
        float
            Modification time of the newest file
        """
        files = self.listFilesExt()
        return self._fileCreation(files)

    @cached_property
    def parentsComplete(self):
        """
        Check if product's parent are complete.

        Returns
        -------
        bool
            True if all parents are complete, False otherwise
        """
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
        """
        Print a report about this product's status.

        Parameters
        ----------
        withParents : bool, default True
            Whether to include parent reports
        """
        nMissing = self.fn.nMissing(self.level)
        print(
            self.camera,
            self.level,
            "nMissing",
            nMissing,
            "newest file",
            tools.timestamp2str(self.fileCreation),
            "younger than parents",
            self._youngerThanParents,
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
    def dataAvailable(self):
        """
        Check if data is available for this product.

        Returns
        -------
        bool
            True if data is available, False otherwise
        """
        return len(self.fn.listFiles("level0txt")) > 0

    @cached_property
    def allComplete(self):
        """
        Check if this product and all its dependencies are complete.

        Returns
        -------
        bool
            True if all is complete, False otherwise
        """
        return self.isComplete and self._youngerThanParents and self.parentsComplete

    @cached_property
    def nFiles(self):
        """
        Get the number of files for this product.

        Returns
        -------
        int
            Number of files
        """
        return len(self.fn.listFilesExt(self.level))

    def listFilesExt(self):
        """
        List all files for this product.

        Returns
        -------
        list
            List of file paths
        """
        return self.fn.listFilesExt(self.level)

    def listFiles(self):
        """
        List files for this product.

        Returns
        -------
        list
            List of file paths
        """
        return self.fn.listFiles(self.level)

    def listBroken(self):
        """
        List broken files for this product.

        Returns
        -------
        list
            List of broken file paths
        """
        return self.fn.listBroken(self.level)

    def listNoData(self):
        """
        List files with no data for this product.

        Returns
        -------
        list
            List of no-data file paths
        """
        return self.fn.listNoData(self.level)

    def cleanUpBroken(self, withParents=False, withNoData=False):
        """
        Clean up broken files.

        Parameters
        ----------
        withParents : bool, default False
            Whether to clean up parents too
        withNoData : bool, default False
            Whether to clean up no-data files too
        """
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
        """
        Clean up duplicate files.

        Parameters
        ----------
        withParents : bool, default False
            Whether to clean up parents too
        """
        for fname in self.fn.reportDuplicates(self.level):
            os.remove(fname)
            log.warning(f"{fname} removed")
        if withParents:
            for name, parent in self.parents.items():
                parent.cleanUpDuplicates(withParents=False)


class allDone(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize an allDone product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("allDone", case, settings, fileQueue, camera)


class level2track(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level2track product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level2track", case, settings, fileQueue, camera)


class level2match(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level2match product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level2match", case, settings, fileQueue, camera)


class level2detect(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level2detect product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level2detect", case, settings, fileQueue, camera)


class level1track(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level1track product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level1track", case, settings, fileQueue, camera)


class level1match(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level1match product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level1match", case, settings, fileQueue, camera)


class metaRotation(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a metaRotation product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("metaRotation", case, settings, fileQueue, camera)


class level1detect(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level1detect product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("level1detect", case, settings, fileQueue, camera)


# class level1shape(DataProduct):
#     def __init__(self, case, settings, fileQueue, camera="leader"):
#         super().__init__("level1shape", case, settings, fileQueue, camera)


class metaFrames(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a metaFrames product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("metaFrames", case, settings, fileQueue, camera)


class metaEvents(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a metaEvents product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
        super().__init__("metaEvents", case, settings, fileQueue, camera)


class level0(DataProduct):
    def __init__(self, case, settings, fileQueue, camera="leader"):
        """
        Initialize a level0 product.

        Parameters
        ----------
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        """
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
        """
        Initialize a range of DataProducts for multiple days.

        Parameters
        ----------
        level : str
            Processing level
        nDays : int
            Number of days going back or date string "YYYYMMDD" or "YYYYMMDD-YYYYMMDD"
            or "YYYYMMDD,YYYYMMDD,YYYYMMDD".
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str, default "leader"
            Camera identifier
        addRelatives : bool, default True
            Whether to add relatives of the corresponding product
        """
        import taskqueue

        self.settings = settings
        self.config = tools.readSettings(settings)
        self.days = tools.getDateRange(nDays, self.config, endYesterday=False)
        self.dailies = {}
        self.level = level
        self.camera = camera
        self.allCommands = []

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
                childrensRelatives=tools.DictNoDefault(
                    {}
                ),  # not sure why this is requried, bugs appear otehrwise
            )

    def generateAllCommands(self, skipExisting=True, withParents=True):
        """
        Generate all commands for the range of products.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        withParents : bool, default True
            Whether to include parent commands

        Returns
        -------
        list
            List of commands to execute
        """
        for dd in self.days:
            self.allCommands += self.dailies[dd].generateAllCommands(
                skipExisting=skipExisting,
                withParents=withParents,
            )
        return self.allCommands

    def submitCommands(
        self,
        skipExisting=True,
        checkForDuplicates=False,
        withParents=True,
        runWorkers=False,
    ):
        """
        Submit commands for the range of products.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        checkForDuplicates : bool, default False
            Whether to check for duplicate commands
        withParents : bool, default True
            Whether to include processing of product's parents
        runWorkers : bool, default False
            Whether to run workers immediately
        """
        if len(self.allCommands) == 0:
            self.generateAllCommands(
                skipExisting=skipExisting,
                withParents=withParents,
            )
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
            self.runWorkers()

        return

    @property
    def isComplete(self):
        """
        Check if all products in the range are complete.

        Returns
        -------
        bool
            True if all products are complete, False otherwise
        """
        if len(self.days) == 0:
            log.warning(f"Number of days is zero")
        isComplete = True
        for dd in self.days:
            isComplete = isComplete and self.dailies[dd].isComplete
        return isComplete

    def runWorkers(self, nJobs=os.cpu_count(), waitTime=1):
        """
        Run worker processes for the range.

        Parameters
        ----------
        nJobs : int, default os.cpu_count()
            Number of jobs to run
        """
        tools.workers(self.fileQueue, nJobs=nJobs, waitTime=waitTime)

    def deleteQueue(self):
        """
        Delete all tasks from the queue.
        """
        log.info(f"Deleting {self.tq.enqueued} tasks")
        [self.tq.delete(t) for t in self.tq.tasks()]
        return

    def cleanUpBroken(self, withParents=False, withNoData=False):
        """
        Clean up broken files for the range.

        Parameters
        ----------
        withParents : bool, default False
            Whether to clean up parents too
        withNoData : bool, default False
            Whether to clean up no-data files too
        """
        for dd in self.days:
            self.dailies[dd].cleanUpBroken(
                withParents=withParents, withNoData=withNoData
            )

    def cleanUpDuplicates(self, withParents=False):
        """
        Clean up duplicate files for the range.

        Parameters
        ----------
        withParents : bool, default False
            Whether to clean up parents too
        """
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
    """
    Submit all processing jobs for a given range of days.

    Parameters
    ----------
    nDays : int
        Number of days to process
    settings : str
        Path to settings file
    fileQueue : str
        File queue for task management
    doMetaRot : bool, default True
        Whether to perform meta rotation
    submitJobs : bool, default True
        Whether to submit jobs to the queue
    skipExisting : bool, default True
        Whether to skip existing files
    checkForDuplicates : bool, default True
        Whether to check for duplicate commands
    runWorkers : bool, default False
        Whether to run workers immediately
    cleanUpBroken : bool, default False
        Whether to clean up broken files
    cleanUpDuplicates : bool, default False
        Whether to clean up duplicate files

    Returns
    -------
    object
        DataProductRange object
    """
    if submitJobs:
        import taskqueue

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
        log.warning(
            f"{sys.executable} -m VISSSlib scripts.loopCreateMetaRotation  {settings} {nDays}"
        )
        scripts.loopCreateMetaRotation(settings, skipExisting=skipExisting, nDays=nDays)

    return prod


@tools.loopify
def processCases(case, config, ignoreErrors=False, nJobs=os.cpu_count, fileQueue=None):
    if fileQueue is None:
        fileQueue = f"/tmp/visss_{''.join(
            random.choice(string.ascii_uppercase) for _ in range(10)
            )}"

    products = [
        "metaEvents",
        "level1detect",
    ]
    if config.level1match.processL1match:
        products += [
            "metaRotation",
            "level1match",
            "level1track",
            "level2match",
            "level2track",
        ]
    if config.level2.processL2detect:
        products += ["level2detect"]
    if config.level3.combinedRiming.processRetrieval:
        products += ["level3combinedRiming"]
    products += [
        "allDone",
    ]

    followerProducts = ["metaEvents", "level1detect", "level2detect"]
    for prod in products:
        print("#" * 10, prod, "#" * 10)
        dp1 = DataProduct(prod, case, config, fileQueue, "leader")
        dp1.submitCommands(withParents=False)
        if prod in followerProducts:
            dp2 = DataProduct(prod, case, config, fileQueue, "follower")
            dp2.submitCommands(withParents=False)
        VISSSlib.tools.workers(fileQueue, waitTime=1, nJobs=nJobs)
        if not ignoreErrors:
            assert len(dp1.listBroken()) == 0, "leader files broken"
            assert len(dp1.listFiles()) > 0, "no leader output"
            if prod in followerProducts:
                assert len(dp2.listBroken()) == 0, "follower files broken"
                assert len(dp2.listFiles()) > 0, "no follower output"
