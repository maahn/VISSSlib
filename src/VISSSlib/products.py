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
    """Class for processing VISSS data.

    This class handles the creation and management of data products at different
    processing levels for VISSS data.

    Attributes
    ----------
    level : str
        The processing level of the data product
    config : dict
        Configuration settings for the processing
    settings : str
        Path to the settings file
    relatives : str
        Relative path information for the product
    childrensRelatives : dict
        Dictionary of child relatives
    cameraFull : str
        Full camera identifier
    camera : str
        Camera identifier (leader or follower)
    case : str
        Case identifier
    fileQueue : str
        File queue path
    tq : taskqueue.TaskQueue
        Task queue object
    commands : list
        List of commands to execute
    fn : files.FindFiles
        File finder object
    path : str
        Path pattern for the data
    parents : dict
        Parent data products
    parentNames : list
        Names of parent data products
    """

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
        childrensRelatives=None,
    ):
        """

        Initialize a DataProduct instance.

        Parameters
        ----------
        level : str
            Processing level (e.g., 'level0', 'level1detect')
        case : str
            Case identifier
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str
            Camera identifier ('leader' or 'follower')
        relatives : str, optional
            Relative path information
        addRelatives : bool, default True
            Whether to add relatives
        fileObject : object, optional
            File object
        childrensRelatives : dict, optional
            Children relatives dictionary
        """
        log.info(f"created  {level} {camera} for {case} with {childrensRelatives}.")
        self.level = level
        self.config = tools.readSettings(settings)
        if relatives is not None:
            self.relatives = f"{relatives}.{level}"
        else:
            self.relatives = level
        if childrensRelatives is None:
            self.childrensRelatives = tools.DictNoDefault({})
        else:
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
                f"leader_metaEvents",  # metaEvents are aded to all the L2 products to force regenration when event file is updated (ie more data is transferred)
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
            for parentCam in self.parentNames:
                camera, parent = parentCam.split("_")
                # save time by not adding a product more than once
                if parentCam in self.childrensRelatives.keys():
                    # print(f"{self.relatives}, found {parentCam} from other relative")
                    self.parents[parentCam] = self.childrensRelatives[parentCam]
                    assert self.case == self.childrensRelatives[parentCam].case
                    continue
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

    def generateAllCommands(
        self, skipExisting=True, withParents=True, doNotWaitForMissingThreadFiles=False
    ):
        """Generate all commands for this data product and its parents.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        withParents : bool, default True
            Whether to include parent commands
        doNotWaitForMissingThreadFiles : bool, default False
            Whether to wait for missing thread files

        Returns
        -------
        list
            List of commands to execute
        """
        # cache for this function
        isComplete = self.isComplete

        if (not self.dataAvailable) and (self.config.end == "today"):
            log.warning(
                f"{self.case} {self.relatives}: no data found (yet?) in {self.fn.fnamesPattern.level0txt}"
            )
            return []

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
            commands = self.generateCommands(
                skipExisting=skipExisting,
                doNotWaitForMissingThreadFiles=doNotWaitForMissingThreadFiles,
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
                    doNotWaitForMissingThreadFiles=doNotWaitForMissingThreadFiles,
                )
        self.commands = list(set(commands))
        if (len(self.commands) == 0) and (withParents):
            log.warning(
                f"{self.level} {self.camera} {self.case} no commands created",
            )
        return self.commands

    def generateCommands(
        self, skipExisting=True, nCPU=1, bin=None, doNotWaitForMissingThreadFiles=False
    ):
        """Generate commands for this data product.

        Parameters
        ----------
        skipExisting : bool, default True
            Whether to skip existing files
        nCPU : int, default 1
            Number of CPU cores to use
        bin : str, optional
            Python binary path
        doNotWaitForMissingThreadFiles : bool, default False
            Whether to wait for missing thread files

        Returns
        -------
        list
            List of commands to execute
        """
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
            if doNotWaitForMissingThreadFiles:
                extraStr = "1"
            else:
                extraStr = ""
            return self.commandTemplateL1(
                originLevel,
                call,
                skipExisting=skipExisting,
                nCPU=nCPU,
                bin=bin,
                extraStr=extraStr,
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
        # elif self.level == "level1shape":
        #     originLevel = "level1track"
        #     call = "particleshape.classifyParticles"
        #     return self.commandTemplateL1(
        #         originLevel, call, skipExisting=skipExisting, nCPU=nCPU, bin=bin
        #     )
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
            raise ValueError(f"Do not understand {self.level}")

    def commandTemplateL1(
        self,
        originLevel,
        call,
        skipExisting=True,
        nCPU=1,
        bin=None,
        extraOrigin=None,
        extraStr="",
    ):
        """Template for generating L1 commands.

        Parameters
        ----------
        originLevel : str
            Origin level for the command
        call : str
            Function to call
        skipExisting : bool, default True
            Whether to skip existing files
        nCPU : int, default 1
            Number of CPU cores to use
        bin : str, optional
            Python binary path
        extraOrigin : str, optional
            Extra origin level
        extraStr : str, default ""
            Extra string parameter

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

            command = f"{bin} -m VISSSlib {call}  {pName} {self.config.filename} {extraStr}"
            if nCPU is not None:
                command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {command}"
            commands.append((command, outFile))
        return commands

    def commandTemplateDaily(self, call, skipExisting=True, nCPU=1, bin=None):
        """Template for generating daily commands.

        Parameters
        ----------
        call : str
            Function to call
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
        if skipExisting and (len(exisiting) >= 1) and (self.youngerThanParents):
            log.info(f"{self.relatives} skip exisiting {exisiting[0]}")
            return []

        command = f"{bin} -m VISSSlib {call} {self.config.filename} {case} {skipExisitingInt}"
        if nCPU is not None:
            command = f"export OPENBLAS_NUM_THREADS={nCPU}; export MKL_NUM_THREADS={nCPU}; export NUMEXPR_NUM_THREADS={nCPU}; export OMP_NUM_THREADS={nCPU}; {command}"
        commands = [(command, outFile)]
        return commands

    def submitCommands(
        self,
        skipExisting=True,
        checkForDuplicates=False,
        withParents=True,
        runWorkers=False,
        doNotWaitForMissingThreadFiles=False,
    ):
        """Submit commands to the task queue.

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
        doNotWaitForMissingThreadFiles : bool, default False
            Whether to wait for missing thread files
        """
        if len(self.commands) == 0:
            self.generateAllCommands(
                skipExisting=skipExisting,
                withParents=withParents,
                doNotWaitForMissingThreadFiles=doNotWaitForMissingThreadFiles,
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

    def runWorkers(self, nJobs=os.cpu_count()):
        """Run worker processes.

        Parameters
        ----------
        nJobs : int, default os.cpu_count()
            Number of jobs to run
        """
        tools.workers(self.fileQueue, nJobs=nJobs)

    def deleteQueue(self):
        """Delete all tasks from the queue."""
        log.info(f"Deleting {self.tq.enqueued} tasks")
        [self.tq.delete(t) for t in self.tq.tasks()]
        return

    @cached_property
    def isComplete(self):
        """Check if the data product is complete.

        Returns
        -------
        bool
            True if complete, False otherwise
        """
        nMissing = self.fn.nMissing(self.level)
        if nMissing > 0:
            log.info(f"{self.case} {self.relatives} {nMissing} files are missing")
        return nMissing == 0

    @cached_property
    def _youngerThanParentsDict(self):
        """Dictionary of whether this product is younger than its parents.

        Returns
        -------
        dict
            Dictionary mapping parent names to boolean values
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
    def youngerThanParents(self):
        """Check if this product is younger than all parents.

        Returns
        -------
        bool
            True if younger than all parents, False otherwise
        """
        youngerThanParents = np.all(list(self._youngerThanParentsDict.values()))
        return youngerThanParents

    @cached_property
    def parentsYoungerThanGrandparents(self):
        """Check if parents are younger than grandparents.

        Returns
        -------
        bool
            True if parents are younger than grandparents, False otherwise
        """
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
        """Get the latest modification time of files.

        Parameters
        ----------
        files : list
            List of file paths

        Returns
        -------
        float
            Latest modification time
        """
        if len(files) > 0:
            return np.max([os.path.getmtime(f) for f in files])
        else:
            return 0

    @cached_property
    def fileCreation(self):
        """Get the file creation time.

        Returns
        -------
        float
            File creation time
        """
        files = self.listFilesExt()
        return self._fileCreation(files)

    @cached_property
    def parentsComplete(self):
        """Check if all parents are complete.

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
        """Print a report about the data product.

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
    def dataAvailable(self):
        """Check if data is available.

        Returns
        -------
        bool
            True if data is available, False otherwise
        """
        return len(self.fn.listFiles("level0txt")) > 0

    @cached_property
    def allComplete(self):
        """Check if all conditions are met for completion.

        Returns
        -------
        bool
            True if all conditions are met, False otherwise
        """
        return self.isCompleteand and self.youngerThanParents and self.parentsComplete

    @cached_property
    def nFiles(self):
        """Get the number of files.

        Returns
        -------
        int
            Number of files
        """
        return len(self.fn.listFilesExt(self.level))

    def listFilesExt(self):
        """List all files with extension.

        Returns
        -------
        list
            List of file paths
        """
        return self.fn.listFilesExt(self.level)

    def listFiles(self):
        """List all files.

        Returns
        -------
        list
            List of file paths
        """
        return self.fn.listFiles(self.level)

    def listBroken(self):
        """List broken files.

        Returns
        -------
        list
            List of broken file paths
        """
        return self.fn.listBroken(self.level)

    def listNoData(self):
        """List files with no data.

        Returns
        -------
        list
            List of no-data file paths
        """
        return self.fn.listNoData(self.level)

    def cleanUpBroken(self, withParents=False, withNoData=False):
        """Clean up broken files.

        Parameters
        ----------
        withParents : bool, default False
            Whether to clean up parents too
        withNoData : bool, default False
            Whether to clean up no-data files too
        """
        for fname in self.listBroken():
            assert fname.endswith("broken.txt")
            try:
                os.remove(fname)
            except FileNotFoundError: # usally caused by caching listBroken
                log.warning(f"{fname} not found")
            else:
                log.warning(f"{fname} removed")
        if withNoData:
            for fname in self.listNoData():
                assert fname.endswith("nodata")
                try:
                    os.remove(fname)
                except FileNotFoundError: # usally caused by caching listNoData
                    log.warning(f"{fname} not found")
                else:
                    log.warning(f"{fname} removed")
        if withParents:
            for name, parent in self.parents.items():
                parent.cleanUpBroken(withParents=False, withNoData=withNoData)

    def reportBroken(self, withParents=False, returnAllInformation=True):
        """Report broken files.

        Parameters
        ----------
        withParents : bool, default False
            Whether to include parent reports
        returnAllInformation : bool, default True
            Whether to return all information

        Returns
        -------
        pandas.DataFrame
            DataFrame with broken file information
        """
        import pandas as pd

        results_data = []
        for brokenFile in self.listBroken():
            with open(brokenFile) as f:
                lines = f.readlines()
            if len(lines) == 1:
                command = "n/a"
                outfile = "n/a"
                gist = lines[0].rstrip()
                fullError = "".join(lines)
            else:
                command = lines[1][9:].split(";")[-1].strip()
                outfile = lines[2][9:].rstrip()
                gist = f"{lines[-2].rstrip(), lines[-1].rstrip()}"
                fullError = "".join(lines[4:])
            ff = files.FilenamesFromLevel(brokenFile, self.config)
            index = f"{ff.camera.split("_")[0]}_{ff.case}_{self.level}"

            # Create a dict and append it to the list
            row = {
                "index": index,
                "command": command,
                "outfile": outfile,
                "gist": gist,
                "fullError": fullError,
            }
            results_data.append(row)

        if len(results_data) == 0:
            df = pd.DataFrame(
                columns=["index", "command", "outfile", "gist", "fullError"]
            )
        else:
            df = pd.DataFrame(
                results_data,
            )

        df = df.set_index("index")

        if withParents:
            df1 = [df]
            for name, parent in self.parents.items():
                df1.append(
                    parent.reportBroken(
                        withParents=False,
                        returnAllInformation=returnAllInformation,
                    )
                )
            df = pd.concat(df1)
            # df = df.iloc[~df.index.duplicated()]
            df = df.sort_index()

        if returnAllInformation:
            return df
        else:
            return df[["command", "gist"]]

    def cleanUpDuplicates(self, withParents=False):
        """Clean up duplicate files.

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
    """Data product for all-done status."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize allDone instance.

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
    """Data product for level 2 tracking."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize level2track instance.

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
    """Data product for level 2 matching."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize level2match instance.

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
    """Data product for level 2 detection."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize level2detect instance.

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
    """Data product for level 1 tracking."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize level1track instance.

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
    """Data product for level 1 matching."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize level1match instance.

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
    """Data product for metadata rotation."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize metaRotation instance.

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
    """Data product for level 1 detection."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize level1detect instance.

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
    """Data product for metadata frames."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize metaFrames instance.

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
    """Data product for metadata events."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize metaEvents instance.

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
    """Data product for level 0."""

    def __init__(self, case, settings, fileQueue, camera="leader"):
        """Initialize level0 instance.

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


class DataProductRange(DataProduct):
    """Range of data products for multiple cases."""

    def __init__(
        self,
        level,
        cases,
        settings,
        fileQueue,
        camera,
        relatives=None,
        addRelatives=True,
        childrensRelatives=None,
    ):
        """Initialize DataProductRange instance.

        Parameters
        ----------
        level : str
            Processing level
        cases : str or list
            Case identifiers
        settings : str
            Path to settings file
        fileQueue : str or taskqueue.TaskQueue
            File queue for task management
        camera : str
            Camera identifier
        relatives : str, optional
            Relative path information
        addRelatives : bool, default True
            Whether to add relatives
        childrensRelatives : dict, optional
            Children relatives dictionary
        """
        import taskqueue

        self.cases = tools.getCaseRange(cases, settings)
        self.config = tools.readSettings(settings)

        if fileQueue is None:
            fileQueue = f"/tmp/visss_{''.join(random.choice(string.ascii_uppercase) for _ in range(10))}"
        if type(fileQueue) is str:
            self.fileQueue = fileQueue
            self.tq = taskqueue.TaskQueue(f"fq://{self.fileQueue}")
        else:
            self.tq = fileQueue
            self.fileQueue = self.tq.path.path

        self._instances = [
            DataProduct(
                level,
                case,
                self.config ,
                self.tq,
                camera,
                relatives=relatives,
                addRelatives=False,
            )
            for case in self.cases
        ]
        self.level = level
        self.camera = camera
        self.casesStr = str(cases)

        if childrensRelatives is None:
            self.childrensRelatives = tools.DictNoDefault({})
        else:
            self.childrensRelatives = childrensRelatives
        self.parents = tools.DictNoDefault({})

        if addRelatives:
            for name in self._instances[0].parentNames:
                cam, parent_level = name.split("_")
                if name in self.childrensRelatives.keys():
                    self.parents[name] = self.childrensRelatives[name]
                    continue
                self.parents[name] = DataProductRange(
                    parent_level,
                    self.cases,
                    self.config,
                    self.tq,
                    cam,
                    addRelatives=True,
                    childrensRelatives=self.childrensRelatives,
                )
                self.parents.update(self.parents[name].parents)
                self.childrensRelatives.update(self.parents)

    def __getitem__(self, key):
        """Get item by key.

        Parameters
        ----------
        key : str or int
            Key to retrieve

        Returns
        -------
        DataProduct
            Data product instance
        """
        if isinstance(key, str):
            try:
                return self._instances[self.cases.index(key)]
            except ValueError:
                raise KeyError(f"Case '{key}' not found. Available: {self.cases}")
        return self._instances[key]

    def __iter__(self):
        """Iterate over instances.

        Yields
        ------
        DataProduct
            Data product instances
        """
        return iter(self._instances)

    def __len__(self):
        """Get length of instances.

        Returns
        -------
        int
            Number of instances
        """
        return len(self._instances)

    def __dir__(self):
        """Get directory of attributes.

        Returns
        -------
        list
            List of attribute names
        """
        own = set(super().__dir__())
        instance_attrs = set(dir(self._instances[0])) if self._instances else set()
        return sorted(own | instance_attrs)

    def __getattr__(self, name):
        """Get attribute value.

        Parameters
        ----------
        name : str
            Attribute name

        Returns
        -------
        object
            Attribute value
        """
        # Guard against calls during __init__ before _instances is set
        if name == "_instances" or "_instances" not in self.__dict__:
            raise AttributeError(name)
        if not self._instances:
            raise AttributeError(name)
        attr = getattr(self._instances[0], name)
        if callable(attr):

            def multi_method(*args, **kwargs):
                results = [getattr(dp, name)(*args, **kwargs) for dp in self._instances]
                return tools._aggregate(results)

            return multi_method
        elif name == "config":  # the config is the same for all cases
            return getattr(self._instances[0], name)
        else:
            results = [getattr(dp, name) for dp in self._instances]
            return tools._aggregate(results)

    # overwrite some functions
    def listBroken(self):
        """List broken files for all instances.

        Returns
        -------
        list
            List of broken file paths
        """
        return tools._aggregate([dp.listBroken() for dp in self._instances])

    def listFiles(self):
        """List files for all instances.

        Returns
        -------
        list
            List of file paths
        """
        return tools._aggregate([dp.listFiles() for dp in self._instances])

    def listFilesExt(self):
        """List files with extension for all instances.

        Returns
        -------
        list
            List of file paths
        """
        return tools._aggregate([dp.listFilesExt() for dp in self._instances])

    def listNoData(self):
        """List no-data files for all instances.

        Returns
        -------
        list
            List of no-data file paths
        """
        return tools._aggregate([dp.listNoData() for dp in self._instances])

    def submitCommands(
        self,
        skipExisting=True,
        checkForDuplicates=False,
        withParents=True,
        runWorkers=False,
        doNotWaitForMissingThreadFiles=False,
    ):
        """Submit commands for all instances.

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
        doNotWaitForMissingThreadFiles : bool, default False
            Whether to wait for missing thread files
        """
        commands = tools._aggregate([dp.generateAllCommands(
            skipExisting=skipExisting,
            withParents=withParents,
            doNotWaitForMissingThreadFiles=doNotWaitForMissingThreadFiles,
        ) for dp in self._instances])

        if not commands:
            log.error("nothing to submit")
            return
        if checkForDuplicates:
            running = [t.args[0] for t in self.tq.tasks()]
            commands = [c for c in commands if c[0] not in running]
        log.warning(f"sending {len(commands)} commands to {self.fileQueue}")
        self.tq.insert([partial(runCommandInQueue, c) for c in commands])
        log.warning(f"{self.tq.enqueued} tasks in Queue")
        if runWorkers:
            self.runWorkers()


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
    """Submit all data processing jobs.

    Parameters
    ----------
    nDays : int or str
        Number of days or date range
    settings : str
        Path to settings file
    fileQueue : str
        File queue path
    doMetaRot : bool, default True
        Whether to do metadata rotation
    submitJobs : bool, default True
        Whether to submit jobs
    skipExisting : bool, default True
        Whether to skip existing files
    checkForDuplicates : bool, default True
        Whether to check for duplicates
    runWorkers : bool, default False
        Whether to run workers immediately
    cleanUpBroken : bool, default False
        Whether to clean up broken files
    cleanUpDuplicates : bool, default False
        Whether to clean up duplicate files

    Returns
    -------
    DataProductRange or None
        Data product range or None if not submitting jobs
    """
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
        log.warning(
            f"{sys.executable} -m VISSSlib scripts.loopCreateMetaRotation  {settings} {nDays}"
        )
        scripts.loopCreateMetaRotation(settings, skipExisting=skipExisting, nDays=nDays)

    return prod
