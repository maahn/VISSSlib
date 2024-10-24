import logging
import multiprocessing
import os
import socket
import sys
import time

import numpy as np
import psutil
import taskqueue

from . import (
    av,
    detection,
    distributions,
    files,
    fixes,
    matching,
    metadata,
    products,
    quicklooks,
    scripts,
    tools,
    tracking,
)

# to be deleted
from .tools import runCommandInQueue

log = logging.getLogger(__name__)


def main():
    # we dont wan't verbosity here

    print(
        "%s %s %i %s" % (sys.executable, sys.version, os.getpid(), socket.gethostname())
    )  # , file=sys.stderr)

    if sys.argv[1] == "scripts.loopCreateEvents":
        settings = sys.argv[2]
        nDays = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopCreateEvents(settings, skipExisting=skipExisting, nDays=nDays)

    elif sys.argv[1] == "metadata.createEvent":
        settings = sys.argv[2]
        camera, case = sys.argv[3].split("+")
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True
        metadata.createEvent(case, camera, settings, skipExisting=skipExisting)

    elif sys.argv[1] == "scripts.loopCreateMetaFrames":
        settings = sys.argv[2]
        try:
            camera, case = sys.argv[3].split("+")
        except ValueError:
            nDays = sys.argv[3]
            camera = "all"
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopCreateMetaFrames(
            settings, skipExisting=skipExisting, nDays=nDays, camera=camera
        )

    elif sys.argv[1] == "quicklooks.createLevel1detectQuicklook":
        settings = sys.argv[2]
        camera, case = sys.argv[3].split("+")
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        quicklooks.createLevel1detectQuicklook(
            case, camera, settings, skipExisting=skipExisting
        )

    elif sys.argv[1] == "quicklooks.createLevel1matchParticlesQuicklook":
        settings = sys.argv[2]
        case = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        quicklooks.createLevel1matchParticlesQuicklook(
            case, settings, skipExisting=skipExisting
        )

    elif sys.argv[1] == "quicklooks.createMetaCoefQuicklook":
        case = sys.argv[2]
        config = sys.argv[3]
        version = sys.argv[4]
        try:
            skipExisting = bool(int(sys.argv[5]))
        except IndexError:
            skipExisting = True
        quicklooks.createMetaCoefQuicklook(
            case, config, version=version, skipExisting=skipExisting
        )

    elif sys.argv[1] == "scripts.loopLevel1detectQuicklooks":
        settings = sys.argv[2]
        nDays = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopLevel1detectQuicklooks(
            settings, nDays=nDays, skipExisting=skipExisting
        )

    elif sys.argv[1] == "scripts.loopMetaCoefQuicklooks":
        settings = sys.argv[2]
        version = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopMetaCoefQuicklooks(
            settings, version=version, skipExisting=skipExisting
        )

    elif sys.argv[1] == "scripts.loopMetaFramesQuicklooks":
        settings = sys.argv[2]
        nDays = sys.argv[3]

        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopMetaFramesQuicklooks(
            settings, nDays=nDays, skipExisting=skipExisting
        )

    elif sys.argv[1] == "scripts.loopLevel0Quicklook":
        settings = sys.argv[2]
        nDays = sys.argv[3]

        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopLevel0Quicklook(settings, nDays=nDays, skipExisting=skipExisting)

    elif sys.argv[1] == "detection.detectParticles":
        fname = sys.argv[2]
        settings = sys.argv[3]

        detection.detectParticles(fname, settings)

    elif sys.argv[1] == "matching.createMetaRotation":
        settings = sys.argv[2]
        case = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        matching.createMetaRotation(case, settings, skipExisting=skipExisting)

    elif sys.argv[1] == "matching.matchParticles":
        fname = sys.argv[2]
        settings = sys.argv[3]

        matching.matchParticles(fname, settings)

    elif sys.argv[1] == "tracking.trackParticles":
        fname = sys.argv[2]
        settings = sys.argv[3]

        tracking.trackParticles(fname, settings)

    elif sys.argv[1] == "distributions.createLevel2detect":
        settings = sys.argv[2]
        camera, case = sys.argv[3].split("+")
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        distributions.createLevel2detect(
            case, settings, skipExisting=skipExisting, camera=camera
        )

    elif sys.argv[1] == "distributions.createLevel2match":
        settings = sys.argv[2]
        case = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        distributions.createLevel2match(case, settings, skipExisting=skipExisting)

    elif sys.argv[1] == "distributions.createLevel2track":
        settings = sys.argv[2]
        case = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        distributions.createLevel2track(case, settings, skipExisting=skipExisting)

    elif sys.argv[1] == "scripts.loopCreateLevel1detect":
        settings = sys.argv[2]
        nDays = sys.argv[3]

        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopCreateLevel1detect(
            settings, skipExisting=skipExisting, nDays=nDays, cameras="all", nCPU=None
        )

    elif sys.argv[1] == "scripts.loopCreateLevel1match":
        settings = sys.argv[2]
        nDays = sys.argv[3]

        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopCreateLevel1match(
            settings, skipExisting=skipExisting, nDays=nDays, nCPU=None
        )

    elif sys.argv[1] == "scripts.loopCreateLevel1track":
        settings = sys.argv[2]
        nDays = sys.argv[3]

        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopCreateLevel1track(
            settings, skipExisting=skipExisting, nDays=nDays, nCPU=None
        )

    elif sys.argv[1] == "scripts.loopCreateLevel2detect":
        settings = sys.argv[2]
        nDays = sys.argv[3]

        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopCreateLevel2detect(
            settings, skipExisting=skipExisting, nDays=nDays, nCPU=None
        )

    elif sys.argv[1] == "scripts.loopCreateLevel2match":
        settings = sys.argv[2]
        nDays = sys.argv[3]

        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopCreateLevel2match(
            settings, skipExisting=skipExisting, nDays=nDays, nCPU=None
        )

    elif sys.argv[1] == "scripts.loopCreateLevel2track":
        settings = sys.argv[2]
        nDays = sys.argv[3]

        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopCreateLevel2track(
            settings, skipExisting=skipExisting, nDays=nDays, nCPU=None
        )

    elif sys.argv[1] == "scripts.loopLevel2matchQuicklooks":
        settings = sys.argv[2]
        nDays = sys.argv[3]

        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopLevel2matchQuicklooks(
            settings, skipExisting=skipExisting, nDays=nDays
        )
    elif sys.argv[1] == "scripts.loopLevel2trackQuicklooks":
        settings = sys.argv[2]
        nDays = sys.argv[3]

        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopLevel2trackQuicklooks(
            settings, skipExisting=skipExisting, nDays=nDays
        )

    elif sys.argv[1] == "scripts.loopLevel1matchQuicklooks":
        settings = sys.argv[2]
        nDays = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopLevel1matchQuicklooks(
            settings, nDays=nDays, skipExisting=skipExisting
        )

    elif sys.argv[1] == "scripts.loopLevel1matchParticlesQuicklooks":
        settings = sys.argv[2]
        nDays = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopLevel1matchParticlesQuicklooks(
            settings, nDays=nDays, skipExisting=skipExisting
        )

    elif sys.argv[1] == "scripts.loopLevel2matchQuicklooks":
        settings = sys.argv[2]
        nDays = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopLevel2matchQuicklooks(
            settings, nDays=nDays, skipExisting=skipExisting
        )

    elif sys.argv[1] == "scripts.loopCreateMetaRotation":
        settings = sys.argv[2]
        nDays = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopCreateMetaRotation(settings, nDays=nDays, skipExisting=skipExisting)

    elif sys.argv[1] == "scripts.loopMetaRotationQuicklooks":
        settings = sys.argv[2]
        nDays = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True

        scripts.loopMetaRotationQuicklooks(
            settings, nDays=nDays, skipExisting=skipExisting
        )

    elif sys.argv[1] == "scripts.loopCreateBatch":
        settings = sys.argv[2]
        nDays = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True
        scripts.loopCreateBatch(settings, nDays=nDays, skipExisting=skipExisting)

    elif sys.argv[1] == "scripts.loopCreateBatchTest":
        settings = sys.argv[2]
        nDays = sys.argv[3]
        try:
            skipExisting = bool(int(sys.argv[4]))
        except IndexError:
            skipExisting = True
        scripts.loopCreateBatch(
            settings, nDays=nDays, skipExisting=skipExisting, products=["test"]
        )

    elif sys.argv[1] == "scripts.reportLastFiles":
        settings = sys.argv[2]
        output = scripts.reportLastFiles(settings)
        print(output)

    elif sys.argv[1] == "quicklooks.metaRotationYearlyQuicklook":
        settings = sys.argv[2]
        year = sys.argv[3]
        quicklooks.metaRotationYearlyQuicklook(year, settings)

    elif sys.argv[1] == "products.submitAll":
        settings = sys.argv[2]
        nDays = sys.argv[3]
        taskQueue = sys.argv[4]
        products.submitAll(nDays, settings, taskQueue)

    elif sys.argv[1] == "worker":
        queue = sys.argv[2]
        assert os.path.isdir(queue)
        try:
            nJobs = int(sys.argv[3])
        except IndexError:
            nJobs = os.cpu_count()
        waitTime = 60

        class TaskQueuePatched(taskqueue.TaskQueue):
            def is_empty_wait(self):
                # first delay everything if there are no jobs
                for i in range(2):
                    if not self.is_empty():
                        break
                    print(f"waiting for jobs... {i}", flush=True)
                    time.sleep(waitTime)

                # if there are really no jobs, nothing to do
                if self.is_empty():
                    return True

                # if there are jobs, check for killwitch file
                if os.path.isfile("VISSS_KILLSWITCH"):
                    print(f"{ii}, found file VISSS_KILLSWITCH, stopping", flush=True)
                    return True

                # if tehre are jobsm check for memory and wait otherwise
                while True:
                    if psutil.virtual_memory().percent < 95:
                        break
                    print(f"waiting for available memory...", flush=True)
                    time.sleep(waitTime)
                return self.is_empty()

        def worker1(ww, status, queue):
            print(f"starting worker {ww} for {queue}", flush=True)
            time.sleep(ww / 5.0)  # to avoid race conditions
            tq = TaskQueuePatched(f"fq://{queue}")
            out = None
            while True:
                if not tq.is_empty():
                    status[ww] = 1
                    try:
                        out = tq.poll(
                            verbose=True,
                            tally=True,
                            stop_fn=tq.is_empty_wait,
                            lease_seconds=2,
                            backoff_exceptions=[BlockingIOError],
                        )
                    except:
                        pass
                    finally:
                        status[ww] = 0
                else:
                    print(f"worker {ww} queueu {queue} empty", flush=True)
                if np.all([ss == 0 for ss in status]):
                    print(
                        f"do not restart worker {ww} because all empty {[status[i] for i in range(len(status))]}",
                        flush=True,
                    )
                    break
                print(
                    f"restart worker {ww} {[status[i] for i in range(len(status))]}",
                    flush=True,
                )
                time.sleep(waitTime)

            return out

        # for communication between subprocesses
        print(f"starting {nJobs} workers")
        status = multiprocessing.Array("i", [0] * nJobs)
        for ww in range(nJobs):
            x = multiprocessing.Process(
                target=worker1,
                args=(
                    ww,
                    status,
                    queue,
                ),
            )
            x.start()

    else:
        print(f"Do not understand {sys.argv[1]}")
        return 1
    return 0


sys.exit(main())
