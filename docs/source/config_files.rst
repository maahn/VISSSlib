Configuration files
===================

The YAML configuration file collects all settings related to a certain period of a 
deployment.



.. code:: yaml

    computers:
    - VISSS3t
    - VISSS3t
    instruments:
    - leader_H2212557
    - follower_H2212528
    instrumentsOrig:
    - leader_H2212557
    - follower_H2212528
    cropImage: null
    fps: 220
    exposureTime: 0.00012
    resolution: 46
    frame_height: 1280
    frame_width: 1024
    height_offset: 64
    leader: leader_H2212557
    follower: follower_H2212528
    minMovingPixels:
    - 20
    - 10
    - 5
    - 2
    - 2
    - 2
    - 2
    threshs:
    - 20
    - 30
    - 40
    - 60
    - 80
    - 100
    - 120
    movieExtension: mkv
    nThreads: 2
    path: /projekt1/ag_maahn/data_obs/hyytiala/visss3/{level}
    pathOut: /projekt6/ag_maahn/data_obs_nobackup/hyytiala/visss3_{version}/{level}
    pathTmp: /projekt6/ag_maahn/data_obs_nobackup/hyytiala/visss3_{version}/{level}
    pathQuicklooks: /projekt6/ag_maahn/quicklooks/hyytiala/visss3/{level}
    site: hyytiala
    visssGen: visss3
    goodFiles:
    - None
    - None
    start: "2023-12-05"
    end: "today"
    name: "Hyytiälä"
    newFileInt: 600
    dataFixes: []
    model: M2050
    calibration: #based on LIM sphere calibration 11/23
      slope: 0.021409638071724385
      slope_err: 0.8203339503704683
    rotate:
      20231205-0000: #actually 0700
        transformation:
          camera_Ofz: 9.020959
          camera_phi: 0.588068
          camera_theta: -1.225703
        transformation_err:
          camera_Ofz: 0.260346
          camera_phi: 0.020125
          camera_theta: 0.020638