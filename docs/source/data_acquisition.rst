Setup Data acquisition software
===============================

Preparation
-----------

The data acquisition software is available at https://github.com/maahn/VISSS. 

Follow the instructions to prepare a computer and install the software. 

Establish connection to the cameras
-----------------------------------

Establishing a connection to the cameras can be a bit tricky in the beginning because VISSS data acquisition software requires fixed IPs, which is not the default configuration. The following order is recommended with new cameras:

1. Configure the network port of the computer to use "Link-Local Only" which is the default setting of the cameras.
2. Use the GUI program /home/visss/DALSA/GigeV/tools/GigeDeviceStatus to make sure the cameras are detected. Alternatively, the terminal program in /home/visss/DALSA/GigeV/tools/lsgev can be used as well. It should be installed already and available in $PATH. In any case, remember the MAC address of the camera.
3. Compile the command line program /usr/local/bin/gevipconfig and use it to assign a fixed IP to the camera (typically 192.168.100.2 for leader and 192.168.200.2 for follower) ``/usr/local/bin/gevipconfig -p $MAC $IP 255.255.255.0`` where $MAC is the MAC address of the camera, which you obtained in the step before. In theory, gevipconfig should also work when the camera is not properly detected by the computer due to a wrong network configuration. 
4. Configure the network port of the computer manually and assign the IP 192.168.100.1 (192.168.200.1) for the interface connected to the leader (follower).
5. Use GigeDeviceStatus or lsgev to check whether the camera is detected with the fixed IP. Restarting the camera and/or computer might be required. 

Create Configuration file
-------------------------

A YAML configuration file is required for the data acquisition software visss_GUI. Make sure serialnumber, interface, ip, and MAC are consistent because no sanity checks are performed.

.. code:: yaml

    # max. package size supported by your network interface. Use "ip -d link list"
    # to find appropriate value for your interface.
    maxmtu: '9216' 
    # data storage directory
    outdir: /data/ 
    # encoding options passed over to ffmpeg. crf is the quality where smaller 
    # numbers are higher quality. Beware that file size increases strongly for
    # smaller values. threads is for the number of parallel threads used. A single
    # thread cannot handle all the data but too many threads get in the way of 
    # each other so there is a sweet spot. See ffmpeg documentation for additional details.
    encoding: "-c:v libx264 -preset veryfast -crf 15 -threads 8 -pix_fmt yuv420p" 
    # abbreviation of the site
    site: NYA 
    # Number of storage threads used by the data acquisition software. 
    # Recommended is 1 for M1280 and 2 for M2050 camera due to the higher frame
    # rate. Setting it to 2 means that camera images are alternatively processed
    # by 2 independent threads, i.e. you get also 2 output files and each thread
    # starts its own ffmpeg encoding process with the number of threads defined in 
    # encoding.
    storagethreads: 2
    # Frames per second used in the output video file. Should match AcquisitionFrameRate
    # below
    fps: 275
    # for testing: store all frames no matter whether something is moving or not.
    storeallframes: False
    # create new file every X seconds
    newfileinterval: 600
    # display every xth frame in live preview image
    liveratio: 100
    # for testing: display current camera gain
    querygain: false
    # minimum brightness change to start recording. Should be 20
    minBrightchange: 20
    # lat and lon of site, only relevant if externalTrigger is used below
    latitude: 78.9
    longitude: 11.9
    # rotate image by 90 degrees
    rotateimage: False
    # define the cameras. If you operate both cameras with one computer, use a 
    # single yaml file with two cameras defined (see below). Otherwise, use a
    # yaml file on each computer with one camera each
    camera:
    - name: visss2_leader
      # camera serial number
      serialnumber: S1242799
      # set true when follower
      follower: False
      # ethernet interface connected to camera
      interface: enp4s0
      # camera IP
      ip: 192.168.100.2
      # camera mac
      mac: 00:01:0D:C5:59:C6
      # parameters send to camera. See teledyne documentation for details and options
      teledyneparameters:
        # offset of the image in X and Y
        OffsetY: 260
        OffsetX: 392
        # width and height of the sensor. Note that a height greater than 1024 px
        # is not possible with a frame rate of 275
        Width: 1280
        Height: 1024
        # camera gain. Increasing this value makes the image brighter but increases
        # noise which makes data files much larger.
        Gain: 1
        TestImageSelector: Off
        # end exposure after ExposureTime
        ExposureMode: Timed
        # camera exposure time. Note that effective exposure time is determined by 
        # backlight flash length
        ExposureTime: 120
        # configure camera test image 
        TriggerMode: Off
        # camera frame rate
        AcquisitionFrameRate: 275
        IO:
        - LineSelector: Line3
          # Send pulse on line3 to activate backlights for 60 us (effective camera 
          # exposure time). Delay of 20 us is required to make sure follower
          # camera has time to activate. Values might need to be adjusted depending
          # on setup
          outputLineSource: PulseOnStartofExposure
          outputLinePulseDuration: 60
          outputLinePulseDelay: 20
        - LineSelector: Line4
          # Send pulse on line4 to activate follower camera without delay
          outputLineSource: PulseOnStartofExposure
          outputLinePulseDuration: 30
          outputLinePulseDelay: 0
    - name: visss2_follower 
      # same configuration for follower
      serialnumber: S1242357
      follower: True
      interface: enp4s0
      ip: 192.168.200.2
      mac: 00:01:0D:C5:57:55
      teledyneparameters:
        OffsetY: 260
        OffsetX: 392
        Width: 1280
        Height: 1024
        Gain: 1
        TestImageSelector: Off
        ExposureMode: Timed
        ExposureTime: 120
        # start exposure when pulse on line1 is received
        TriggerSelector: FrameStart
        TriggerMode: On
        TriggerSource: Line1
    # configure external trigger to turn the instrument (e.g. radar) on and off depending
    # on precipitation to limit light pollution. Disabled when set to []
    externalTrigger: []
