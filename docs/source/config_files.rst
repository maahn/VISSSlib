Configuration files
===================

The YAML configuration file collects all settings related to a certain period of a 
deployment.

Key configuration parameters:

+-------------------+----------------------------------------------------------+
| Key               | Description                                              |
+===================+==========================================================+
| computers         | List of computer hostnames where the data was collected  |
+-------------------+----------------------------------------------------------+
| fps               | Camera frame rate (frames per second)                   |
+-------------------+----------------------------------------------------------+
| resolution        | Image resolution in micrometers per pixel               |
+-------------------+----------------------------------------------------------+
| frame_height      | Image height in pixels                                  |
+-------------------+----------------------------------------------------------+
| frame_width       | Image width in pixels                                   |
+-------------------+----------------------------------------------------------+
| leader            | Camera ID for the leader camera                         |
+-------------------+----------------------------------------------------------+
| follower          | Camera ID for the follower camera                       |
+-------------------+----------------------------------------------------------+
| nThreads          | Number of processing threads to use                     |
+-------------------+----------------------------------------------------------+
| path              | Base path for input data files                          |
+-------------------+----------------------------------------------------------+
| pathOut           | Base path for output data files                         |
+-------------------+----------------------------------------------------------+
| pathQuicklooks    | Base path for quicklook/preview files                   |
+-------------------+----------------------------------------------------------+
| visssGen          | VISSS instrument generation (e.g., visss3)              |
+-------------------+----------------------------------------------------------+
| site              | Deployment site identifier three letters                |
+-------------------+----------------------------------------------------------+
| start             | Start date of deployment period (inclusive)             |
+-------------------+----------------------------------------------------------+
| end               | End date of deployment period (inclusive)               |
+-------------------+----------------------------------------------------------+
| name              | Human-readable deployment name                          |
+-------------------+----------------------------------------------------------+
| model             | Camera model name, e.g. M1280                           |
+-------------------+----------------------------------------------------------+
