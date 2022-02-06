# Selfdriving racing rc car, heavily inspired by Nvidia Jetracer
SLAM is in process of porting SSRG-ProSLAM to CUDA.
Path Planning is not started yet
Visual Studio is configured to start in container and when compiled and run will expose port 8765

## Foxglove is used as GUI for car and could be started
```
docker run --rm -p "8080:8080" ghcr.io/foxglove/studio:latest
```
Open Foxglove interface at http://localhost:8080/
Then Foxglove can connect to local `ws://localhost:8765`