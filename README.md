# Two Loggers Gazebo Simulation
**Tutorials of the contents in this repo are all located at [Docs](https://github.com/linZHank/two_loggers/tree/master/Docs). The order is: [system_setup](https://github.com/linZHank/two_loggers/blob/master/Docs/system_setup.md)->[create_urdf_tutorial](https://github.com/linZHank/two_loggers/blob/master/Docs/create_urdf_tutorial.md)->[gazebo_ros_tutorial](https://github.com/linZHank/two_loggers/blob/master/Docs/gazebo_ros_tutorial.md)**

## Setup
- [Ubuntu 16.04](http://releases.ubuntu.com/16.04/) or [Ubuntu 18.04](http://releases.ubuntu.com/18.04/) or Ubuntu 20.04
- [ROS-Kinetic](http://wiki.ros.org/kinetic) in Ubuntu 16.04 or [ROS-Melodic](http://wiki.ros.org/melodic) in Ubuntu 18.04 or ROS-Noetic in Ubuntu 20.04
- [Python 2.7](https://www.python.org/download/releases/2.7/),or Python3
- Pytorch-gpu or cpu edition

- gazebo_ros_pkgs
``` console
sudo apt-get install ros-melodic-gazebo-ros-pkgs ros-melodic-gazebo-ros-control
```
- [Create a catkin workspace](http://wiki.ros.org/catkin/Tutorials/create_a_workspace), assume your workspace is at `~/ros_ws/`
- Clone this repo to your catkin workspace
```console
cd ~/ros_ws/src
git clone https://github.com/CHH3213/loggers_multiobjective.git
```
- Build ROS packages (`loggers_description`, `loggers_gazebo`, `loggers_control`)

``` console
cd ~/ros_ws/
catkin_make
source devel/setup.bash
```
> [Catkin Command Line Tools](https://catkin-tools.readthedocs.io/en/latest/) is a substitution to `catkin_make`.

- make sure following two lines are in your `~/.bashrc` file.
``` bash
source /opt/ros/melodic/setup.bash
source /home/firefly/ros_ws/devel/setup.bash
```
> Replace `melodic` with `kinetic` in the lines above if you are using ROS-Kinetic.

