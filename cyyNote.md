## Goal

My goal is to only run the grasp sampling result to compare with mine.



## Installation Troubleshoot.

- Install the pcl package on the Ubuntu 20 in a virtual environment.

  The suggested process, which is to first install the pcl using apt and the install the python package, cannot work. Following [this instruction](https://github.com/strawlab/python-pcl/pull/407) also cannot work.
  
  Solved by the following. First install the newest PCL version, then install the python3-pcl. This installed the library to the system site package. Then create the virtual environment with the inheritance of the system site packages.
  
  In a summary:
  
  ```bash
  sudo apt install libpcl-dev
  sudo apt install python3-pcl
  python3 -m venv pnGPD --system-site-packages
  ```
  
  
