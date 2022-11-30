# yolov5_ros


<p>
   <img width = "1000" src="https://github.com/deyakovleva/yolov5_ros_wmeters/blob/master/yolov5_ros/yolov5_ros/media/meter_fo_gh.jpg"></a>
</p>


## Requirements:

### 1. Create an anaconda virtual environment

```
conda env create -f yolov5.yml
```

### 2. Edit ~/.bashrc file, set to use python3.8 in mypytorch environment

```
alias python='/home/your/anaconda3/envs/yolov5/bin/python3.8'
```

### 3. Execute after save and exit:

```
source ~/.bashrc
```

## Installation yolov5_ros

```
cd /your/catkin_ws/src

git clone https://github.com/deyakovleva/yolov5_ros_wmeters

cd yolov5_ros/yolov5

sudo pip install -r requirements.txt

cd /your/catkin_ws

catkin build

source devel/setup.bash
```


## Basic Usage

Before start, change:

- path in [line 143](https://github.com/deyakovleva/yolov5_ros_wmeters/blob/master/yolov5_ros/yolov5_ros/scripts/yolo_v5_meters.py#L143)
- if you use digits detection change path in [line 56](https://github.com/deyakovleva/yolov5_ros_wmeters/blob/master/yolov5_ros/yolov5_ros/scripts/yolo_v5_meters.py#L56)
- also see node parameters
1. Start water meter detecting node

```
roslaunch yolov5_ros yolo_v5_meters.launch
```

2. Call rosservice to crop watermeter

```
rosservice call /response_meter "{}"
```

3. Start digits detecting node

```
roslaunch yolov5_ros yolo_v5_digits.launch
```

4. Call rosservice to recieve the reading from watermeter

```
rosservice call /response_digits "{}"
```
  
Modify the parameters in the [launch file](https://github.com/deyakovleva/yolov5_ros_wmeters/blob/master/yolov5_ros/yolov5_ros/launch/yolo_v5_meters.launch)

### Node parameters

* **`image_topic`** 

    Subscribed camera topic.

* **`weights_path`** 

    Path to weights file.
    
* **`confidence`** 

    Confidence threshold for detected objects.
    


