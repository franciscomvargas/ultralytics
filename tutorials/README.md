## UltraLytics Tutorials

This folder have my annotation from [Ultralytics Tutorials](https://www.youtube.com/playlist?list=PL1FZnkj4ad1PFJTjW4mWpHZhzgJinkNV0)

### [Episode 1](https://www.youtube.com/watch?v=5ku7npMrW40&list=PL1FZnkj4ad1PFJTjW4mWpHZhzgJinkNV0&index=8)

#### Attachments
 * [1st Python Script](https://github.com/franciscomvargas/ultralytics/blob/main/tutorials/1st_script.py)


### [Episode 2](https://www.youtube.com/watch?v=o4Zd-IeMlSY&list=PL1FZnkj4ad1PFJTjW4mWpHZhzgJinkNV0&index=9)

#### Segmentation

**CLI Example**
```bash
yolo predict model=yolov8s-seg.pt source='https://ultralytics.com/images/bus.jpg'
```

#### Attachments
 * [2nd Python Script](https://github.com/franciscomvargas/ultralytics/blob/main/tutorials/2nd_script.py)


### [Episode 3](https://www.youtube.com/watch?v=o4Zd-IeMlSY&list=PL1FZnkj4ad1PFJTjW4mWpHZhzgJinkNV0&index=10)

#### YOLOv8 Train Overview with Google [colab](https://colab.research.google.com/drive/1p2iPgdp16nCyluY6-m8IBhraTx3XmKSv?usp=sharing)
 
 * Personal [Roboflow Images Dataset](https://app.roboflow.com/francisco-vargas/rubber-ducks-images/) of [rubber ducks](https://en.wikipedia.org/wiki/Rubber_duck_debugging)


### [Episode 4](https://www.youtube.com/watch?v=o4Zd-IeMlSY&list=PL1FZnkj4ad1PFJTjW4mWpHZhzgJinkNV0&index=11)

#### Run Inference on Custom YOLOv8m Model > [rubber ducks](https://app.roboflow.com/francisco-vargas/rubber-ducks-images/), from [Episode 3](#episode-3)

```bash
python3 tutorials/4th_tut/4th_script.py
```