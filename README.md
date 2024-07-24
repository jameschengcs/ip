# Image Painter
**Chang-Chieh Cheng**\
Information Technology Service Center, National Yang Ming Chiao Tung University, Hsinchu, Taiwan\
Email: jameschengcs@nycu.edu.tw
<p align="center"><img src="demo/demo.png"  width="80%"> </p>
*Two variant painting styles simulated by the proposed method. The left and right images present oil sketch and watercolor styles, respectively, with each rendering utilizing 500 strokes. The input image souced from ImageNet is depicted in the top-left corner of each frame.*

### Hardware requirements
+ CUDA 11.8 compatible GPU
### Software requirements
* [connected-components-3d 3.14.1](https://pypi.org/project/connected-components-3d/)
* OpenCV 4.9
* pillow 10.3.0
* PyTorch 2.2
* scikit-learn 1.4.2
* scikit-image 0.22.0
### Usage
```python ip.py input_path output_path [--a=A] [--b=B] [--f=F] [--m=M] [--s=S] [-j] [-o]```
#### Parameters
* *input_path*, input filepath. The input can be an image or json file.
* *output_path*, output filepath, *.png or *.mp4.
* *--a*, accuracy, 0 and 7 indicate the lowest and highest accuracies respectively. Default is 4.
* *--b*, brush name. Default is *watercolor1*. Ensure that all brush templates are in PNG format and are stored within the *brush* directory. The filenames, excluding extensions, serve as the brush names.
* *--m*, maximum stroke number. Default is 500. 
* *--f*, frame per second (FPS). Default is 8. 
* *--s*, painting stroke batch size. Default is 10. 
* *-o*, using brush template as oppacity.
* *-j*, output json file.
  
#### Examples
```
python ip.py testdata/flower.jpeg output/flower_1.png --m=500 --b=watercolor1
python ip.py testdata/flower.jpeg output/flower_2.png --m=250 --b=chalk -o -j
python ip.py output/flower_2.json output/flower_3.mp4 --b=chalk -o --s=1 --f=20
```

#### json format
```
{
"shape": [height, width, channels],
"bg": [r, g, b],
"Q": [q1, q2, ..., qn] and each q = [x-coordinate, y-coodinate, width, height, orientation, opaccity, r, g, b].
}
```

#### Demo
[![](https://markdown-videos-api.jorgenkh.no/youtube/tsGGtY4C4Tk)](https://youtu.be/tsGGtY4C4Tk) [![](https://markdown-videos-api.jorgenkh.no/youtube/aN9p5iSKGAg)](https://youtu.be/aN9p5iSKGAg)
 
