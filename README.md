# Image Painter: An Optimized Stroke-Based Algorithm for Artistic Image Stylization
**Chang-Chieh Cheng**\
Information Technology Service Center, National Yang Ming Chiao Tung University, Hsinchu, Taiwan\
Email: jameschengcs@nycu.edu.tw

## Abstract
This paper presents a novel stroke-based rendering algorithm designed to transform input photographs into painting-style images using a series of brush strokes. 
Inspired by the contemporary inclination towards abstraction in art, the proposed method incorporates techniques from non-photorealistic rendering to achieve its objective.
Traditional stroke-based rendering methods often struggle to achieve high-quality results with a limited number of strokes. 
To address this, the proposed algorithm utilizes connected-component labeling and principal component analysis to accurately initialize the rendering parameters of each stroke. 
Subsequent refinement of strokes through forward and backward optimizations further enhances the quality of the rendering. 
However, the sequential adjustment of strokes proves to be inefficient. 
To expedite computation, this paper introduces an acceleration strategy that enables simultaneous adjustment of multiple strokes. 
Experimental validation of the proposed method confirms its effectiveness in generating high-quality painting-style images with minimal strokes. 
Comparative analyses against baseline methods demonstrate superior rendering quality based on metrics such as Mean Squared Error, Structural Similarity Index, Peak Signal-to-Noise Ratio, and Learned Perceptual Image Patch Similarity. 
In conclusion, the proposed method offers a promising solution for image stylization and abstraction, with potential applications spanning digital art, entertainment, and image editing software.


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
```
python ip.py input_path output_path
              [--a=int] [--b=str] [--m=int] [--f=n] [--s=int] 
              [--tau_y=int] [--tau_x=int] 
              [--beta=float] [--p=int] [--bo=int]
              [-o] [-j]
```
#### Parameters
* *input_path*: Input filepath. The input can be an image or json file.
* *output_path*: Output filepath, *.png or *.mp4.
* *--a*: Accuracy, 0 and 7 indicate the lowest and highest accuracies respectively. The default value is 4.
* *--b*: Brush name. The default value is *watercolor1*. Ensure that all brush templates are in PNG format and are stored within the *brush* directory. The filenames, excluding extensions, serve as the brush names.
* *--m*: Maximum number of strokes. The default value is 500.
* *--f*: Frame per second (FPS). The default value is 8. 
* *--tau_x*: Width of the sub-region. By default, it is automatically determined based on m and the image width.
* *--tau_y*, Height of the sub-region. By default, it is automatically determined based on m and the image height.
* *--beta*: Difference threshold. The default value is 0.3.
* *--p*: Number of strokes for parallel computin. The default value is 4.
* *--bo*: Number of rounds of backward optimization. By default, it is automatically determined based --a.
* *-o*: Using brush template as oppacity.
* *-j*: Output json file.
  
#### Examples
```
python ip.py testdata/flower.jpeg output/flower_1.png --m=500 --b=watercolor1
python ip.py testdata/flower.jpeg output/flower_2.png --m=250 --b=chalk -o -j
python ip.py output/flower_2.json output/flower_3.mp4 --b=chalk -o --s=1 --f=20
```

#### For comparative analysis with other SBR methods:
$m=$ 50, 100, 150, 500, and 200
```
python ip.py input_path output_path --m=50 --b=CNP
python ip.py input_path output_path --m=100 --b=CNP
python ip.py input_path output_path --m=250 --b=CNP 
python ip.py input_path output_path --m=500 --b=CNP
python ip.py input_path output_path --m=2000 --b=CNP 
```
*The test dataset consisted of CelebA and the validation folder of ImageNet.* <br />
[CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) <br />
[ImageNet](https://www.image-net.org/) <br />

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
 
