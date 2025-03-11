# Convert the images using JPEG under target bpps

## Prerequisites

- Python 3 (tested with Python 3.8)
- Python packages as specified by requirements.txt (`pip install -r requirements.txt`)

## How to use 

To do inference, use the following command

    python handcraft_codecs.py /home/hglee/JPEG/2f23c0fa247bd000.jpg /home/hglee/JPEG/result jp --bpp 0.1
    
`Inputpath` is the where you store the images which you want to convert. `Outputpath` is You can convert the images for several certain `bpp`s.
However, in order to adapt to your own images and your target `bpp`, there are some places where you need to change.


Notice: We can still convert images using algorithm `JPEG2000`, `WebP` and `BPG`. However, it needs you to explore it.

## Reference
@inproceedings{mentzer2018conditional1,
    Author = {Mentzer, Fabian and Agustsson, Eirikur and Tschannen, Michael and Timofte, Radu and Van Gool, Luc},
    Booktitle = {Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
    Title = {Conditional Probability Models for Deep Image Compression},
    Year = {2018}}

