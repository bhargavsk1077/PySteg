# PySteg
A python steganography tool based on singular value decomposition

Forked from another repository which can be found [here](https://github.com/rodartha/SVD_Steg) , and currently working on improving it , So still a work in progress. 

## usage 
1. create virtual env
2. install dependencies using `pip install -r requirements.txt`
3. run using `python __main__.py` and give 3 more arguments , `embed/decode` , `<imagename>.jpg` , `<text-file-name>.txt`

   example: `python3 __main__.py embed img1.jpg text1.txt`
   
Note: the image and text files should be inside the input directory, And the resulting image after embedding and the decoded text after decoding will be present in the output directory.

## Algorithm
The whole implementation is based on Bergman's method with a few changes , Check **OneBlock.py** to understand the basic algorihtm,
the main code is using a modified version of this.
