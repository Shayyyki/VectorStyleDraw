# Introduction
<br>
  This work presents VectorStyleDraw, an algorithm that synthesizes vecor style drawings based on initial input image and style input image. VectorStyleDraw, like CLIPDraw, does not require any training. It use a pre-trained CLIP language-image encoder to maximize similarity between the given initial image and a generated drawing. And it also use Earth Mover Distance (rEMD) as style loss. VectorStyleDraw preserves the characteristics of the initial image and the style of the style image well, presenting a unique and interesting image result
<br>

<br>
The file <inputs> conains several initial image examples
The file <style> conains style image examples
<br>
<br>
  ![](content/res/framework.png?raw=true)
<br>

<br>
clipdraw.py: transer the initial image into the vector image
![](content/video/camel.avi?raw=true)

cliptexture.py: transer the initial image into the vector image with style from style image
![](content/video/famel_style2.avi?raw=true)

![](content/video/famel_style3.avi?raw=true)

![](content/video/famel_style6.avi?raw=true)
<br>
