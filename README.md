# VectorStyleDraw


<br>
This work presents VectorStyleDraw, an algorithm that synthesizes vecor style drawings based on initial input image and style input image. VectorStyleDraw, like CLIPDraw, does not require any training. It use a pre-trained CLIP language-image encoder to maximize similarity between the given initial image and a generated drawing. And it also use Earth Mover Distance (rEMD) as style loss. VectorStyleDraw preserves the characteristics of the initial image and the style of the style image well, presenting a unique and interesting image result.

<br>

<br>
The file <inputs> conains several initial image examples

The file <style> conains several style image examples

<br>


## Framework
![](content/res/framework.png?raw=true)


## 
clipdraw.py: transer the initial image into the vector image

![](content/res/camel.png?raw=true)
![](content/res/camel_generate.png?raw=true)

## 
cliptexture.py: transer the initial image into the vector image with style from style image

![](style/style1.jpg?raw=true)
![](content/res/camel_style1.png?raw=true)

![](style/style7.jpeg?raw=true)
![](content/res/camel_style7.png?raw=true)

