from PIL import Image
import sys
import mandel

image_size = (300,300)
mandel_size = 16

im = Image.new("RGB", image_size)

for row in xrange(image_size[0]):
    print row
    
    float_coord = (-3.0+float(row+1)/float(image_size[0])*6.0, -1.5)
    row_pixels = mandel.get_row(float_coord[0],float_coord[1],image_size[1],3.0/image_size[1], 16)

    for column,pixel in enumerate(row_pixels):
        if pixel:
            im.putpixel( (row, column), (255,255,255))
        else:
            im.putpixel( (row, column), (0,0,0))
            

im.save("mandelbrot.png", "PNG")
