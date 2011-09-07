import pygame
import mandel
import numpy

class FractalView(object):
    def __init__(self, pixel_size, center, extent):
        self._pixel_size = pixel_size
        self._center = center
        self._extent = extent
        self._screen = None
    def main_loop(self):

        pygame.init()
        self._screen=pygame.display.set_mode(self._pixel_size)
        self._screen.fill( (0,0,0) )
        pygame.display.flip()
        done=False
        
        self._screen.fill( (0,0,0) )
        self._replot()

        while done==False:
            for event in pygame.event.get(): # User did something
                if event.type == pygame.QUIT: # If user clicked close
                    done=True # Flag that we are done so we exit this loop
                if event.type == pygame.MOUSEBUTTONUP:
                    self._screen.fill( (0,0,0) )
                    pos = pygame.mouse.get_pos()
                    pixel_pos = pos[0:2]
                    numerical_pos = self._pixelToNumerical(pixel_pos)

                    self._center = numerical_pos
                    self._extent = (self._extent[0] / 2.0, self._extent[1] / 2.0)
                    self._replot()
                if event.type == pygame.MOUSEMOTION:
                    pixel_pos = pygame.mouse.get_pos()
                    numerical_pos = self._pixelToNumerical(pixel_pos)
                    
                    self._plot_mandelbrot(numerical_pos)
        pygame.quit()
    def _replot(self):
        all_values = []
        pixelarray=pygame.PixelArray(self._screen)

        max_value = 0
        for row in xrange(self._pixel_size[1]):
            first_pixel_pos = ( float( 0 - self._pixel_size[0]/2)/self._pixel_size[0] * self._extent[0] + self._center[0],
                                float(- row + self._pixel_size[1]/2)/self._pixel_size[1] * self._extent[1] + self._center[1])
             
            delta = float(self._extent[0]) / self._pixel_size[0]
            row_pixels = mandel.get_row(self._pixel_size[0],delta, first_pixel_pos[0], first_pixel_pos[1], 10)
            all_values.append(row_pixels)
            max_value = max(max_value, max(row_pixels))
            for column,pixel in enumerate(row_pixels):
                if pixel > 0:
                    pixelarray[column][row] = (255,255,255)
                else:
                    pixelarray[column][row] = (0,0,0)
             
            pygame.display.flip()

        for row in xrange(self._pixel_size[1]):
            for column in xrange(self._pixel_size[0]):
                color = float(all_values[row][column])/max_value*255
                pixelarray[column][row] = (color, color, color)
            pygame.display.flip()
    def _pixelToNumerical(self, pixel_pos):
        return ( float(pixel_pos[0] - self._pixel_size[0]/2)/self._pixel_size[0] * self._extent[0] + self._center[0],
                 float(- pixel_pos[1] + self._pixel_size[1]/2)/self._pixel_size[1] * self._extent[1] + self._center[1])
    def _plot_mandelbrot(self, numerical_pos):
        pixels = mandel.mandelbrot(numerical_pos[0], numerical_pos[1], 200)
        max_score = max(pixels)+1
        pixel_colors = map(int, tuple(255*numpy.array(pixels)/float(max_score)))
        pixelarray=pygame.PixelArray(self._screen)
        for row in xrange(200):
            for column in xrange(200):

                pixelarray[self._pixel_size[0]-200+column][row] = ( pixel_colors[column + row * 200], )*3
        
        pygame.display.flip()
               
f=FractalView(pixel_size = [500,500], center=(0.0,0.0), extent=(4.0, 3.0))
f.main_loop() 
