from distutils.core import setup, Extension

module1 = Extension('mandel', sources = ['_mandel.c'])

setup (name = 'mandel',
       version = '1.0',
       description = 'mandelbrot',
       ext_modules = [module1])
