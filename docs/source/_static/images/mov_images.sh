#!/bin/bash

rsync -av ../../../../py/problems/figs/*.png .
# ImageMagick is required
convert -delay 5 -loop 0 -layers optimize ../../../../py/problems/figs/kelvin_helmholtz/*.png kelvin_helmholtz.gif
convert -delay 5 -loop 0 -layers optimize ../../../../py/problems/figs/rayleigh_taylor/*.png rayleigh_taylor.gif
convert -delay 5 -loop 0 -layers optimize ../../../../py/problems/figs/mhd_vortex_2d/*.png mhd_vortex_2d.gif
convert -delay 5 -loop 0 -layers optimize ../../../../py/problems/figs/geomagnetosphere_3d/*.png geomagnetosphere_3d.gif