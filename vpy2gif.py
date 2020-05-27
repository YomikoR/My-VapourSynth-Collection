# VPY -> GIF
# needs ImageMagick, probably configured with QuantumDepth=16
# usage:
# python vpy2gif.py [filter.vpy] [out.gif]
# python vpy2gif.py [filter.vpy] (filter.gif)

import os, sys
import tempfile
import runpy
import vapoursynth as vs

assert len(sys.argv) >= 2
VPY = str(sys.argv[1])
assert VPY.lower().endswith('.vpy')
OUT = VPY[:-4] + '.gif'
if len(sys.argv) >= 3:
    OUT = str(sys.argv[2])
assert OUT.lower().endswith('.gif')

# Load video clip
runpy.run_path(VPY, {}, '__vapoursynth__')
clip = vs.get_output(0)

# To RGB48
if clip.format.id != vs.RGB48:
    if clip.format.color_family == vs.RGB:
        clip = vs.core.resize.Spline36(clip, format=vs.RGB48)
    else:
        clip = vs.core.resize.Spline36(clip, format=vs.RGB48, matrix_in_s='709')

IM_convert = 'magick convert'
# modify the above for the 'convert' program, e.g.
#IM_convert = r'C:\Program Files\ImageMagick-7.0.10-Q16-HDRI\convert.exe'

# FPS
IM_fps = str(clip.fps.denominator) + r'x' + str(clip.fps.numerator)

with tempfile.TemporaryDirectory() as tmpdir:
    # Take screenshots, 16-bit slightly enhances the quality (and equally slow, I think)
    wri = vs.core.imwri.Write(clip, 'PNG48', tmpdir + os.pathsep + r'img%06d.png', compression_type='None')
    for idx in range(len(clip)):
        wri.get_frame(idx)
    # Create gif
    os.system(IM_convert + r' -delay ' + IM_fps + ' -loop 0 ' + tmpdir + os.pathsep + r'*.png' + r' -fuzz 0.5% -coalesce -layers OptimizeTransparency ' + OUT)
