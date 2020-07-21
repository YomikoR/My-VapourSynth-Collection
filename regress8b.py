### naive linear regression for 8-bit filtering on 16-bit input

import vapoursynth as vs
from functools import partial
core = vs.core

def Sep16b(clip):
    fmt8 = clip.format.replace(bits_per_sample=8).id
    hi = core.resize.Point(clip, format=fmt8)
    lo = core.std.Expr([clip,hi], ['x y 256 * -'], fmt8)
    return [hi, lo]

def Join16b(hi, lo):
    fmt16 = hi.format.replace(bits_per_sample=16).id
    return core.std.Expr([hi,lo], ['x 256 * y +'], fmt16)

def Regress8b(hi, lo, hi_fil, limit=255, crop=0):
    # lo = k * hi + b, minimize (b, b) => k = (hi, lo) / (hi, hi)
    # expected lo_diff = k * (hi_fil - hi) clamped by limit
    # result = hi_fil << 8 + lo + lo_diff
    # use crop to exclude the effect of poor border handling

    if isinstance(limit, int) or isinstance(limit, float):
        limit = [abs(limit)] * 3
    elif isinstance(limit, list) and len(limit) > 0:
        limit = [abs(l) for l in limit]
        while len(limit) < 3:
            limit.append(limit[-1])
    else:
        limit = [255] * 3
    fmt16 = hi.format.replace(bits_per_sample=16).id
    him = core.std.Expr([hi,hi_fil], ['x y = 0 x ?'])
    hi_lo = core.std.Expr([him,lo], ['x y *'], fmt16)
    hi_hi = core.std.Expr([him], ['x x *'], fmt16)
    # crop borders
    if crop > 0:
        cw = hi.width - 2 * crop
        ch = hi.height - 2 * crop
        hi_lo = core.resize.Point(hi_lo, src_left=crop, src_width=cw, width=cw, src_top=crop, src_height=ch, height=ch)
        hi_hi = core.resize.Point(hi_hi, src_left=crop, src_width=cw, width=cw, src_top=crop, src_height=ch, height=ch)
    # proceed for all planes
    np = hi.format.num_planes
    sep_res = []
    for p in range(np):
        def reg(n, f, hip, lop, hfp, limit):
            num = f[0].props['PlaneStatsAverage']
            den = f[1].props['PlaneStatsAverage']
            k = 0 if den == 0 else num / den
            return core.std.Expr([hip,lop,hfp], 'z x - {slp} * {lim} min {nlim} max y + z 256 * +'.format(slp=k, lim=limit, nlim=-limit), vs.GRAY16)
        hip = core.std.ShufflePlanes(hi, p, vs.GRAY)
        evl = core.std.FrameEval(core.std.BlankClip(hip, format=vs.GRAY16), partial(reg, hip=hip, \
                lop=core.std.ShufflePlanes(lo, p, vs.GRAY), \
                hfp=core.std.ShufflePlanes(hi_fil, p, vs.GRAY), \
                limit=limit[p]), [core.std.PlaneStats(hi_lo, plane=p), core.std.PlaneStats(hi_hi, plane=p)])
        sep_res.append(evl)
    res = core.std.ShufflePlanes(sep_res, [0] * np, hi.format.color_family)
    return res

def Filter8b(clip, filter, limit=255, crop=0):
    hi, lo = Sep16b(clip)
    hi_fil = filter(hi)
    return Regress8b(hi, lo, hi_fil, limit=limit, crop=crop)
