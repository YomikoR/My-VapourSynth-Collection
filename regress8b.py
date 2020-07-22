### naive linear regression for 8-bit filtering on 16-bit input

import vapoursynth as vs
from functools import partial
core = vs.core

def Sep16b(clip):
    fmt8 = clip.format.replace(bits_per_sample=8).id
    hi_div = core.std.Expr(clip, ['x 256 /'], fmt8)
    hi = core.std.Expr([clip,hi_div], ['y 256 * x > y 1 - y ?'], fmt8)
    lo = core.std.Expr([clip,hi], ['x y 256 * -'], fmt8)
    return [hi, lo]

def Join16b(hi, lo):
    fmt16 = hi.format.replace(bits_per_sample=16).id
    return core.std.Expr([hi,lo], ['x 256 * y +'], fmt16)

def Regress8b(hi, lo, hi_fil, limit=255, crop=0, lut_factor=12):
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
    hi_lo = core.std.Expr([him,lo], ['x y *'], format=fmt16)
    hi_hi = core.std.Expr([him], ['x x *'], format=fmt16)
    # crop borders
    if crop > 0:
        cw = hi.width - 2 * crop
        ch = hi.height - 2 * crop
        hi_lo = core.resize.Point(hi_lo, src_left=crop, src_width=cw, width=cw, src_top=crop, src_height=ch, height=ch)
        hi_hi = core.resize.Point(hi_hi, src_left=crop, src_width=cw, width=cw, src_top=crop, src_height=ch, height=ch)
    # proceed for all planes
    np = hi.format.num_planes
    sep_res = []
    sep_hi = [core.std.ShufflePlanes(hi, p, vs.GRAY) for p in range(np)]
    sep_lo = [core.std.ShufflePlanes(lo, p, vs.GRAY) for p in range(np)]
    sep_hf = [core.std.ShufflePlanes(hi_fil, p, vs.GRAY) for p in range(np)]
    sep_hls = [core.std.PlaneStats(hi_lo, plane=p) for p in range(np)]
    sep_hhs = [core.std.PlaneStats(hi_hi, plane=p) for p in range(np)]
    sep_hf16 = [core.resize.Point(sep_hf[p], format=vs.GRAY16) for p in range(np)]
    for p in range(np):
        lut_size = max(1000, limit[p] * lut_factor)
        if lut_size >= len(hi):
            def reg(n, f, hip, lop, hfp, limit):
                num = f[0].props['PlaneStatsAverage']
                den = f[1].props['PlaneStatsAverage']
                k = 0 if den == 0 else num / den
                return core.std.Expr([hip,lop,hfp], 'z x - {slp} * {lim} min {nlim} max y + z 256 * +'.format(slp=k, lim=limit, nlim=-limit), vs.GRAY16)
            evl = core.std.FrameEval(sep_hf16[p], partial(reg, hip=sep_hi[p], lop=sep_lo[p], hfp=sep_hf[p], limit=limit[p]), [sep_hls[p], sep_hhs[p]])
        elif limit[p] == 0:
            evl = sep_hf16[p]
        else:
            slp_step = limit[p] / lut_size
            evls = [core.std.Expr([sep_hi[p],sep_lo[p],sep_hf[p]], 'z x - {slp} * {lim} min {nlim} max y + z 256 * +'.format(slp=slp_step * n, lim=limit[p], nlim=-limit[p]), vs.GRAY16) for n in range(lut_size + 1)]
            def reg(n, f, evls):
                num = f[0].props['PlaneStatsAverage']
                den = f[1].props['PlaneStatsAverage']
                k = 0 if den == 0 else num / den
                idx = min(int(k / slp_step), lut_size)
                return evls[idx]
            evl = core.std.FrameEval(sep_hf16[p], partial(reg, evls=evls), [sep_hls[p], sep_hhs[p]])
        sep_res.append(evl)
    res = core.std.ShufflePlanes(sep_res, [0] * np, hi.format.color_family)
    return res

def Filter8b(clip, filter, limit=255, crop=0, lut_factor=12):
    hi, lo = Sep16b(clip)
    hi_fil = filter(hi)
    return Regress8b(hi, lo, hi_fil, limit=limit, crop=crop, lut_factor=lut_factor)
