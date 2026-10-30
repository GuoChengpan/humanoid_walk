#!/usr/bin/env python3
"""按帧边界无损裁短 GIF（保留原帧编码，不重新压缩）：保留前 N 帧 + 结束符 0x3B。

    /home/guo/tools_envs/mujoco_base/bin/python3 debug_probes/gif_trim.py <in.gif> <out.gif> <保留帧数>

只解析块结构（GIF 是"扩展块 / 图像描述符 + 图像数据子块"的序列），因此保留下来的帧与原来**逐字节相同**。
"""
import sys


def skip_subblocks(d, i):
    while i < len(d) and d[i] != 0x00:
        i += 1 + d[i]
    return i + 1


def frame_offsets(d):
    """返回每个图像描述符(0x2C)的起始偏移。"""
    flags = d[10]
    i = 13 + ((3 << ((flags & 7) + 1)) if flags & 0x80 else 0)
    starts = []
    while i < len(d):
        b = d[i]
        if b == 0x3B:                                   # trailer
            break
        if b == 0x21:                                   # 扩展块
            i = skip_subblocks(d, i + 2)
        elif b == 0x2C:                                 # 图像描述符
            starts.append(i)
            i = skip_subblocks(d, i + 10 + 1)           # 描述符 10B + 局部色表标志 + 数据子块
            # 上面 +1 处是 packed flags 里的局部色表；若有则再跳过
        else:
            i += 1
    return starts, i


def frame_offsets_strict(d):
    flags = d[10]
    i = 13 + ((3 << ((flags & 7) + 1)) if flags & 0x80 else 0)
    starts = []
    n = len(d)
    while i < n:
        b = d[i]
        if b == 0x3B:
            break
        if b == 0x21:                                   # 扩展块：label(1) + 子块链
            i = skip_subblocks(d, i + 2)
        elif b == 0x2C:                                 # 图像描述符: 2C + 9B + packed(1)
            starts.append(i)
            packed = d[i + 9]
            j = i + 10
            if packed & 0x80:                           # 局部色表
                j += 3 << ((packed & 7) + 1)
            j += 1                                      # LZW 最小码长
            i = skip_subblocks(d, j)                    # 图像数据子块链
        else:
            raise ValueError(f"偏移 {i} 处不是块边界（0x{b:02X}）")
    return starts, i


def trim(src, dst, keep):
    d = open(src, "rb").read()
    assert d[:3] == b"GIF", "不是 GIF"
    starts, trailer = frame_offsets_strict(d)
    total = len(starts)
    if keep >= total:
        print(f"保留帧数 {keep} ≥ 实际 {total}，原样复制")
        out = d[:trailer + 1]
    else:
        cut = starts[keep]
        out = d[:cut] + b"\x3B"
    open(dst, "wb").write(out)
    print(f"{src}\n  → {dst}   帧 {total} → {min(keep,total)}   体积 {len(d)/1e6:.1f}MB → {len(out)/1e6:.1f}MB")


if __name__ == "__main__":
    trim(sys.argv[1], sys.argv[2], int(sys.argv[3]))
