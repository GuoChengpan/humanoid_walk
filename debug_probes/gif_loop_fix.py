#!/usr/bin/env python3
"""给 GIF 无损加上"无限循环"块（NETSCAPE2.0 / loop=0）。

imageio 默认不写这个扩展块 ⇒ 播放器只播一遍。这里**不重编码**，只在
GIF 头 + 全局色表之后插入 19 字节的应用扩展块，画质与帧数完全不变。

    /home/guo/tools_envs/mujoco_base/bin/python3 debug_probes/gif_loop_fix.py <a.gif> <b.gif> ...
    # 不带参数 = 修 README 里引用的那 8 个
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# NETSCAPE 应用扩展块：
#   21 FF | 0B | "NETSCAPE2.0" | 03 01 | loop(2B, 小端, 0=无限) | 00(块终止符)
# 注意最后一字节是 **0x00 终止符**，写成别的值会让解析器错位（少帧甚至打不开）。
BLOCK = b"\x21\xFF\x0B" + b"NETSCAPE2.0" + b"\x03\x01" + (0).to_bytes(2, "little") + b"\x00"


def patch(path):
    with open(path, "rb") as f:
        data = f.read()
    if data[:3] != b"GIF":
        return f"跳过（不是 GIF）: {path}"
    if b"NETSCAPE2.0" in data[:4096]:
        return f"已有循环块: {path}"

    flags = data[10]
    gct = (3 << ((flags & 7) + 1)) if (flags & 0x80) else 0
    off = 13 + gct
    if off >= len(data):
        return f"跳过（文件过短）: {path}"
    nxt = data[off]
    if nxt not in (0x21, 0x2C):        # 必须落在块边界：扩展块或图像描述符
        return f"跳过（结构不认识 0x{nxt:02X}）: {path}"

    out = data[:off] + BLOCK + data[off:]
    tmp = path + ".loop.tmp"
    with open(tmp, "wb") as f:
        f.write(out)
    os.replace(tmp, path)             # 原子替换，中途崩溃不会截断原文件
    return f"已加循环块 (+{len(BLOCK)}B): {path}"
def from_readme():
    p = os.path.join(ROOT, "README.md")
    if not os.path.exists(p):
        return []
    txt = open(p, encoding="utf-8").read()
    return [os.path.join(ROOT, m) for m in re.findall(r"\]\((humanoid_walk_gif/[^)]+\.gif)\)", txt)]


if __name__ == "__main__":
    targets = sys.argv[1:] or from_readme()
    if not targets:
        print("没有目标：给路径参数，或在 README 里引用 GIF")
        sys.exit(1)
    for t in targets:
        print(patch(t if os.path.isabs(t) else os.path.join(ROOT, t)))
