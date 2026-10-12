# -*- coding: utf-8 -*-
"""
Heimdall · 组会汇报 PPT 生成脚本
论文：Heimdall: Dual-Scan State-Space Modeling for Closed-Loop
      Discovery of IPv6 Fully Responsive Prefixes  (Target: IEEE ICNP)
底板：/root/paper/ppt模板.pptx  (16:9)
输出：/root/6FRP/paper/Heimdall_Sharing.pptx
风格与 make_topconf_survey_ppt.py 保持一致。
"""

import os
from pptx import Presentation
from pptx.util import Inches, Pt, Emu
from pptx.enum.shapes import MSO_SHAPE
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR

TEMPLATE  = "/root/paper/ppt模板.pptx"
OUTPUT    = "/root/6FRP/paper/Heimdall_Sharing.pptx"
FIG_DIR   = "/root/6FRP/space_count"

SLIDE_W = Emu(12192000)
SLIDE_H = Emu(6858000)

# ------------------- 配色 -------------------
C_PRIMARY = RGBColor(0x1F, 0x3A, 0x68)   # 深蓝
C_ACCENT  = RGBColor(0xE3, 0x5D, 0x00)   # 橙
C_ACCENT2 = RGBColor(0x2E, 0x86, 0xAB)   # 蓝
C_DARK    = RGBColor(0x1A, 0x1A, 0x1A)
C_GRAY    = RGBColor(0x66, 0x66, 0x66)
C_LIGHT   = RGBColor(0xF4, 0xF6, 0xFB)
C_GREEN   = RGBColor(0x2E, 0x7D, 0x32)
C_RED     = RGBColor(0xC6, 0x28, 0x28)
C_WHITE   = RGBColor(0xFF, 0xFF, 0xFF)
C_PURPLE  = RGBColor(0x6A, 0x3C, 0x9E)
C_TEAL    = RGBColor(0x00, 0x7A, 0x7A)
C_YELLOW  = RGBColor(0xF9, 0xA8, 0x25)

FONT_TITLE = "微软雅黑"
FONT_BODY  = "微软雅黑"
FONT_CODE  = "Consolas"


# ------------------- 基础工具 -------------------
def remove_all_slides(prs):
    xml_slides = prs.slides._sldIdLst
    slides_lst = list(xml_slides)
    part = prs.part
    rels_to_drop = []
    for rel in list(part.rels.values()):
        if "slide" in rel.reltype and "slideLayout" not in rel.reltype and "slideMaster" not in rel.reltype:
            rels_to_drop.append(rel.rId)
    for s in slides_lst:
        xml_slides.remove(s)
    for rId in rels_to_drop:
        try:
            part.drop_rel(rId)
        except Exception:
            pass


def set_run(r, text, size=18, bold=False, color=C_DARK, font=FONT_BODY, italic=False):
    r.text = text
    r.font.name = font
    r.font.size = Pt(size)
    r.font.bold = bold
    r.font.italic = italic
    r.font.color.rgb = color


def add_textbox(slide, left, top, width, height, text_paragraphs,
                anchor=MSO_ANCHOR.TOP):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(50000)
    tf.margin_top = tf.margin_bottom = Emu(30000)
    tf.vertical_anchor = anchor
    for i, runs in enumerate(text_paragraphs):
        if i == 0:
            p = tf.paragraphs[0]
        else:
            p = tf.add_paragraph()
        p.alignment = runs[0].get("align", PP_ALIGN.LEFT) if runs else PP_ALIGN.LEFT
        if runs and "space_after" in runs[0]:
            p.space_after = Pt(runs[0]["space_after"])
        for run_cfg in runs:
            r = p.add_run()
            set_run(r,
                    run_cfg.get("text", ""),
                    size=run_cfg.get("size", 18),
                    bold=run_cfg.get("bold", False),
                    color=run_cfg.get("color", C_DARK),
                    font=run_cfg.get("font", FONT_BODY),
                    italic=run_cfg.get("italic", False))
    return tb


def add_rect(slide, left, top, width, height, fill=C_LIGHT, line=None, radius=False):
    shape_type = MSO_SHAPE.ROUNDED_RECTANGLE if radius else MSO_SHAPE.RECTANGLE
    shp = slide.shapes.add_shape(shape_type, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
    shp.shadow.inherit = False
    return shp


def add_slide_blank(prs):
    return prs.slides.add_slide(prs.slide_layouts[6])


def add_picture(slide, path, left, top, width=None, height=None, caption=None,
                cap_color=C_GRAY, cap_size=11):
    if not os.path.exists(path):
        # 占位框
        add_rect(slide, left, top, width or Emu(4000000), height or Emu(2500000),
                 fill=C_LIGHT, line=C_GRAY)
        add_textbox(slide, left, top, width or Emu(4000000), height or Emu(2500000),
                    [[{"text": f"[缺失图片] {os.path.basename(path)}",
                       "size": 12, "color": C_RED, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        return
    kwargs = {"left": left, "top": top}
    if width is not None:
        kwargs["width"] = width
    if height is not None:
        kwargs["height"] = height
    pic = slide.shapes.add_picture(path, **kwargs)
    if caption:
        cap_top = top + pic.height + Emu(60000)
        add_textbox(slide, left, cap_top, pic.width, Emu(300000),
                    [[{"text": caption, "size": cap_size, "italic": True,
                       "color": cap_color, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.TOP)


def page_header(slide, title, section=None, page_no=None, total=None):
    add_rect(slide, Emu(0), Emu(500000), Emu(160000), Emu(700000), fill=C_ACCENT)
    add_textbox(slide, Emu(300000), Emu(380000), Emu(9200000), Emu(900000),
                [[{"text": title, "size": 26, "bold": True,
                   "color": C_PRIMARY, "font": FONT_TITLE}]],
                anchor=MSO_ANCHOR.MIDDLE)
    tag = ""
    if section:
        tag = section
    if page_no is not None and total is not None:
        tag = f"{tag}    {page_no:02d} / {total:02d}"
    if tag:
        add_textbox(slide, Emu(9500000), Emu(430000), Emu(2500000), Emu(500000),
                    [[{"text": tag, "size": 12, "color": C_GRAY, "align": PP_ALIGN.RIGHT}]],
                    anchor=MSO_ANCHOR.MIDDLE)
    add_rect(slide, Emu(300000), Emu(1350000), Emu(11500000), Emu(15000), fill=C_GRAY)


def page_footer_conclusion(slide, text):
    add_rect(slide, Emu(0), Emu(6480000), Emu(12192000), Emu(380000), fill=C_LIGHT)
    add_textbox(slide, Emu(300000), Emu(6480000), Emu(11500000), Emu(380000),
                [[{"text": "💡 " + text, "size": 14, "bold": True, "color": C_PRIMARY}]],
                anchor=MSO_ANCHOR.MIDDLE)


# =============================================================
#                          页面
# =============================================================
TOTAL = 20  # 总页数（含末尾展望&Thanks）


# ---------- P1 封面 ----------
def slide_title(prs):
    slide = add_slide_blank(prs)
    # 侧栏 badge
    add_rect(slide, Emu(0), Emu(0), Emu(3800000), Emu(860000), fill=C_PRIMARY)
    add_textbox(slide, Emu(180000), Emu(150000), Emu(3600000), Emu(600000),
                [[{"text": "组会汇报 · 6FRP 项目",
                   "size": 16, "bold": True, "color": C_WHITE}]],
                anchor=MSO_ANCHOR.MIDDLE)

    # 主标题
    add_textbox(slide, Emu(600000), Emu(1200000), Emu(11000000), Emu(1000000),
                [[{"text": "Heimdall",
                   "size": 66, "bold": True, "color": C_PRIMARY, "font": FONT_TITLE}]])
    add_textbox(slide, Emu(600000), Emu(2100000), Emu(11000000), Emu(700000),
                [[{"text": "Dual-Scan State-Space Modeling for",
                   "size": 22, "color": C_ACCENT, "font": FONT_TITLE}]])
    add_textbox(slide, Emu(600000), Emu(2500000), Emu(11000000), Emu(700000),
                [[{"text": "Closed-Loop Discovery of IPv6 Fully Responsive Prefixes",
                   "size": 22, "bold": True, "color": C_ACCENT, "font": FONT_TITLE}]])
    add_textbox(slide, Emu(600000), Emu(3200000), Emu(11000000), Emu(500000),
                [[{"text": "面向 IEEE ICNP · IPv6 主动测量 × 状态空间模型 × 闭环反馈",
                   "size": 16, "italic": True, "color": C_GRAY}]])

    # 三个亮点 badge
    badges = [
        ("双向扫描 (Dual-Scan)\n对齐位轴 + 同AS样本轴", C_PRIMARY),
        ("三阶段掩码填充\n线性时间 · 非自回归", C_ACCENT),
        ("闭环在线反馈\nAPD结果 → 在线微调", C_GREEN),
    ]
    bw, bh, gap = Emu(3500000), Emu(1000000), Emu(200000)
    total_w = bw * 3 + gap * 2
    x0 = (SLIDE_W - total_w) // 2
    y0 = Emu(4000000)
    for i, (name, color) in enumerate(badges):
        x = x0 + i * (bw + gap)
        add_rect(slide, x, y0, bw, bh, fill=color, radius=True)
        add_textbox(slide, x, y0, bw, bh,
                    [[{"text": name, "size": 15, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)

    # 关键指标条
    add_rect(slide, Emu(600000), Emu(5350000), Emu(11000000), Emu(650000),
             fill=C_LIGHT, line=C_GRAY)
    add_textbox(slide, Emu(700000), Emu(5350000), Emu(10800000), Emu(650000),
                [[{"text": "1.9× 发现吞吐  ·  1.2× AS 覆盖  ·  214K 种子 / 299 AS  ·  1.7× 训练加速",
                   "size": 18, "bold": True, "color": C_PRIMARY,
                   "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)

    # 页脚
    add_rect(slide, Emu(0), Emu(6300000), Emu(12192000), Emu(558000), fill=C_PRIMARY)
    add_textbox(slide, Emu(600000), Emu(6300000), Emu(11000000), Emu(558000),
                [[{"text": "分享人：谷雨宸    ·    组会分享    ·    2026 / 09",
                   "size": 14, "bold": True, "color": C_WHITE}]],
                anchor=MSO_ANCHOR.MIDDLE)


# ---------- P2 目录 ----------
def slide_toc(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "汇报大纲", section="Outline", page_no=2, total=TOTAL)

    items = [
        ("01", "背景与动机", "IPv6 主动测量 · FRP 的危害 · 已有方案", C_PRIMARY),
        ("02", "问题分析", "Pattern Entropy · Gungnir 的 3 大局限 (L1/L2/L3)", C_ACCENT),
        ("03", "Heimdall 设计", "Dual-Scan HMamba · 层次化 MLM · 三阶段推理", C_ACCENT2),
        ("04", "闭环在线扫描", "双向 SeedFilter · 在线微调 · 自适应温度", C_GREEN),
        ("05", "实验评估", "端到端发现 · 时间 · 闭环增益 · 消融", C_PURPLE),
        ("06", "长期观测发现", "3 个月多协议扫描 · 集中度 / 服务错配 / 安全信号", C_TEAL),
        ("07", "新学期工作展望", "投稿 · 系统开源 · 后续方向", C_RED),
    ]
    left = Emu(700000)
    top0 = Emu(1600000)
    row_h = Emu(650000)
    for i, (idx, title, sub, color) in enumerate(items):
        y = top0 + row_h * i
        add_rect(slide, left, y, Emu(700000), Emu(550000), fill=color, radius=True)
        add_textbox(slide, left, y, Emu(700000), Emu(550000),
                    [[{"text": idx, "size": 22, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        add_textbox(slide, left + Emu(850000), y, Emu(4500000), Emu(550000),
                    [[{"text": title, "size": 20, "bold": True, "color": C_PRIMARY}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        add_textbox(slide, left + Emu(5400000), y, Emu(6300000), Emu(550000),
                    [[{"text": sub, "size": 14, "color": C_GRAY, "italic": True}]],
                    anchor=MSO_ANCHOR.MIDDLE)


# ---------- P3 背景：IPv6 主动测量 & FRP 简介 ----------
def slide_background(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "背景 · IPv6 主动测量与 FRP 问题",
                section="01 · Background", page_no=3, total=TOTAL)

    # 左：IPv4 vs IPv6 对比
    add_rect(slide, Emu(400000), Emu(1550000), Emu(5500000), Emu(2200000),
             fill=C_LIGHT, line=C_PRIMARY, radius=True)
    add_textbox(slide, Emu(500000), Emu(1650000), Emu(5300000), Emu(500000),
                [[{"text": "从 IPv4 到 IPv6：扫描范式的根本转变",
                   "size": 18, "bold": True, "color": C_PRIMARY}]])
    lines_l = [
        ("• IPv4：2³² 空间，ZMap 40 分钟扫全网 (穷举可行)", C_DARK),
        ("• IPv6：2¹²⁸ 空间，物理上无法穷举", C_DARK),
        ("• 必须依赖 Target Generation Algorithm (TGA)", C_ACCENT),
        ("   —— 6Forest / 6Hit / 6Graph / DET …", C_GRAY),
        ("• TGA 的一切都取决于种子 (seed) 质量", C_RED),
    ]
    y = Emu(2200000)
    for txt, col in lines_l:
        add_textbox(slide, Emu(500000), y, Emu(5300000), Emu(280000),
                    [[{"text": txt, "size": 14, "color": col}]])
        y += Emu(280000)

    # 右：FRP 定义
    add_rect(slide, Emu(6200000), Emu(1550000), Emu(5600000), Emu(2200000),
             fill=C_LIGHT, line=C_ACCENT, radius=True)
    add_textbox(slide, Emu(6300000), Emu(1650000), Emu(5400000), Emu(500000),
                [[{"text": "Fully Responsive Prefix (FRP)",
                   "size": 18, "bold": True, "color": C_ACCENT}]])
    lines_r = [
        ("• 定义：前缀内任意地址探测都返回响应", C_DARK),
        ("• 100% probe-reply rate（但未必是不同主机）", C_GRAY),
        ("• 成因：CDN 负载均衡 / 蜜罐 / NAT 代理池 /", C_DARK),
        ("        Linux ip_nonlocal_bind sysctl 掩护", C_GRAY),
        ("• 危害：污染 TGA 训练、膨胀 hitlist、", C_RED),
        ("       浪费探测预算、误导拓扑推断", C_RED),
    ]
    y = Emu(2150000)
    for txt, col in lines_r:
        add_textbox(slide, Emu(6300000), y, Emu(5400000), Emu(260000),
                    [[{"text": txt, "size": 14, "color": col}]])
        y += Emu(260000)

    # 下方：为什么此题很重要
    add_rect(slide, Emu(400000), Emu(3900000), Emu(11400000), Emu(2400000),
             fill=C_WHITE, line=C_PRIMARY)
    add_textbox(slide, Emu(500000), Emu(4000000), Emu(11200000), Emu(500000),
                [[{"text": "▶ 为什么 FRP 发现是 IPv6 主动测量的关键前置任务",
                   "size": 18, "bold": True, "color": C_PRIMARY}]])
    reasons = [
        "① 从种子集中剔除 FRP，可使 6Forest / 6Graph 等 TGA 的命中率提升数倍",
        "② FRP 本身即安全研究对象：负载均衡拓扑、蜜罐部署、代理池 IP 池等",
        "③ 若不识别 FRP，主动扫描的『发现』大量是重复响应，浪费探测预算",
        "④ 已成为 IMC / SIGCOMM / ICNP 等测量顶会持续关注的方向",
    ]
    y = Emu(4550000)
    for r in reasons:
        add_textbox(slide, Emu(500000), y, Emu(11200000), Emu(360000),
                    [[{"text": r, "size": 15, "color": C_DARK}]])
        y += Emu(400000)

    page_footer_conclusion(slide,
        "识别并剔除 FRP，是所有下游 IPv6 主动测量任务的基础前置步骤")


# ---------- P4 已有工作 · MAPD / Luori / Gungnir ----------
def slide_related(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "相关工作 · FRP 发现三代技术路线",
                section="01 · Related Work", page_no=4, total=TOTAL)

    cards = [
        {
            "title": "① MAPD / APD",
            "venue": "IMC 2018 · Gasser et al.",
            "color": C_GRAY,
            "method": "枚举固定长度 (通常 /48) 的所有 BGP 前缀，\n每个前缀随机采 16 个地址做 APD 探测",
            "pros":  "简单可靠 · 长度小的时候完备",
            "cons":  "组合爆炸 · 无法发现任意粒度 FRP",
        },
        {
            "title": "② Luori",
            "venue": "IEEE ICNP 2024 · Cheng et al.",
            "color": C_ACCENT2,
            "method": "MCTS 蒙特卡洛树搜索 IPv6 地址树\n逐 nibble 展开 → 每协议/端口一棵树",
            "pros":  "支持任意粒度 · 显式建模空间探索",
            "cons":  "长度耦合 · 多协议扩展成本 = N×",
        },
        {
            "title": "③ Gungnir (SOTA)",
            "venue": "IEEE ICNP 2025 · Wei et al.",
            "color": C_ACCENT,
            "method": "自回归 Transformer 生成 128 bit\n条件：AS / ORG / Category / Proto",
            "pros":  "27× 快于 Luori · 4.2× 覆盖 · 首个生成式方案",
            "cons":  "单轴注意力 · O(L²) · 离线开环",
        },
    ]
    left0 = Emu(400000)
    card_w = Emu(3700000)
    gap = Emu(280000)
    top = Emu(1500000)
    card_h = Emu(4600000)
    for i, c in enumerate(cards):
        x = left0 + i * (card_w + gap)
        add_rect(slide, x, top, card_w, card_h, fill=C_WHITE, line=c["color"], radius=True)
        # 顶部彩条
        add_rect(slide, x, top, card_w, Emu(700000), fill=c["color"], radius=True)
        add_textbox(slide, x, top, card_w, Emu(400000),
                    [[{"text": c["title"], "size": 18, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        add_textbox(slide, x, top + Emu(400000), card_w, Emu(300000),
                    [[{"text": c["venue"], "size": 11, "italic": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        # 方法
        y = top + Emu(850000)
        add_textbox(slide, x + Emu(100000), y, card_w - Emu(200000), Emu(300000),
                    [[{"text": "▶ 核心方法", "size": 13, "bold": True, "color": c["color"]}]])
        add_textbox(slide, x + Emu(100000), y + Emu(320000), card_w - Emu(200000), Emu(900000),
                    [[{"text": c["method"], "size": 13, "color": C_DARK}]])
        # 优点
        y2 = y + Emu(1400000)
        add_textbox(slide, x + Emu(100000), y2, card_w - Emu(200000), Emu(300000),
                    [[{"text": "✅ 优点", "size": 13, "bold": True, "color": C_GREEN}]])
        add_textbox(slide, x + Emu(100000), y2 + Emu(320000), card_w - Emu(200000), Emu(600000),
                    [[{"text": c["pros"], "size": 13, "color": C_DARK}]])
        # 缺点
        y3 = y2 + Emu(1050000)
        add_textbox(slide, x + Emu(100000), y3, card_w - Emu(200000), Emu(300000),
                    [[{"text": "❌ 局限", "size": 13, "bold": True, "color": C_RED}]])
        add_textbox(slide, x + Emu(100000), y3 + Emu(320000), card_w - Emu(200000), Emu(600000),
                    [[{"text": c["cons"], "size": 13, "color": C_DARK}]])

    page_footer_conclusion(slide,
        "Gungnir 是当前 SOTA，但仍留有 3 个未开发的机会 —— 这正是 Heimdall 的切入点")


# ---------- P5 Gungnir 的三大局限 (Motivation) ----------
def slide_motivation(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "动机 · Gungnir 尚未解决的 3 大问题",
                section="02 · Motivation", page_no=5, total=TOTAL)

    add_textbox(slide, Emu(400000), Emu(1500000), Emu(11400000), Emu(400000),
                [[{"text": "► 我们观察到当前 SOTA 仍有以下三处结构性欠缺：",
                   "size": 16, "italic": True, "color": C_GRAY}]])

    lessons = [
        {
            "tag": "L1", "color": C_PRIMARY,
            "title": "同 AS 内的 FRP 相关信号被浪费",
            "detail": ("同一 AS 的 FRP 通常共享 nibble 分配 / 前缀长度分布，\n"
                       "但 Transformer 只在一条前缀内做 self-attention，\n"
                       "无法在训练时看到同 AS 的其他 FRP —— 跨样本信号丢失"),
        },
        {
            "tag": "L2", "color": C_ACCENT,
            "title": "自注意力二次复杂度",
            "detail": ("Attention 是 O(L²) —— 128 bit + 多字段元数据的长序列下,\n"
                       "预训练与推理都被拖慢；\n"
                       "GPU 内存也难以支撑更长上下文"),
        },
        {
            "tag": "L3", "color": C_GREEN,
            "title": "生成是开环的，无法在线纠错",
            "detail": ("Gungnir 训练完之后一次性生成全部候选,\n"
                       "APD 探测反馈只用于事后粒度修正 (granularity)；\n"
                       "错误的前缀模式会反复重复生成，直到人工重灌种子"),
        },
    ]

    left0 = Emu(400000)
    w = Emu(3700000)
    gap = Emu(280000)
    top = Emu(2050000)
    h = Emu(3800000)
    for i, l in enumerate(lessons):
        x = left0 + i * (w + gap)
        add_rect(slide, x, top, w, h, fill=C_WHITE, line=l["color"], radius=True)
        # 大标签
        add_rect(slide, x + Emu(100000), top + Emu(150000), Emu(700000), Emu(700000),
                 fill=l["color"], radius=True)
        add_textbox(slide, x + Emu(100000), top + Emu(150000), Emu(700000), Emu(700000),
                    [[{"text": l["tag"], "size": 26, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        # 标题
        add_textbox(slide, x + Emu(850000), top + Emu(150000),
                    w - Emu(950000), Emu(700000),
                    [[{"text": l["title"], "size": 15, "bold": True, "color": l["color"]}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        # 详情
        add_textbox(slide, x + Emu(150000), top + Emu(1000000),
                    w - Emu(300000), h - Emu(1150000),
                    [[{"text": l["detail"], "size": 13, "color": C_DARK}]])

    page_footer_conclusion(slide,
        "L1 → 加入 sample-axis 扫描 ｜ L2 → 用 Mamba SSM 取代 Attention ｜ L3 → 生成变成闭环")


# ---------- P6 Pattern Entropy（插入 entropy.png） ----------
def slide_entropy(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "动机数据 · 同 AS 内 FRP 高度聚类",
                section="02 · Motivation", page_no=6, total=TOTAL)

    # 左：图
    add_picture(slide,
                os.path.join(FIG_DIR, "entropy.png"),
                left=Emu(400000), top=Emu(1550000),
                width=Emu(6500000),
                caption="Fig · 214K 种子 FRP 上按不同维度切片后的 pattern entropy")

    # 右：解读
    add_rect(slide, Emu(7100000), Emu(1550000), Emu(4700000), Emu(4700000),
             fill=C_LIGHT, line=C_PRIMARY, radius=True)
    add_textbox(slide, Emu(7250000), Emu(1650000), Emu(4400000), Emu(500000),
                [[{"text": "关键观察",
                   "size": 20, "bold": True, "color": C_PRIMARY}]])

    y = Emu(2150000)
    items = [
        ("• 全体 FRP 的 pattern entropy",  C_DARK),
        ("  已远低于 uniform 的 log₂ 理论上限", C_GRAY),
        ("", None),
        ("• 按 CATEGORY / ORG / AS 切片后", C_DARK),
        ("  熵进一步下降，AS 维度下降最多", C_ACCENT),
        ("", None),
        ("• 说明：同 AS 内的 FRP 有极强的",  C_DARK),
        ("  『共享模式』—— 是可利用的结构信息", C_GREEN),
        ("", None),
        ("• Transformer 单条前缀训练 → 看不到", C_RED),
        ("  这条信号；必须加 sample-axis 扫描", C_RED),
    ]
    for txt, col in items:
        if col is None:
            y += Emu(120000)
            continue
        add_textbox(slide, Emu(7250000), y, Emu(4400000), Emu(300000),
                    [[{"text": txt, "size": 14, "color": col,
                       "bold": (col in (C_ACCENT, C_GREEN, C_RED))}]])
        y += Emu(320000)

    page_footer_conclusion(slide,
        "同 AS 内 FRP 高度聚类 —— 这是 Dual-Scan 的直接经验依据")


# ---------- P7 Heimdall 三大创新 (Overview) ----------
def slide_overview_ideas(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "Heimdall · 三大关键设计",
                section="03 · Design", page_no=7, total=TOTAL)

    ideas = [
        {
            "tag": "①", "color": C_PRIMARY,
            "title": "Dual-Scan Hierarchical Mamba",
            "vs":    "解决 L1 & L2",
            "detail": ("• Outer scan：沿 128 bit 位轴 (within-prefix)\n"
                       "• Inner scan：沿同 AS 样本轴 (cross-prefix)\n"
                       "• AS-conditioned gate 融合两个扫描\n"
                       "• Mamba S6 状态空间模型 → O(L) 线性复杂度"),
        },
        {
            "tag": "②", "color": C_ACCENT,
            "title": "层次化 MLM · 三阶段生成",
            "vs":    "对齐 IPv6 分层地址",
            "detail": ("• Coarse [0,32)  · Medium [32,64) · Fine [64,128)\n"
                       "• 每阶段并行掩码填充 (非自回归)\n"
                       "• 按位 Shannon-Entropy 加权 loss\n"
                       "• 三次前向 vs 128 步自回归 → 更快更准"),
        },
        {
            "tag": "③", "color": C_GREEN,
            "title": "闭环在线扫描",
            "vs":    "解决 L3",
            "detail": ("• 双向 SeedFilter：L1 去重 + L2 in-range 丢弃\n"
                       "• 在线微调：APD 阳性/阴性反馈成为新训练信号\n"
                       "• 自适应温度：卡在已知区就升温、发现新贵矿则降温\n"
                       "• Confirmed-FRP 自增：闭环持续扩展 known-set"),
        },
    ]

    top = Emu(1550000)
    left = Emu(400000)
    w = Emu(11400000)
    h = Emu(1500000)
    gap = Emu(150000)
    for i, idea in enumerate(ideas):
        y = top + i * (h + gap)
        add_rect(slide, left, y, w, h, fill=C_WHITE, line=idea["color"], radius=True)
        # 左侧数字块
        add_rect(slide, left + Emu(100000), y + Emu(100000),
                 Emu(1000000), h - Emu(200000),
                 fill=idea["color"], radius=True)
        add_textbox(slide, left + Emu(100000), y + Emu(100000),
                    Emu(1000000), h - Emu(200000),
                    [[{"text": idea["tag"], "size": 60, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        # 标题 + 副
        add_textbox(slide, left + Emu(1250000), y + Emu(100000),
                    Emu(6500000), Emu(500000),
                    [[{"text": idea["title"], "size": 20, "bold": True, "color": idea["color"]}]])
        add_textbox(slide, left + Emu(8000000), y + Emu(150000),
                    Emu(3300000), Emu(400000),
                    [[{"text": f"◆ {idea['vs']}", "size": 13, "italic": True,
                       "color": C_GRAY, "align": PP_ALIGN.RIGHT}]])
        # 详情
        add_textbox(slide, left + Emu(1250000), y + Emu(600000),
                    w - Emu(1400000), h - Emu(700000),
                    [[{"text": idea["detail"], "size": 13, "color": C_DARK}]])

    page_footer_conclusion(slide,
        "把 Transformer 自回归 → 换成 Mamba 双扫描；把开环生成 → 换成闭环强化")


# ---------- P8 系统总览（插入 arch_dualscan.png）----------
def slide_overview(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "系统总览 · Semantic → Inference → Closed-Loop",
                section="03 · Design", page_no=8, total=TOTAL)

    # 顶部三阶段说明
    stages = [
        ("① Semantic Learning", "DS-HMamba 离线预训练 + 熵加权 MLM", C_PRIMARY),
        ("② Prefix Inference",   "三阶段掩码填充 · BGP 锁定 · 温度控制",   C_ACCENT),
        ("③ Closed-Loop Scanning", "APD 探测 + SeedFilter + 在线微调 + 温度自适应", C_GREEN),
    ]
    left = Emu(400000)
    w = Emu(3700000)
    gap = Emu(280000)
    y = Emu(1550000)
    for i, (t, s, c) in enumerate(stages):
        x = left + i * (w + gap)
        add_rect(slide, x, y, w, Emu(750000), fill=c, radius=True)
        add_textbox(slide, x, y, w, Emu(400000),
                    [[{"text": t, "size": 17, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        add_textbox(slide, x, y + Emu(400000), w, Emu(350000),
                    [[{"text": s, "size": 11, "color": C_WHITE,
                       "italic": True, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)

    # 主架构图（保持宽高比，居中）
    add_picture(slide,
                os.path.join(FIG_DIR, "arch_dualscan.png"),
                left=Emu(400000), top=Emu(2500000),
                width=Emu(11400000),
                caption="Fig · Heimdall 系统架构：Seed Store → DS-HMamba → SeedFilter → APD → Online Fine-tuner")

    page_footer_conclusion(slide,
        "红色虚线 = APD 反馈通道，把开环模型『强化』成闭环强化学习式的迭代系统")


# ---------- P9 DS-HMamba 模型细节 ----------
def slide_dshmamba(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "DS-HMamba · Dual-Scan Hierarchical Mamba",
                section="03 · Design", page_no=9, total=TOTAL)

    # 左侧：Outer scan
    add_rect(slide, Emu(400000), Emu(1550000), Emu(5500000), Emu(2100000),
             fill=C_WHITE, line=C_PRIMARY, radius=True)
    add_rect(slide, Emu(400000), Emu(1550000), Emu(5500000), Emu(450000),
             fill=C_PRIMARY, radius=True)
    add_textbox(slide, Emu(400000), Emu(1550000), Emu(5500000), Emu(450000),
                [[{"text": "Outer Scan · 位轴 (within-prefix)",
                   "size": 16, "bold": True, "color": C_WHITE,
                   "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)
    outer = [
        "• 沿 128 bit 位轴运行标准 Mamba S6 block",
        "• 输入依赖的 Δ / A / B / C  → 线性选择扫描",
        "• 捕捉 bit-by-bit 位间依赖 (与 Gungnir 同轴)",
        "• 复杂度 O(L·D·N)  ← 与 Attention 的 O(L²) 对比",
    ]
    y = Emu(2100000)
    for t in outer:
        add_textbox(slide, Emu(550000), y, Emu(5200000), Emu(300000),
                    [[{"text": t, "size": 13, "color": C_DARK}]])
        y += Emu(320000)

    # 右侧：Inner scan
    add_rect(slide, Emu(6200000), Emu(1550000), Emu(5600000), Emu(2100000),
             fill=C_WHITE, line=C_ACCENT, radius=True)
    add_rect(slide, Emu(6200000), Emu(1550000), Emu(5600000), Emu(450000),
             fill=C_ACCENT, radius=True)
    add_textbox(slide, Emu(6200000), Emu(1550000), Emu(5600000), Emu(450000),
                [[{"text": "Inner Scan · 同 AS 样本轴 (cross-prefix)",
                   "size": 16, "bold": True, "color": C_WHITE,
                   "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)
    inner = [
        "• 按 AS 分组，同组 G_a 个前缀在每个 bit 位置",
        "  运行一个 Inner Mamba 块",
        "• 输出：peer-AS 的 nibble / 长度 / 子前缀共享模式",
        "• 单例组退化为 identity —— 无副作用",
    ]
    y = Emu(2100000)
    for t in inner:
        add_textbox(slide, Emu(6350000), y, Emu(5300000), Emu(300000),
                    [[{"text": t, "size": 13, "color": C_DARK}]])
        y += Emu(320000)

    # 下：AS-conditioned Gate
    add_rect(slide, Emu(400000), Emu(3800000), Emu(11400000), Emu(1700000),
             fill=C_LIGHT, line=C_GREEN, radius=True)
    add_textbox(slide, Emu(500000), Emu(3900000), Emu(11200000), Emu(500000),
                [[{"text": "▶ AS-Conditioned Gating Fusion",
                   "size": 17, "bold": True, "color": C_GREEN}]])
    formulas = [
        "  [gout, gin] = softmax(Wg · SiLU(Wg′ · c))       ← c = AS 元数据条件向量",
        "  X(ℓ+1) = LayerNorm( gout · Y_outer + gin · Y_inner + α · X(ℓ) )    ← α 可学",
        "  堆叠 N 层 DS-HMamba → 线性头输出 128×2 位 logits",
    ]
    y = Emu(4400000)
    for f in formulas:
        add_textbox(slide, Emu(600000), y, Emu(11000000), Emu(300000),
                    [[{"text": f, "size": 13, "color": C_DARK, "font": FONT_CODE}]])
        y += Emu(320000)

    # 底部：为什么这样设计
    add_textbox(slide, Emu(400000), Emu(5700000), Emu(11400000), Emu(600000),
                [[{"text": "▶ 让模型自适应地决定：这个 AS 更依赖 within-prefix 还是 cross-prefix 结构",
                   "size": 15, "bold": True, "color": C_PRIMARY, "italic": True}]])

    page_footer_conclusion(slide,
        "两个正交扫描 + AS-gate 融合 = 首个在 IPv6 生成任务上引入 SSM 的工作")


# ---------- P10 层次化 MLM 训练 + 三阶段推理 ----------
def slide_training(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "训练 · 层次化 MLM ＋ 熵加权 · 三阶段推理",
                section="03 · Design", page_no=10, total=TOTAL)

    # 上部：层次结构（对齐 RFC 4291）
    add_rect(slide, Emu(400000), Emu(1550000), Emu(11400000), Emu(1200000),
             fill=C_LIGHT, line=C_PRIMARY, radius=True)
    add_textbox(slide, Emu(500000), Emu(1600000), Emu(11200000), Emu(400000),
                [[{"text": "▶ IPv6 分层 (RFC 4291) → Coarse / Medium / Fine",
                   "size": 15, "bold": True, "color": C_PRIMARY}]])

    # 128-bit 位段可视化
    seg_left = Emu(600000)
    seg_top  = Emu(2050000)
    seg_h    = Emu(500000)
    total_bits = 128
    total_w = Emu(11000000)
    segs = [
        (0, 32, "Coarse · Routing (/32-/48)", C_PRIMARY),
        (32, 64, "Medium · Subnet (/48-/64)", C_ACCENT),
        (64, 128, "Fine · Interface ID (/64-/128)", C_GREEN),
    ]
    for a, b, name, c in segs:
        x = seg_left + Emu(int((total_w / total_bits) * a))
        w = Emu(int((total_w / total_bits) * (b - a)))
        add_rect(slide, x, seg_top, w, seg_h, fill=c, radius=True)
        add_textbox(slide, x, seg_top, w, seg_h,
                    [[{"text": f"{name}\nbits [{a},{b})",
                       "size": 12, "bold": True, "color": C_WHITE,
                       "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)

    # 中部：训练 vs 推理 双栏
    y0 = Emu(2900000)
    # 训练卡
    add_rect(slide, Emu(400000), y0, Emu(5500000), Emu(3100000),
             fill=C_WHITE, line=C_ACCENT, radius=True)
    add_rect(slide, Emu(400000), y0, Emu(5500000), Emu(450000), fill=C_ACCENT, radius=True)
    add_textbox(slide, Emu(400000), y0, Emu(5500000), Emu(450000),
                [[{"text": "训练 · Layer-wise MLM + Entropy Weighting",
                   "size": 15, "bold": True, "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)
    train_lines = [
        "1. 每步随机采样一个 stage (Coarse/Med/Fine)",
        "2. 将该段 bit 替换为 [MASK]，模型重建",
        "3. 仅前缀长度内的位置计 loss",
        "",
        "▶ 熵加权：wᵢ = Hᵢ / H̄",
        "  · 前缀 (BGP 已锁定) 位 → 权重小",
        "  · 变化剧烈的位 → 权重大",
        "  · 让容量集中到真正信息量大的 bit",
    ]
    y = y0 + Emu(500000)
    for t in train_lines:
        add_textbox(slide, Emu(500000), y, Emu(5300000), Emu(280000),
                    [[{"text": t, "size": 13, "color": C_DARK,
                       "bold": t.startswith("▶")}]])
        y += Emu(300000)

    # 推理卡
    add_rect(slide, Emu(6200000), y0, Emu(5600000), Emu(3100000),
             fill=C_WHITE, line=C_GREEN, radius=True)
    add_rect(slide, Emu(6200000), y0, Emu(5600000), Emu(450000), fill=C_GREEN, radius=True)
    add_textbox(slide, Emu(6200000), y0, Emu(5600000), Emu(450000),
                [[{"text": "推理 · Three-Stage Masked Filling",
                   "size": 15, "bold": True, "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)
    infer_lines = [
        "0. b ← [MASK]¹²⁸",
        "1. BGP lock：top-ℓ_bgp 位固定为路由前缀",
        "2. Coarse fill：一次前向 → 采样 [0,32)",
        "3. Medium fill：一次前向 → 采样 [32,64)",
        "4. Fine fill：一次前向 → 采样 [64,128)",
        "5. 按目标长度裁剪",
        "",
        "▶ 每阶段内所有 bit 并行填充 (非自回归)",
    ]
    y = y0 + Emu(500000)
    for t in infer_lines:
        add_textbox(slide, Emu(6300000), y, Emu(5400000), Emu(280000),
                    [[{"text": t, "size": 13, "color": C_DARK,
                       "bold": t.startswith("▶")}]])
        y += Emu(300000)

    page_footer_conclusion(slide,
        "3 次前向 replaces 128 步自回归 —— 推理速度提升的核心来源")


# ---------- P11 闭环在线扫描 ----------
def slide_closedloop(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "闭环在线扫描 · SeedFilter + 在线微调 + 温度自适应",
                section="04 · Closed-Loop", page_no=11, total=TOTAL)

    # 顶部环形/流程
    add_rect(slide, Emu(400000), Emu(1550000), Emu(11400000), Emu(1150000),
             fill=C_LIGHT, line=C_PRIMARY, radius=True)
    add_textbox(slide, Emu(500000), Emu(1600000), Emu(11200000), Emu(400000),
                [[{"text": "▶ 一轮闭环：Generate → Filter → Probe → Refine → Fine-tune",
                   "size": 16, "bold": True, "color": C_PRIMARY}]])
    steps = [
        ("Generate", C_PRIMARY),
        ("SeedFilter\n(L1+L2)", C_ACCENT2),
        ("APD Probe", C_ACCENT),
        ("Granularity\nRefine", C_PURPLE),
        ("Online\nFine-tune", C_GREEN),
    ]
    bw = Emu(2000000)
    y = Emu(2050000)
    gap = Emu(150000)
    x0 = Emu(500000)
    for i, (name, c) in enumerate(steps):
        x = x0 + i * (bw + gap)
        add_rect(slide, x, y, bw, Emu(580000), fill=c, radius=True)
        add_textbox(slide, x, y, bw, Emu(580000),
                    [[{"text": name, "size": 13, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        # 箭头
        if i < len(steps) - 1:
            ax = x + bw + Emu(10000)
            add_textbox(slide, ax, y, gap - Emu(20000), Emu(580000),
                        [[{"text": "▶", "size": 20, "bold": True,
                           "color": C_GRAY, "align": PP_ALIGN.CENTER}]],
                        anchor=MSO_ANCHOR.MIDDLE)

    # 三张详情卡
    cards = [
        {
            "title": "双向 SeedFilter", "color": C_ACCENT2,
            "detail": ("L1  精确重复剔除\n"
                       "     candidate == known seed → drop\n\n"
                       "L2  内含区间静默丢弃\n"
                       "     candidate ⊂ known seed → silent drop\n"
                       "     不产生负样本 (仍可能是真 FRP)\n"
                       "     但记录『卡在已知区』比例 r_L2\n\n"
                       "▶ 覆盖 known seed 的更宽前缀保留探测"),
        },
        {
            "title": "在线微调", "color": C_GREEN,
            "detail": ("θ ← θ − η · ∇[ L_MLM(F⁺) − λ·L_MLM(F⁻) ]\n\n"
                       "• λ = 0.1，讨厌但不禁止负样本\n"
                       "• 学习率 = 10% × 预训练\n"
                       "• 梯度裁剪 ||∇||₂ ≤ 1\n"
                       "• 少量 step，避免飘离预训练最优点\n\n"
                       "▶ APD 反馈成为在线训练信号"),
        },
        {
            "title": "自适应温度", "color": C_ACCENT,
            "detail": ("r_L2 > 0.3 或 r_FRP = 0\n"
                       "  → T ← min(2.0, 1.1·T)  (探索升温)\n\n"
                       "r_FRP > 0.5\n"
                       "  → T ← max(0.3, 0.95·T)  (利用降温)\n\n"
                       "▶ 自动在『卡住 vs 打通』之间切换\n"
                       "  实验：任意 T₀∈[0.4,1.0] 收敛到 ~0.55"),
        },
    ]

    left0 = Emu(400000)
    w = Emu(3700000)
    gap2 = Emu(280000)
    top = Emu(2900000)
    h = Emu(3400000)
    for i, c in enumerate(cards):
        x = left0 + i * (w + gap2)
        add_rect(slide, x, top, w, h, fill=C_WHITE, line=c["color"], radius=True)
        add_rect(slide, x, top, w, Emu(450000), fill=c["color"], radius=True)
        add_textbox(slide, x, top, w, Emu(450000),
                    [[{"text": c["title"], "size": 16, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        add_textbox(slide, x + Emu(150000), top + Emu(500000),
                    w - Emu(300000), h - Emu(600000),
                    [[{"text": c["detail"], "size": 12, "color": C_DARK}]])

    page_footer_conclusion(slide,
        "把静态生成变成一个『生成-探测-学习』的强化学习闭环")


# ---------- P12 实验设置 & 端到端发现表 ----------
def slide_eval_e2e(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "实验 · 端到端 FRP 发现效果",
                section="05 · Evaluation", page_no=12, total=TOTAL)

    # 顶部：Setup
    add_rect(slide, Emu(400000), Emu(1550000), Emu(11400000), Emu(900000),
             fill=C_LIGHT, line=C_PRIMARY, radius=True)
    add_textbox(slide, Emu(500000), Emu(1600000), Emu(11200000), Emu(400000),
                [[{"text": "▶ 实验设置",
                   "size": 16, "bold": True, "color": C_PRIMARY}]])
    setup_line = ("数据：214,102 ICMPv6 FRP · 299 AS · 273 ORG · 51 国家 · 15 CAT / 24 SUB    ｜    "
                  "BGP 表：2026-04-29 快照 · 227,894 前缀    ｜    "
                  "基线：MAPD / Luori / Gungnir (SOTA)")
    add_textbox(slide, Emu(500000), Emu(2000000), Emu(11200000), Emu(400000),
                [[{"text": setup_line, "size": 12, "color": C_DARK}]])

    # 表：端到端发现
    add_textbox(slide, Emu(400000), Emu(2600000), Emu(11400000), Emu(400000),
                [[{"text": "▶ Table 1 · ICMPv6 end-to-end discovery (B_prefix ∈ {10, 30, 50})",
                   "size": 15, "bold": True, "color": C_PRIMARY}]])

    # 手动画表
    table_top = Emu(3050000)
    row_h = Emu(400000)
    col_widths = [Emu(2400000), Emu(1700000), Emu(1700000), Emu(1700000),
                  Emu(1700000), Emu(1900000)]
    headers = ["Metric", "MAPD", "Luori", "Gungnir", "Budget", "Heimdall (Ours)"]
    rows = [
        ["Cov_Routing @ B=10", "660", "409", "1158", "10", "1452"],
        ["Cov_Routing @ B=30", "809", "1202", "1733", "30", "2284"],
        ["Cov_Routing @ B=50", "909", "1765", "1903", "50", "2516"],
        ["Cov_AS      @ B=50", "143", "141",  "169", "50", "201"],
    ]
    # 表头
    x = Emu(400000)
    for cw, htxt in zip(col_widths, headers):
        add_rect(slide, x, table_top, cw, row_h, fill=C_PRIMARY, line=C_WHITE)
        add_textbox(slide, x, table_top, cw, row_h,
                    [[{"text": htxt, "size": 13, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        x += cw
    # 数据行
    for ri, row in enumerate(rows):
        y = table_top + row_h * (ri + 1)
        x = Emu(400000)
        for ci, (cw, val) in enumerate(zip(col_widths, row)):
            fill_c = C_WHITE if ri % 2 == 0 else C_LIGHT
            add_rect(slide, x, y, cw, row_h, fill=fill_c, line=C_GRAY)
            highlight = (ci == 5)  # 最后一列 Heimdall
            add_textbox(slide, x, y, cw, row_h,
                        [[{"text": val, "size": 13,
                           "bold": highlight,
                           "color": C_ACCENT if highlight else C_DARK,
                           "align": PP_ALIGN.CENTER}]],
                        anchor=MSO_ANCHOR.MIDDLE)
            x += cw

    # 结论条
    add_rect(slide, Emu(400000), Emu(4900000), Emu(11400000), Emu(1400000),
             fill=C_LIGHT, line=C_GREEN, radius=True)
    add_textbox(slide, Emu(500000), Emu(4950000), Emu(11200000), Emu(400000),
                [[{"text": "▶ 关键结论",
                   "size": 15, "bold": True, "color": C_GREEN}]])
    conclusions = [
        "• B_prefix = 50 时：Cov_Routing 比 Gungnir +32%，比 MAPD 快 2.8×",
        "• Cov_AS 比 Gungnir +19% (169 → 201)；Cov_Space 是 Gungnir 的 1.4×",
        "• TCP/80 场景仍领先：Cov_Routing 1238 vs Gungnir 981 vs Luori 672",
    ]
    y = Emu(5300000)
    for c in conclusions:
        add_textbox(slide, Emu(600000), y, Emu(11100000), Emu(320000),
                    [[{"text": c, "size": 13, "color": C_DARK}]])
        y += Emu(320000)

    page_footer_conclusion(slide,
        "在同等探测预算下，Heimdall 全面 dominate 三个基线")


# ---------- P13 对比曲线（大图）----------
def slide_curves(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "对比曲线 · 6 维度指标下的 Heimdall vs Luori vs Gungnir",
                section="05 · Evaluation", page_no=13, total=TOTAL)

    add_picture(slide,
                os.path.join(FIG_DIR, "comparison_curves.png"),
                left=Emu(400000), top=Emu(1500000),
                width=Emu(11400000),
                caption="Fig · 六指标 × 五探测预算下 (2M-10M) 的对比曲线；红色实线 = Heimdall")

    page_footer_conclusion(slide,
        "六个维度上 Heimdall 全线领先，且预算越大领先幅度越明显")


# ---------- P14 增量净新 & 覆盖分布 ----------
def slide_coverage(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "增量净新 & AS/Category/Country 覆盖",
                section="05 · Evaluation", page_no=14, total=TOTAL)

    # 2×2 网格
    grid = [
        ("netnew_curve.png",       "增量净新 FRP 曲线 (剔除已知种子)"),
        ("as_coverage_netnew.png", "AS 覆盖 · 净新增"),
        ("prefix_coverage_netnew.png", "BGP 前缀覆盖 · 净新增"),
        ("category_coverage_netnew.png", "AS 业务类别覆盖 · 净新增"),
    ]
    positions = [
        (Emu(400000),  Emu(1500000)),
        (Emu(6300000), Emu(1500000)),
        (Emu(400000),  Emu(4050000)),
        (Emu(6300000), Emu(4050000)),
    ]
    for (fn, cap), (x, y) in zip(grid, positions):
        add_picture(slide,
                    os.path.join(FIG_DIR, fn),
                    left=x, top=y,
                    width=Emu(5500000),
                    caption=cap, cap_size=10)

    page_footer_conclusion(slide,
        "『净新』视角下更能体现 Heimdall 相对基线的真实增量能力")


# ---------- P15 运行时间对比 ----------
def slide_time(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "运行时间 · Heimdall vs Gungnir vs Luori",
                section="05 · Evaluation", page_no=15, total=TOTAL)

    # 表
    add_textbox(slide, Emu(400000), Emu(1550000), Emu(11400000), Emu(400000),
                [[{"text": "▶ Table 2 · 端到端 wall-clock (hours) on N protocol/port targets",
                   "size": 15, "bold": True, "color": C_PRIMARY}]])

    table_top = Emu(2050000)
    row_h = Emu(430000)
    col_widths = [Emu(3200000), Emu(2500000), Emu(2500000), Emu(3200000)]
    headers = ["Phase / N", "Luori", "Gungnir", "Heimdall (Ours)"]
    rows = [
        ["Pre-training (h)",    "≈ 4.9·N", "6.8",  "4.0"],
        ["Inference     (h)",   "≈ 85·N",  "≈ 2.5·N", "1.6·N"],
        ["Total  (N=1)",        "89.9",   "9.3",  "5.6"],
        ["Total  (N=3)",        "269.7",  "14.3", "8.8"],
        ["Total  (N=5)",        "449.5",  "19.3", "12.0"],
    ]

    x = Emu(400000)
    for cw, htxt in zip(col_widths, headers):
        add_rect(slide, x, table_top, cw, row_h, fill=C_PRIMARY, line=C_WHITE)
        add_textbox(slide, x, table_top, cw, row_h,
                    [[{"text": htxt, "size": 14, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        x += cw

    for ri, row in enumerate(rows):
        y = table_top + row_h * (ri + 1)
        x = Emu(400000)
        for ci, (cw, val) in enumerate(zip(col_widths, row)):
            fill_c = C_WHITE if ri % 2 == 0 else C_LIGHT
            add_rect(slide, x, y, cw, row_h, fill=fill_c, line=C_GRAY)
            highlight = (ci == 3)
            add_textbox(slide, x, y, cw, row_h,
                        [[{"text": val, "size": 14,
                           "bold": highlight,
                           "color": C_ACCENT if highlight else C_DARK,
                           "align": PP_ALIGN.CENTER}]],
                        anchor=MSO_ANCHOR.MIDDLE)
            x += cw

    # 收益分析
    add_rect(slide, Emu(400000), Emu(4550000), Emu(11400000), Emu(1750000),
             fill=C_LIGHT, line=C_GREEN, radius=True)
    add_textbox(slide, Emu(500000), Emu(4600000), Emu(11200000), Emu(400000),
                [[{"text": "▶ 加速来源",
                   "size": 15, "bold": True, "color": C_GREEN}]])
    reasons = [
        "① 预训练 1.7× 加速 —— Mamba 选择性扫描 O(L) vs Transformer O(L²)",
        "② 每协议推理 1.5× 加速 —— 3 次并行 masked fill vs 128 步自回归",
        "③ N=5 协议时相比 Luori 37× 加速；相比 Gungnir 1.6× 加速",
    ]
    y = Emu(4950000)
    for r in reasons:
        add_textbox(slide, Emu(600000), y, Emu(11100000), Emu(320000),
                    [[{"text": r, "size": 14, "color": C_DARK}]])
        y += Emu(360000)

    page_footer_conclusion(slide,
        "更强的效果 + 更快的速度 = Pareto dominance")


# ---------- P16 闭环增益 & 消融 ----------
def slide_ablation(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "闭环增益 & 关键组件消融",
                section="05 · Evaluation", page_no=16, total=TOTAL)

    # 左：闭环 10 轮
    add_rect(slide, Emu(400000), Emu(1550000), Emu(5500000), Emu(4600000),
             fill=C_WHITE, line=C_PRIMARY, radius=True)
    add_rect(slide, Emu(400000), Emu(1550000), Emu(5500000), Emu(450000), fill=C_PRIMARY, radius=True)
    add_textbox(slide, Emu(400000), Emu(1550000), Emu(5500000), Emu(450000),
                [[{"text": "Table 3 · 10-round 闭环发现率 ρ / 累计 FRP",
                   "size": 15, "bold": True, "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)

    # 小表
    tbl_top = Emu(2100000)
    row_h = Emu(430000)
    cw = [Emu(2000000), Emu(1150000), Emu(1150000), Emu(1200000)]
    hdr = ["Round", "1", "5", "10"]
    x = Emu(500000)
    for c, h in zip(cw, hdr):
        add_rect(slide, x, tbl_top, c, row_h, fill=C_PRIMARY, line=C_WHITE)
        add_textbox(slide, x, tbl_top, c, row_h,
                    [[{"text": h, "size": 13, "bold": True, "color": C_WHITE,
                       "align": PP_ALIGN.CENTER}]], anchor=MSO_ANCHOR.MIDDLE)
        x += c
    rows = [
        ["w/o online ft", "0.34 / 17", "0.27 / 71",  "0.24 / 119"],
        ["Heimdall full", "0.34 / 17", "0.48 / 91",  "0.55 / 175"],
    ]
    for ri, r in enumerate(rows):
        y = tbl_top + row_h * (ri + 1)
        x = Emu(500000)
        for ci, (c, v) in enumerate(zip(cw, r)):
            fill_c = C_LIGHT if ri == 0 else C_WHITE
            add_rect(slide, x, y, c, row_h, fill=fill_c, line=C_GRAY)
            hi = (ri == 1)
            add_textbox(slide, x, y, c, row_h,
                        [[{"text": v, "size": 12,
                           "bold": hi,
                           "color": C_GREEN if hi else C_DARK,
                           "align": PP_ALIGN.CENTER}]], anchor=MSO_ANCHOR.MIDDLE)
            x += c

    # 结论
    add_textbox(slide, Emu(500000), Emu(3900000), Emu(5300000), Emu(400000),
                [[{"text": "▶ 关键：无在线微调 → 发现率单调衰减",
                   "size": 13, "bold": True, "color": C_RED}]])
    add_textbox(slide, Emu(500000), Emu(4250000), Emu(5300000), Emu(400000),
                [[{"text": "     Heimdall full → 发现率反而上升",
                   "size": 13, "bold": True, "color": C_GREEN}]])
    add_textbox(slide, Emu(500000), Emu(4600000), Emu(5300000), Emu(400000),
                [[{"text": "     10 轮后累计 FRP 增加 1.47×",
                   "size": 13, "bold": True, "color": C_ACCENT}]])
    add_textbox(slide, Emu(500000), Emu(5100000), Emu(5300000), Emu(700000),
                [[{"text": "▶ APD 反馈 = 免费的、生长中的训练信号；\n     模型跟随实际网络演化，而不是被种子锁死",
                   "size": 12, "italic": True, "color": C_GRAY}]])

    # 右：消融
    add_rect(slide, Emu(6200000), Emu(1550000), Emu(5600000), Emu(4600000),
             fill=C_WHITE, line=C_ACCENT, radius=True)
    add_rect(slide, Emu(6200000), Emu(1550000), Emu(5600000), Emu(450000), fill=C_ACCENT, radius=True)
    add_textbox(slide, Emu(6200000), Emu(1550000), Emu(5600000), Emu(450000),
                [[{"text": "Table 4 · Held-out bit-accuracy 消融",
                   "size": 15, "bold": True, "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)

    tbl2_top = Emu(2100000)
    cw2 = [Emu(2900000), Emu(900000), Emu(900000), Emu(900000)]
    hdr2 = ["Variant", "Coarse", "Medium", "Fine"]
    x = Emu(6300000)
    for c, h in zip(cw2, hdr2):
        add_rect(slide, x, tbl2_top, c, row_h, fill=C_ACCENT, line=C_WHITE)
        add_textbox(slide, x, tbl2_top, c, row_h,
                    [[{"text": h, "size": 12, "bold": True, "color": C_WHITE,
                       "align": PP_ALIGN.CENTER}]], anchor=MSO_ANCHOR.MIDDLE)
        x += c

    rows2 = [
        ["Outer-only Mamba",           "0.881", "0.882", "0.911"],
        ["Inner-only Mamba",           "0.864", "0.873", "0.901"],
        ["DS-HMamba w/o gate (mean)",  "0.901", "0.905", "0.928"],
        ["w/o entropy weighting",      "0.911", "0.913", "0.930"],
        ["Heimdall (full)",            "0.927", "0.929", "0.943"],
    ]
    for ri, r in enumerate(rows2):
        y = tbl2_top + row_h * (ri + 1)
        x = Emu(6300000)
        for ci, (c, v) in enumerate(zip(cw2, r)):
            fill_c = C_LIGHT if ri % 2 == 0 else C_WHITE
            is_full = (ri == 4)
            if is_full:
                fill_c = RGBColor(0xFF, 0xF3, 0xE0)
            add_rect(slide, x, y, c, row_h, fill=fill_c, line=C_GRAY)
            add_textbox(slide, x, y, c, row_h,
                        [[{"text": v, "size": 11,
                           "bold": is_full,
                           "color": C_ACCENT if is_full else C_DARK,
                           "align": PP_ALIGN.CENTER}]], anchor=MSO_ANCHOR.MIDDLE)
            x += c

    add_textbox(slide, Emu(6300000), Emu(4900000), Emu(5400000), Emu(400000),
                [[{"text": "▶ Inner scan +2.0，Gate +2.6，Entropy +1.3",
                   "size": 12, "bold": True, "color": C_ACCENT}]])
    add_textbox(slide, Emu(6300000), Emu(5250000), Emu(5400000), Emu(800000),
                [[{"text": "  → 每个组件都可分离地带来独立增益，\n     全套设计缺一不可",
                   "size": 12, "italic": True, "color": C_GRAY}]])

    page_footer_conclusion(slide,
        "闭环反馈 & 每个模块都贡献了独立增益 —— 三大设计不是过度设计")


# ---------- P17 长期观测发现（3 个月多协议）----------
def slide_longterm(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "长期观测 · 3 个月 · 23 协议多轮扫描",
                section="06 · Long-Term Study", page_no=17, total=TOTAL)

    # 顶部指标
    add_rect(slide, Emu(400000), Emu(1550000), Emu(11400000), Emu(900000),
             fill=C_LIGHT, line=C_PRIMARY, radius=True)
    add_textbox(slide, Emu(500000), Emu(1600000), Emu(11200000), Emu(400000),
                [[{"text": "▶ 相比 IPv6 hitlist / Luori FRP list 的增量",
                   "size": 15, "bold": True, "color": C_PRIMARY}]])
    add_textbox(slide, Emu(500000), Emu(2000000), Emu(11200000), Emu(400000),
                [[{"text": "Cov_AS：335 / 530 → 684    ｜    Cov_Routing：3,237 / 7,324 → 9,512",
                   "size": 15, "bold": True, "color": C_ACCENT}]])

    # 三个发现
    findings = [
        {
            "title": "① 集中度 · Concentration", "color": C_PRIMARY,
            "detail": ("三家组织 (Cloudflare, Amazon, Akamai)\n"
                       "贡献了 > 2M 条 ICMPv6 FRP\n"
                       "以 /64 – /56 的细粒度为主\n"
                       "→ 与 ip_nonlocal_bind 大规模绑定行为一致"),
        },
        {
            "title": "② 服务 vs 端口错配", "color": C_ACCENT,
            "detail": ("在 1.6M 条 HTTP-FRP 地址中\n"
                       "仅 15% 实际返回 HTTP 应用内容\n"
                       "其余在 TCP 层响应但无应用\n"
                       "→ 疑似为规避 per-IP 限流的出向代理池\n"
                       "    副作用：绕过 single-IP DDoS 防御"),
        },
        {
            "title": "③ 安全信号 · Security Tells", "color": C_RED,
            "detail": ("SSH-FRP：~11% OpenSSH 7.2p2 保留\n"
                       "  diffie-hellman-group14-sha1 → 蜜罐嫌疑\n\n"
                       "SSL-FRP：~23% 共用一张 Cloudflare 泛证书\n"
                       "  → 一旦私钥泄漏，同批 FRP 集体暴露"),
        },
    ]

    left0 = Emu(400000)
    w = Emu(3700000)
    gap = Emu(280000)
    top = Emu(2650000)
    h = Emu(3600000)
    for i, f in enumerate(findings):
        x = left0 + i * (w + gap)
        add_rect(slide, x, top, w, h, fill=C_WHITE, line=f["color"], radius=True)
        add_rect(slide, x, top, w, Emu(500000), fill=f["color"], radius=True)
        add_textbox(slide, x, top, w, Emu(500000),
                    [[{"text": f["title"], "size": 15, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        add_textbox(slide, x + Emu(150000), top + Emu(550000),
                    w - Emu(300000), h - Emu(650000),
                    [[{"text": f["detail"], "size": 13, "color": C_DARK}]])

    page_footer_conclusion(slide,
        "FRP 不只是测量噪声，更是 IPv6 生态中 CDN / 蜜罐 / 代理 的观察窗口")


# ---------- P18 Ethics & 总结 ----------
def slide_conclusion(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "Ethics & 总结",
                section="Conclusion", page_no=18, total=TOTAL)

    # Ethics
    add_rect(slide, Emu(400000), Emu(1550000), Emu(5500000), Emu(4600000),
             fill=C_WHITE, line=C_ACCENT2, radius=True)
    add_rect(slide, Emu(400000), Emu(1550000), Emu(5500000), Emu(450000), fill=C_ACCENT2, radius=True)
    add_textbox(slide, Emu(400000), Emu(1550000), Emu(5500000), Emu(450000),
                [[{"text": "Ethical Considerations",
                   "size": 16, "bold": True, "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)
    ethics = [
        "• 遵循 Menlo Report 与社区 ethical measurement 指南",
        "• 单一 vantage point，速率 ≤ 1 Mbps",
        "• 随机化目标顺序 · 单批 ≤ 10 分钟",
        "• 维护 opt-out list，24 小时内响应排除请求",
        "• 从不做渗透测试，仅从 banner / version 推断",
        "  绝不对活主机进行漏洞验证",
    ]
    y = Emu(2150000)
    for e in ethics:
        add_textbox(slide, Emu(550000), y, Emu(5200000), Emu(350000),
                    [[{"text": e, "size": 13, "color": C_DARK}]])
        y += Emu(380000)

    # 总结
    add_rect(slide, Emu(6200000), Emu(1550000), Emu(5600000), Emu(4600000),
             fill=C_WHITE, line=C_PRIMARY, radius=True)
    add_rect(slide, Emu(6200000), Emu(1550000), Emu(5600000), Emu(450000), fill=C_PRIMARY, radius=True)
    add_textbox(slide, Emu(6200000), Emu(1550000), Emu(5600000), Emu(450000),
                [[{"text": "总结 · Heimdall Take-away",
                   "size": 16, "bold": True, "color": C_WHITE, "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)

    summary = [
        ("方法", "首个把 SSM (Mamba) 应用于 IPv6 生成的工作", C_PRIMARY),
        ("结构", "Dual-Scan：位轴 + 同 AS 样本轴，AS-gate 融合", C_ACCENT),
        ("训练", "层次化 MLM + 熵加权 + 三阶段并行推理", C_ACCENT2),
        ("闭环", "双向 SeedFilter + 在线微调 + 温度自适应", C_GREEN),
        ("结果", "1.9× 吞吐 · 1.2× AS 覆盖 · 1.7× 训练加速", C_RED),
        ("开源", "代码、curated FRP list 计划公开", C_PURPLE),
    ]
    y = Emu(2100000)
    for tag, txt, c in summary:
        add_rect(slide, Emu(6300000), y, Emu(700000), Emu(400000), fill=c, radius=True)
        add_textbox(slide, Emu(6300000), y, Emu(700000), Emu(400000),
                    [[{"text": tag, "size": 13, "bold": True,
                       "color": C_WHITE, "align": PP_ALIGN.CENTER}]], anchor=MSO_ANCHOR.MIDDLE)
        add_textbox(slide, Emu(7100000), y, Emu(4600000), Emu(400000),
                    [[{"text": txt, "size": 13, "color": C_DARK}]],
                    anchor=MSO_ANCHOR.MIDDLE)
        y += Emu(500000)

    page_footer_conclusion(slide,
        "Dual-Scan × Closed-Loop × Linear-time SSM = 下一代 IPv6 主动测量的技术底座")


# ---------- P19 新学期工作展望 ----------
def slide_future_work(prs):
    slide = add_slide_blank(prs)
    page_header(slide, "新学期工作展望 · 2026 秋 → 2027 春",
                section="07 · Future Work", page_no=19, total=TOTAL)

    # 三条主线
    lanes = [
        {
            "tag": "①", "color": C_PRIMARY,
            "title": "Heimdall 投稿与开源",
            "items": [
                "• 完成 IEEE ICNP 投稿并跟进 Rebuttal",
                "• 迁移适配到 SIGCOMM / NSDI 更长版本",
                "• 开源代码 + curated FRP list 发布",
                "• 与业界数据集 (IPv6 Hitlist) 建立打通",
            ],
        },
        {
            "tag": "②", "color": C_ACCENT,
            "title": "闭环 & SSM 方法延伸",
            "items": [
                "• Dual-Scan 推广到 IPv6 命中率预测任务",
                "• 引入 Vision Mamba 结构处理时序种子演化",
                "• 用 RL policy gradient 替代当前手工温度控制",
                "• 探索多头/多协议同时训练的联合模型",
            ],
        },
        {
            "tag": "③", "color": C_GREEN,
            "title": "系统 & 生产落地",
            "items": [
                "• 与 TGW / 云 IP 复用测量团队方向联动",
                "• 探索面向蜜罐/代理池的 IPv6 攻击面研究",
                "• 生产集群持续多协议扫描 → 半年一次 Report",
                "• 组内例会：每 2 周同步一次进度",
            ],
        },
    ]

    left = Emu(400000)
    w = Emu(3700000)
    gap = Emu(280000)
    top = Emu(1600000)
    h = Emu(4200000)
    for i, l in enumerate(lanes):
        x = left + i * (w + gap)
        add_rect(slide, x, top, w, h, fill=C_WHITE, line=l["color"], radius=True)
        # header
        add_rect(slide, x, top, w, Emu(700000), fill=l["color"], radius=True)
        add_textbox(slide, x + Emu(150000), top + Emu(100000), Emu(500000), Emu(500000),
                    [[{"text": l["tag"], "size": 28, "bold": True,
                       "color": C_WHITE}]], anchor=MSO_ANCHOR.MIDDLE)
        add_textbox(slide, x + Emu(700000), top + Emu(100000),
                    w - Emu(800000), Emu(500000),
                    [[{"text": l["title"], "size": 15, "bold": True,
                       "color": C_WHITE}]], anchor=MSO_ANCHOR.MIDDLE)
        # items
        y = top + Emu(800000)
        for it in l["items"]:
            add_textbox(slide, x + Emu(150000), y, w - Emu(300000), Emu(700000),
                        [[{"text": it, "size": 13, "color": C_DARK}]])
            y += Emu(750000)

    # 底部时间轴 Roadmap
    add_rect(slide, Emu(400000), Emu(5950000), Emu(11400000), Emu(400000),
             fill=C_LIGHT, line=C_PRIMARY, radius=True)
    add_textbox(slide, Emu(500000), Emu(5950000), Emu(11200000), Emu(400000),
                [[{"text": "▶ Roadmap：Q4·2026 完成 ICNP 投稿 → Q1·2027 系统开源 & 联合团队评审 → Q2·2027 冲刺 SIGCOMM/NSDI 扩展版",
                   "size": 13, "bold": True, "color": C_PRIMARY,
                   "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)

    page_footer_conclusion(slide,
        "学术 × 工程 × 组内协作，三条主线并进")


# ---------- P20 Thanks ----------
def slide_thanks(prs):
    slide = add_slide_blank(prs)
    # 满页背景
    add_rect(slide, Emu(0), Emu(0), SLIDE_W, SLIDE_H, fill=C_PRIMARY)

    add_textbox(slide, Emu(0), Emu(1800000), SLIDE_W, Emu(1500000),
                [[{"text": "Thank You !",
                   "size": 96, "bold": True, "color": C_WHITE, "font": FONT_TITLE,
                   "align": PP_ALIGN.CENTER}]],
                anchor=MSO_ANCHOR.MIDDLE)
    add_textbox(slide, Emu(0), Emu(3400000), SLIDE_W, Emu(600000),
                [[{"text": "Heimdall · Dual-Scan Closed-Loop FRP Discovery",
                   "size": 24, "color": RGBColor(0xF9, 0xD1, 0x8B),
                   "align": PP_ALIGN.CENTER, "italic": True}]])
    add_textbox(slide, Emu(0), Emu(4100000), SLIDE_W, Emu(500000),
                [[{"text": "Q & A · 欢迎讨论",
                   "size": 24, "color": C_WHITE,
                   "align": PP_ALIGN.CENTER, "bold": True}]])
    add_textbox(slide, Emu(0), Emu(5900000), SLIDE_W, Emu(400000),
                [[{"text": "分享人：谷雨宸    ·    组会分享    ·    2026 / 09",
                   "size": 14, "color": RGBColor(0xCC, 0xDA, 0xF0),
                   "align": PP_ALIGN.CENTER}]])


# =============================================================
def main():
    prs = Presentation(TEMPLATE)
    remove_all_slides(prs)

    slide_title(prs)          # 1
    slide_toc(prs)            # 2
    slide_background(prs)     # 3
    slide_related(prs)        # 4
    slide_motivation(prs)     # 5
    slide_entropy(prs)        # 6
    slide_overview_ideas(prs) # 7
    slide_overview(prs)       # 8
    slide_dshmamba(prs)       # 9
    slide_training(prs)       # 10
    slide_closedloop(prs)     # 11
    slide_eval_e2e(prs)       # 12
    slide_curves(prs)         # 13
    slide_coverage(prs)       # 14
    slide_time(prs)           # 15
    slide_ablation(prs)       # 16
    slide_longterm(prs)       # 17
    slide_conclusion(prs)     # 18
    slide_future_work(prs)    # 19
    slide_thanks(prs)         # 20

    prs.save(OUTPUT)
    print(f"[OK] PPT 已生成 → {OUTPUT}")
    print(f"     共 {TOTAL} 页 · 底板：{TEMPLATE}")


if __name__ == "__main__":
    main()
