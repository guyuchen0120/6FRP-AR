#!/usr/bin/env python3
"""
Fill in /root/6FRP/探测项目SOP.docx with the real information
gathered from the 6FRP IPv6 FRP detection project.

Project: 6FRP — IPv6 Fully-Responsive Prefix (FRP) Discovery
Model  : DualScanHMamba (Dual-Scan Hierarchical Mamba, pure-PyTorch S6)
Pipeline: BGP seeds -> data_prep -> train -> online inference (generate ->
          APD ICMPv6 probe -> judge FRP -> fine-tune -> iterate)
"""
import copy
from docx import Document
from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

SOP_PATH = "/root/6FRP/探测项目SOP.docx"

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def set_cell_text(cell, text, bold=False, size_pt=12, font_name="仿宋"):
    """Replace a cell's content with multi-line text, uniform仿宋 font."""
    cell.text = ""  # clears but leaves 1 empty paragraph
    lines = text.split("\n")
    p = cell.paragraphs[0]
    for i, line in enumerate(lines):
        if i == 0:
            para = p
            # clear any leftover runs
            for r in list(para.runs):
                r.text = ""
        else:
            para = cell.add_paragraph()
        run = para.add_run(line)
        run.font.name = font_name
        # ensure CJK font is also set
        rPr = run._element.get_or_add_rPr()
        rFonts = rPr.find(qn('w:rFonts'))
        if rFonts is None:
            rFonts = OxmlElement('w:rFonts')
            rPr.append(rFonts)
        rFonts.set(qn('w:eastAsia'), font_name)
        rFonts.set(qn('w:ascii'), font_name)
        rFonts.set(qn('w:hAnsi'), font_name)
        run.font.size = Pt(size_pt)
        run.bold = bold


def _set_cjk_font(run, font_name="仿宋"):
    run.font.name = font_name
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.append(rFonts)
    rFonts.set(qn('w:eastAsia'), font_name)
    rFonts.set(qn('w:ascii'), font_name)
    rFonts.set(qn('w:hAnsi'), font_name)


def rewrite_paragraph(p, text, bold=False, size_pt=12,
                      font_name="仿宋", indent_pt=None):
    """Replace paragraph text while keeping its style (e.g. List Paragraph)."""
    for r in list(p.runs):
        r.text = ""
    # Remove the extra empty runs by setting the first run only
    if p.runs:
        p.runs[0].text = text
        p.runs[0].bold = bold
        p.runs[0].font.size = Pt(size_pt)
        _set_cjk_font(p.runs[0], font_name)
        # erase other runs
        for r in p.runs[1:]:
            r.text = ""
    else:
        run = p.add_run(text)
        run.bold = bold
        run.font.size = Pt(size_pt)
        _set_cjk_font(run, font_name)
    if indent_pt is not None:
        p.paragraph_format.left_indent = Pt(indent_pt)


def insert_paragraph_after(paragraph, text, bold=False, size_pt=12,
                           font_name="仿宋", style="Normal",
                           indent_pt=None):
    """Insert a new paragraph *after* the given paragraph. Returns the new paragraph."""
    new_p = OxmlElement("w:p")
    paragraph._p.addnext(new_p)
    from docx.text.paragraph import Paragraph
    new_para = Paragraph(new_p, paragraph._parent)
    try:
        new_para.style = paragraph.part.document.styles[style]
    except KeyError:
        pass
    run = new_para.add_run(text)
    run.bold = bold
    run.font.size = Pt(size_pt)
    _set_cjk_font(run, font_name)
    if indent_pt is not None:
        new_para.paragraph_format.left_indent = Pt(indent_pt)
    return new_para


# ---------------------------------------------------------------------------
# content
# ---------------------------------------------------------------------------

TITLE = "6FRP（IPv6 全响应前缀探测）项目 SOP"

# -------- Table 0: 基础环境要求 --------
TABLE_ENV = {
    (0, 1): "TencentOS Server 4.4（Linux 6.6.x，x86_64；兼容 CentOS 8 / Ubuntu 22.04 以上）",
    (1, 1): "16 GB（最低 8 GB；在线推理+APD 并发建议 ≥ 16 GB）",
    (2, 1): "100 GB（当前已用约 60 GB；数据/模型/日志预留 40 GB）",
    (3, 1): "0 GB（当前 CPU 单机运行；若做大批量训练，建议 ≥ 8 GB，如 RTX 3080/4090）",
    (4, 1): "8 核（最低 4 核；在线推理 APD 多并发建议 ≥ 8 核）",
}

# -------- Table 1: 项目源码信息 --------
PROJECT_ROOT = "/root/6FRP"

TABLE_SRC = {
    (0, 1): "本地路径：/root/6FRP（远端仓库未设置，建议后续推送到团队 Git；对应论文已在 paper/6FRP.docx、space_count/6FRP_infocom2027.tex）",
    (1, 1): "V1.0（dual_scan / online_infer 版本，对应 checkpoints/dual_scan/best_model.pt，训练于 2025-07-05）",
    (2, 1): "main（本地分支；暂未提交到远程）",
    (3, 1): (
        "语言：Python 3.11.6；\n"
        "深度学习：PyTorch 2.12.0（纯 CPU 可运行；selective_scan 用纯 PyTorch 实现，无需 mamba-ssm/causal-conv1d 等 CUDA 扩展）；\n"
        "数值/数据：numpy 2.4.6、ipaddress（标准库）、json/csv（标准库）；\n"
        "网络探测：原始 ICMPv6 Socket（utils/apd_core.py），需 root 或 CAP_NET_RAW；\n"
        "其他：requests（可选，远程 APD 场景）"
    ),
    (4, 1): (
        "generators/            —— 模型实现：\n"
        "  dual_scan_hmamba.py    DualScanHMamba 主模型（Outer+Inner Scan + Gating Fusion）\n"
        "  hmamba.py / unified.py 基线/对比模型\n"
        "  inner_only_mamba.py / outer_only_mamba.py  消融模型\n"
        "  compare_models.py / visualize_full_hmamba.py / demonstrate_hmamba_flow.py\n"
        "data_prep/prepare.py   —— 种子+BGP+AS 元数据预处理（最长前缀匹配 LPM，输出 meta.json 等）\n"
        "data/                   —— 原始种子（seeds.txt、ipasn_*.dat、as_org_category.csv、Gungnir_seeds_original.txt）\n"
        "data_fixed/             —— 预处理结果（seeds_prepared.txt、meta.json、as_seed_db.json、vocabs.json、stats.json、filtered_seeds.txt）\n"
        "train_dual_scan.py     —— 训练入口（MLM 分层掩码，三阶段 coarse/medium/fine）\n"
        "debug_train.py         —— 单 batch 调试训练脚本\n"
        "checkpoints/dual_scan/ —— 训练产物（best_model.pt、checkpoint_epochN.pt、config.json）\n"
        "online_infer_dual_scan.py —— 在线推理主入口（生成→过滤→APD 探测→FRP 判定→在线微调→循环）\n"
        "standalone_apd.py      —— 独立 APD 工具（纯 ICMPv6 FRP 探测，不依赖模型）\n"
        "utils/                  —— 公共模块：\n"
        "  apd.py / apd_core.py  ICMPv6 Echo 探测、FRP 判定、树对齐细化（Tree Alignment）\n"
        "  ipv6_utils.py         IPv6 前缀生成/归一化/子网判断\n"
        "  entropy_analysis.py   位熵分析，用于加权 MLM\n"
        "space_count/            —— 覆盖率/熵/绘图脚本与结果（count.py、plot_*.py、*.pdf/.png、frps/、6FRP/、Gungnir/、Luori/ 等对比数据）\n"
        "paper/                  —— 论文稿与说明（6FRP.docx、Heimdall.tex、9-14.md 等）\n"
        "results/                —— 在线推理运行结果（round_results.json、confirmed_frps.txt、known_frps.txt 等）\n"
        "test.py / dedup_ipv6_prefixes.py / find_middle_jsonl.py —— 辅助/验证脚本"
    ),
    (5, 1): (
        "关键配置：\n"
        "  checkpoints/dual_scan/config.json —— 模型超参（d_model=128，num_layers=4，vocab_size=3，seq_len=128，\n"
        "                                       coarse_end=32，medium_end=64，num_as=300，num_org=273，num_geo=51 等）\n"
        "  data_fixed/stats.json             —— 词表大小、种子统计（输入 214158，解析 214102，AS 词表 300）\n"
        "  data_fixed/vocabs.json            —— AS/org/country/cat1/cat2 名称→id 映射\n"
        "数据位置：\n"
        "  BGP 前缀表：data/ipasn_20260429.dat（5.9 MB）\n"
        "  AS 元数据：data/as_org_category.csv（9.3 MB）\n"
        "  原始种子：data/seeds.txt（5.07 MB，214k 条前缀）、data/Gungnir_seeds_original.txt\n"
        "  预处理种子：data_fixed/seeds_prepared.txt、data_fixed/filtered_seeds.txt\n"
        "  元数据：data_fixed/meta.json（45 MB，每种子 AS/org/geo/rtt 信息）\n"
        "  AS→种子集合：data_fixed/as_seed_db.json（6.5 MB）\n"
        "模型权重：\n"
        "  checkpoints/dual_scan/best_model.pt（8.3 MB，val-loss 最佳）\n"
        "  checkpoints/dual_scan/checkpoint_epoch{4,9,14,19}.pt\n"
        "启停脚本：\n"
        "  数据准备：python3 -m data_prep.prepare --bgp-table data/ipasn_20260429.dat \\\n"
        "              --as-meta data/as_org_category.csv --seeds data/seeds.txt \\\n"
        "              --out-dir data_fixed\n"
        "  训练：    python3 train_dual_scan.py --data-dir data_fixed \\\n"
        "              --output-dir checkpoints/dual_scan --epochs 30 --batch-size 64\n"
        "  在线推理：sudo python3 online_infer_dual_scan.py \\\n"
        "              --checkpoint checkpoints/dual_scan/best_model.pt \\\n"
        "              --data-dir data_fixed --output-dir results/online_run1\n"
        "数据库：无（结果以 JSON/TXT 文件形式持久化到 results/*/）"
    ),
    (6, 1): "未提供 Dockerfile / docker-compose.yml（当前为裸机 Python 环境运行；如需容器化，建议基于 python:3.11-slim 镜像，apt 安装 iputils-ping、libpcap，pip 安装 torch==2.12.0、numpy、requests，再 COPY /root/6FRP 代码，并以 --cap-add=NET_RAW 运行以支持 ICMPv6）",
    (7, 1): "无对外 HTTP/gRPC 接口（纯本地 Python CLI）；\n命令行参数见：python3 online_infer_dual_scan.py --help、python3 train_dual_scan.py --help；\n模型内部接口详见 generators/dual_scan_hmamba.py 中 DualScanHMamba.compute_mlm_loss / generate",
    (8, 1): (
        "paper/9-14.md       —— 项目方法总结、与 PaS / SubRecon / AddrProbe 等相关工作对比\n"
        "space_count/        —— 覆盖率/熵对比图（address_space*.pdf、comparison_curves*.pdf、entropy.pdf 等）\n"
        "常见坑：\n"
        "  1) 在线推理必须使用 root（原始 ICMPv6 socket），否则 APD 全部超时；\n"
        "  2) 机器需要可用的 IPv6 全球单播地址，否则 APD 无法发出 Echo；\n"
        "  3) PyTorch 2.12.0 的 selective_scan 是 CPU 纯实现，batch_size 过大会导致内存爆；\n"
        "  4) 训练数据中 AS id 压缩到 0-299（见 stats.json 的 max_as_id_compact=299），新 AS 需重跑 prepare.py；\n"
        "  5) 以 /48 或更短前缀为单位进行 APD，probe_count 默认 16，success_rate 阈值 0.875（即 14/16 成功才判 FRP）。"
    ),
    (9, 1): (
        "论文方向：IPv6 Fully-Responsive Prefix（FRP）发现，用于别名前缀识别/修正，\n"
        "         配套对比方法 Gungnir（ICNP 2025）、Luori（TON 2026）、AddrProbe（ToN 2025/2026）；\n"
        "数据集：214,158 条种子前缀，覆盖 299 个 AS、51 个国家、15/24 个 cat1/cat2 组织类别；\n"
        "模型规模：d_model=128，4 层 DualScan（Outer+Inner Scan + Gating Fusion），参数量约 1-2 M；\n"
        "最优 checkpoint：checkpoints/dual_scan/best_model.pt（2025-07-05 训练）；\n"
        "探测限额：默认在线推理单次运行上限 6.5 M 次 ICMPv6 探测包（--max-total-probes 可调），\n"
        "         每次运行会在 500k/2.5M/4.5M/6.5M 次探测时自动保存 FRP 快照。"
    ),
}

# -------- Table 2: 项目当前管理人员信息 --------
TABLE_OWNER = {
    (0, 1): "（待填写：当前负责人姓名）",
    (1, 1): "（待填写：学号 / 工号）",
    (2, 1): "（待填写：联系电话）",
    (3, 1): "（待填写：邮箱）",
    (4, 1): "（待填写：学校 / 实验室 / 部门）",
}

# -------- Table 3: 远程信息 --------
TABLE_REMOTE = {
    (0, 1): (
        "主机名：VM-42-229-tencentos\n"
        "IP 地址：21.214.42.229（内网 172.17.0.1）；需额外一个 IPv6 全球单播地址用于 APD 发包\n"
        "端口：22（SSH；项目本身无对外端口）\n"
        "用户名：root（在线推理需要 root，用于原始 ICMPv6 socket）\n"
        "操作系统：TencentOS Server 4.4\n"
        "项目路径：/root/6FRP\n"
        "公钥：（请在交接时追加当前负责人 ~/.ssh/id_ed25519.pub 内容）"
    ),
    (1, 1): (
        "专人负责的 SSH 公钥（追加到 ~/.ssh/authorized_keys 后即可免密登录）：\n"
        "（请在交接时粘贴新同学 id_ed25519.pub / id_rsa.pub 的完整内容，示例：\n"
        "  ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAA... user@host）\n"
        "注意：\n"
        "  1) 原有公钥（ssh-rsa-20240508235923）保留，用于回滚；\n"
        "  2) 禁止将私钥提交到任何仓库；\n"
        "  3) 远程服务器 /root/.ssh/ 权限需保持 700，authorized_keys 为 600。"
    ),
}

# -------- 安装步骤 (项目安装步骤段落 7 之后的 … 占位符 8-12) --------
INSTALL_STEPS = [
    "步骤 1：准备运行环境。安装 Python 3.11（TencentOS 4 自带 /usr/bin/python3.11，若未安装则：dnf install python3.11 python3.11-pip -y）；确认 pip3 可用：pip3 --version。",
    "步骤 2：克隆/拷贝源码。将 /root/6FRP 整个目录拷贝到目标机器（约 1.2 GB，含 data/、data_fixed/、checkpoints/、paper/、space_count/ 等）；或后续从 Git 仓库 git clone 下来后，再把大文件 best_model.pt、meta.json、seeds.txt 等按说明从数据服务器 scp 过来。",
    "步骤 3：安装 Python 依赖。执行：pip3 install --upgrade pip && pip3 install torch==2.12.0 numpy requests 。若需 GPU 加速训练，另装与 CUDA 匹配的 torch 版本；本项目纯 CPU 可运行推理，无需 mamba-ssm / causal-conv1d 等 CUDA 扩展。",
    "步骤 4：验证模型/数据可加载。执行：cd /root/6FRP && python3 -c \"import torch; ckpt=torch.load('checkpoints/dual_scan/best_model.pt', map_location='cpu'); print(list(ckpt.keys()), ckpt['config'])\"；再执行 python3 debug_train.py 做一次 1-batch 前向验证，确认模型/数据正常。",
    "步骤 5：配置 IPv6 与网络能力。确认 ip -6 addr 能看到至少一个 global scope IPv6 地址；赋予 Python 原始 socket 权限（二选一）：① 以 root 身份运行 online_infer_dual_scan.py；② 为 python3 可执行文件设置 capability：sudo setcap cap_net_raw+ep $(readlink -f $(which python3))。无 IPv6 环境可先用 --dry-run 模拟 APD。",
]

# -------- 运行步骤 (段落 13 之后的 … 占位符 14-18) --------
RUN_STEPS = [
    "步骤 1（可选，仅需重建数据集时执行）：数据预处理。cd /root/6FRP && python3 -m data_prep.prepare --bgp-table data/ipasn_20260429.dat --as-meta data/as_org_category.csv --seeds data/seeds.txt --out-dir data_fixed；生成 seeds_prepared.txt、meta.json、as_seed_db.json、vocabs.json、stats.json、filtered_seeds.txt。当前 data_fixed/ 已就绪，日常无需重跑。",
    "步骤 2（可选，仅需重新训练时执行）：模型训练。python3 train_dual_scan.py --data-dir data_fixed --output-dir checkpoints/dual_scan --epochs 30 --batch-size 64 --lr 3e-4 --num-workers 4；每轮打印 Train/Val Loss & bit_acc，三阶段 coarse/medium/fine 掩码统计。默认每 5 epoch 保存一次 checkpoint，val-loss 最优者写入 best_model.pt。",
    "步骤 3（主流程）：在线推理 + 主动探测。以 root 身份运行：sudo python3 online_infer_dual_scan.py --checkpoint checkpoints/dual_scan/best_model.pt --data-dir data_fixed --output-dir results/online_run1 --candidates-per-round 500 --temperature 0.8 --apd-concurrency 8 --apd-workers 4 --max-total-probes 6500000 --bootstrap-known-from-seeds。流程为：生成候选前缀 → 用 filtered_seeds 做子前缀过滤 → APD ICMPv6 探测（默认 16 个子前缀，success_rate≥0.875 判 FRP）→ 结果反馈到模型做在线微调 → 进入下一轮；达到 --max-total-probes 或触发 snapshot 硬停时结束。",
    "步骤 4：结果归档与分析。运行结果位于 results/online_run1/：confirmed_frps.txt（最终 FRP 列表）、round_results.json（每轮详情）、summary.json（汇总，含 discovery_rate）、known_frps_merged.txt（累计快照）、model_final.pt（终态模型）、live_frp.log（增量 FRP 日志）。可用 space_count/count.py、space_count/plot_comparison.py 生成覆盖率/熵曲线图，与 Gungnir/Luori/AddrProbe 等基线对比。",
    "步骤 5（独立场景）：仅做 APD 探测而不生成候选。python3 standalone_apd.py --input prefixes.txt --output apd_result.json --probe-count 16 --timeout 2.0 --concurrency 8；可对任意外部前缀列表（例如 BGP 全表、Gungnir 的候选集）运行 FRP 判定，并通过树对齐（Tree Alignment，--ta）细化到子前缀。日常运维可配合 cron 定期跑在线推理，每日/周落盘一次快照。",
]


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    doc = Document(SOP_PATH)

    # 1) Title
    rewrite_paragraph(doc.paragraphs[0], TITLE, bold=True, size_pt=24)

    # 2) Tables
    tables = doc.tables
    assert len(tables) >= 4, f"expected >=4 tables, got {len(tables)}"

    for (r, c), txt in TABLE_ENV.items():
        set_cell_text(tables[0].rows[r].cells[c], txt, size_pt=12)
    for (r, c), txt in TABLE_SRC.items():
        set_cell_text(tables[1].rows[r].cells[c], txt, size_pt=11)
    for (r, c), txt in TABLE_OWNER.items():
        set_cell_text(tables[2].rows[r].cells[c], txt, size_pt=12)
    for (r, c), txt in TABLE_REMOTE.items():
        set_cell_text(tables[3].rows[r].cells[c], txt, size_pt=11)

    # 3) Replace the "…" placeholders under "项目安装步骤" (paragraphs 8..12)
    install_para_idxs = [8, 9, 10, 11, 12]
    for pi, step in zip(install_para_idxs, INSTALL_STEPS):
        rewrite_paragraph(doc.paragraphs[pi], step,
                          bold=False, size_pt=12, indent_pt=24)

    # 4) Replace the "…" placeholders under "项目运行步骤" (paragraphs 14..18)
    run_para_idxs = [14, 15, 16, 17, 18]
    for pi, step in zip(run_para_idxs, RUN_STEPS):
        rewrite_paragraph(doc.paragraphs[pi], step,
                          bold=False, size_pt=12, indent_pt=24)

    out = SOP_PATH
    doc.save(out)
    print(f"saved -> {out}")


if __name__ == "__main__":
    main()
