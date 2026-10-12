#!/usr/bin/env python3
"""
Convert Heimdall.tex to Heimdall.docx following the IEEE conference template
style from 6FRP.docx.

Since the template uses strict OOXML namespaces that python-docx cannot read,
we fix the namespace in a copy, then use it as a template.
"""

import os
import shutil
import zipfile
import tempfile
from docx import Document
from docx.shared import Pt, Inches, Cm, Emu, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml.ns import qn
from docx.oxml import OxmlElement

# ============================================================
# Step 1: Fix the template namespace so python-docx can open it
# ============================================================

TEMPLATE_PATH = '/root/6FRP/paper/6FRP.docx'
FIXED_TEMPLATE = '/root/6FRP/paper/6FRP_fixed.docx'
OUTPUT_PATH = '/root/6FRP/paper/Heimdall.docx'

# Namespace replacements: strict -> transitional
NS_REPLACEMENTS = {
    b'http://purl.oclc.org/ooxml/officeDocument/relationships/officeDocument':
        b'http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument',
    b'http://purl.oclc.org/ooxml/officeDocument/relationships/':
        b'http://schemas.openxmlformats.org/officeDocument/2006/relationships/',
    b'http://purl.oclc.org/ooxml/wordprocessingml/main':
        b'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
    b'http://purl.oclc.org/ooxml/officeDocument/relationships':
        b'http://schemas.openxmlformats.org/officeDocument/2006/relationships',
    b'http://purl.oclc.org/ooxml/drawingml/main':
        b'http://schemas.openxmlformats.org/drawingml/2006/main',
    b'http://purl.oclc.org/ooxml/officeDocument/math':
        b'http://schemas.openxmlformats.org/officeDocument/2006/math',
    b'http://purl.oclc.org/ooxml/officeDocument/extendedProperties':
        b'http://schemas.openxmlformats.org/officeDocument/2006/extended-properties',
}

def fix_docx_namespaces(src, dst):
    """Copy a strict-mode docx to transitional mode by replacing namespaces."""
    with zipfile.ZipFile(src, 'r') as zin:
        with zipfile.ZipFile(dst, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                data = zin.read(item.filename)
                # Only process XML files
                if item.filename.endswith('.xml') or item.filename.endswith('.rels'):
                    for old_ns, new_ns in NS_REPLACEMENTS.items():
                        data = data.replace(old_ns, new_ns)
                zout.writestr(item, data)

print("Fixing template namespaces...")
fix_docx_namespaces(TEMPLATE_PATH, FIXED_TEMPLATE)

# ============================================================
# Step 2: Try to use the fixed template; if it fails, create from scratch
# ============================================================

try:
    doc = Document(FIXED_TEMPLATE)
    # Clear all content from the template
    for para in doc.paragraphs[:]:
        p = para._element
        p.getparent().remove(p)
    for table in doc.tables[:]:
        t = table._element
        t.getparent().remove(t)
    print("Using fixed template successfully.")
    USE_TEMPLATE = True
except Exception as e:
    print(f"Template loading failed ({e}), creating from scratch with IEEE style.")
    doc = Document()
    USE_TEMPLATE = False

# ============================================================
# Step 3: Set up page layout with IEEE two-column format
# ============================================================

section = doc.sections[0]
section.page_width = Inches(8.5)
section.page_height = Inches(11)
section.top_margin = Inches(0.75)
section.bottom_margin = Inches(1.0)
section.left_margin = Inches(0.625)
section.right_margin = Inches(0.625)

# Set two-column layout via XML
sectPr = section._sectPr
cols = sectPr.find(qn('w:cols'))
if cols is None:
    cols = OxmlElement('w:cols')
    sectPr.append(cols)
cols.set(qn('w:num'), '2')
cols.set(qn('w:space'), '360')  # 0.25 inch gap between columns (in twips: 360 = 0.25*1440)

# ============================================================
# Helper functions
# ============================================================

def set_run_font(run, name='Times New Roman', size=Pt(10), bold=False, italic=False):
    run.font.name = name
    run.font.size = size
    run.bold = bold
    run.italic = italic
    # Set East Asian font
    rPr = run._element.get_or_add_rPr()
    rFonts = rPr.find(qn('w:rFonts'))
    if rFonts is None:
        rFonts = OxmlElement('w:rFonts')
        rPr.insert(0, rFonts)
    rFonts.set(qn('w:eastAsia'), 'SimSun')

def add_title(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.space_before = Pt(0)
    para.space_after = Pt(6)
    run = para.add_run(text)
    set_run_font(run, size=Pt(24), bold=False)
    return para

def add_authors(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.space_before = Pt(6)
    para.space_after = Pt(6)
    run = para.add_run(text)
    set_run_font(run, size=Pt(11))
    return para

def add_affiliation(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.space_before = Pt(0)
    para.space_after = Pt(12)
    run = para.add_run(text)
    set_run_font(run, size=Pt(10), italic=True)
    return para

def add_section_heading(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.space_before = Pt(12)
    para.space_after = Pt(6)
    # IEEE uses small caps Roman numerals for section headings
    run = para.add_run(text)
    set_run_font(run, size=Pt(10), bold=False)
    # Small caps
    rPr = run._element.get_or_add_rPr()
    smallCaps = OxmlElement('w:smallCaps')
    smallCaps.set(qn('w:val'), 'true')
    rPr.append(smallCaps)
    return para

def add_subsection_heading(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para.space_before = Pt(10)
    para.space_after = Pt(4)
    run = para.add_run(text)
    set_run_font(run, size=Pt(10), italic=True)
    return para

def add_subsubsection_heading(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.LEFT
    para.space_before = Pt(8)
    para.space_after = Pt(2)
    run = para.add_run(text)
    set_run_font(run, size=Pt(10), italic=True)
    return para

def add_paragraph_heading(doc, text):
    """For \paragraph{} style headings - bold italic inline"""
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.space_before = Pt(6)
    para.space_after = Pt(0)
    run = para.add_run(text + "  ")
    set_run_font(run, size=Pt(10), bold=True, italic=True)
    return para

def add_body_text(doc, text, first_indent=True):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.space_before = Pt(0)
    para.space_after = Pt(0)
    if first_indent:
        para.paragraph_format.first_line_indent = Pt(14)
    run = para.add_run(text)
    set_run_font(run, size=Pt(10))
    return para

def add_abstract_heading(doc):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.space_before = Pt(12)
    para.space_after = Pt(6)
    run = para.add_run("Abstract")
    set_run_font(run, size=Pt(9), bold=True, italic=True)
    return para

def add_abstract_text(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.space_before = Pt(0)
    para.space_after = Pt(6)
    para.paragraph_format.first_line_indent = Pt(14)
    run = para.add_run(text)
    set_run_font(run, size=Pt(9), bold=True)
    return para

def add_keywords(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.space_before = Pt(0)
    para.space_after = Pt(12)
    para.paragraph_format.first_line_indent = Pt(14)
    run = para.add_run("Keywords—")
    set_run_font(run, size=Pt(9), bold=True, italic=True)
    run2 = para.add_run(text)
    set_run_font(run2, size=Pt(9), italic=True)
    return para

def add_bullet(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.space_before = Pt(2)
    para.space_after = Pt(2)
    para.paragraph_format.left_indent = Pt(28)
    para.paragraph_format.first_line_indent = Pt(-14)
    run = para.add_run("• " + text)
    set_run_font(run, size=Pt(10))
    return para

def add_numbered_item(doc, number, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.space_before = Pt(2)
    para.space_after = Pt(2)
    para.paragraph_format.left_indent = Pt(28)
    para.paragraph_format.first_line_indent = Pt(-14)
    run = para.add_run(f"{number}) " + text)
    set_run_font(run, size=Pt(10))
    return para

def add_equation(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.space_before = Pt(6)
    para.space_after = Pt(6)
    run = para.add_run(text)
    set_run_font(run, size=Pt(10), italic=True)
    return para

def add_table_caption(doc, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    para.space_before = Pt(12)
    para.space_after = Pt(6)
    run = para.add_run("TABLE ")
    set_run_font(run, size=Pt(8))
    # small caps
    rPr = run._element.get_or_add_rPr()
    smallCaps = OxmlElement('w:smallCaps')
    smallCaps.set(qn('w:val'), 'true')
    rPr.append(smallCaps)
    run2 = para.add_run(text)
    set_run_font(run2, size=Pt(8))
    return para

def add_reference(doc, number, text):
    para = doc.add_paragraph()
    para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    para.space_before = Pt(1)
    para.space_after = Pt(1)
    para.paragraph_format.left_indent = Pt(14)
    para.paragraph_format.first_line_indent = Pt(-14)
    run = para.add_run(f"[{number}] ")
    set_run_font(run, size=Pt(8))
    run2 = para.add_run(text)
    set_run_font(run2, size=Pt(8))
    return para

# ============================================================
# Step 4: Build the document content
# ============================================================

print("Building document content...")

# Title
add_title(doc, "Heimdall: Dual-Scan State-Space Modeling for Closed-Loop Discovery of IPv6 Fully Responsive Prefixes")

# Authors
add_authors(doc, "Anonymous Authors")
add_affiliation(doc, "Anonymous Institution\nanon@anon.org")

# Abstract
add_abstract_heading(doc)
abstract_text = (
    "The vast 128-bit address space of IPv6 fundamentally undermines exhaustive "
    "scanning, and target-generation algorithms (TGAs) used in its place are "
    "increasingly distorted by Fully Responsive Prefixes (FRPs)—ranges "
    "in which every probed address appears reachable. Existing FRP-discovery "
    "systems either enumerate a fixed prefix length (MAPD), which scales poorly, "
    "or grow a length-dependent search tree (Luori), which couples decisions "
    "across nibble layers. More recently, Gungnir cast FRP prediction as "
    "autoregressive language modeling with a Transformer; however, its "
    "single-axis attention captures only within-prefix bit "
    "dependencies and is trained offline, leaving two sources of "
    "information unexploited: (i) peer FRPs sharing the same Autonomous System "
    "(AS) and (ii) the probing feedback that becomes available at deployment time."
    "\n\n"
    "We present Heimdall, a closed-loop FRP discovery system built "
    "around a Dual-Scan Hierarchical Mamba generator. Heimdall "
    "introduces a state-space model (SSM) with two orthogonal selective scans "
    "in every layer: an outer scan along the 128 bits of a single "
    "prefix, and an inner scan along the sample axis of all FRPs "
    "belonging to the same AS within a batch. An AS-conditioned gating "
    "mechanism fuses the two scans, while a coarse/medium/fine layer-wise "
    "masked language modeling objective, weighted by per-bit Shannon entropy, "
    "enables non-autoregressive three-stage generation. At deployment, "
    "Heimdall runs a closed-loop scanner that interleaves APD probing, "
    "bidirectional seed filtering, and on-the-fly fine-tuning so that every "
    "verified FRP and every miss directly steer the next round of generation. "
    "On 214,102 ICMPv6 seed FRPs across 299 ASes, Heimdall sustains "
    "≥0.94 bit-accuracy in three-stage filling, raises end-to-end "
    "discovery throughput by up to 1.9× over Gungnir at the same probing "
    "budget, and broadens AS coverage by 1.2×, while running entirely "
    "on commodity hardware with linear-time SSM inference."
)
add_abstract_text(doc, abstract_text)

# Keywords
add_keywords(doc, "IPv6, fully responsive prefix, network measurement, state-space model, Mamba, masked language modeling, closed-loop scanning.")

# ============================================================
# I. Introduction
# ============================================================
add_section_heading(doc, "I. Introduction")

add_body_text(doc, (
    "Internet-wide active scanning is a foundational primitive for security "
    "auditing [1], topology inference [3], and longitudinal Internet "
    "measurement [2]. In IPv4, exhaustive scanning of the 2³² address space "
    "is feasible; in IPv6, the 2¹²⁸-address space makes exhaustion fundamentally "
    "impossible, and scanners therefore rely on Target Generation Algorithms "
    "(TGAs) [4–8] that predict where active addresses are likely to live. "
    "TGAs in turn depend on a clean seed corpus. Unfortunately, a sizeable "
    "fraction of real IPv6 seed prefixes are Fully Responsive Prefixes (FRPs): "
    "blocks where every probed address replies, although these replies do not "
    "necessarily come from distinct hosts [9, 10]. FRPs poison TGA training, "
    "inflate hitlists, and waste probing budget."
))

add_body_text(doc, (
    "Three lines of work have addressed FRP discovery. APD and its "
    "multi-prefix extension MAPD [9] test 16 random addresses inside a "
    "candidate prefix; if all reply, the prefix is labeled FRP. MAPD must "
    "enumerate prefixes of a chosen length and is therefore unable to discover "
    "FRPs of arbitrary granularity. Luori [12] formulates FRP discovery as a "
    "Monte-Carlo Tree-Search problem, expanding the IPv6 address tree nibble "
    "by nibble; the length-dependent expansion limits scalability and forces "
    "one tree per protocol/port. Most recently, Gungnir [13] treats the "
    "problem as autoregressive language modeling: a Transformer decoder emits "
    "prefix bits conditioned on AS, organization, business category, and "
    "protocol-port tokens, generating candidates that an APD module then "
    "verifies, with a granularity-correction step to reduce length errors. "
    "Gungnir reports up to 27× inference speed-up over Luori and 4.2× wider "
    "address coverage, but it still leaves three lessons unexploited:"
))

add_bullet(doc, (
    "L1: Peer-FRP signal is wasted. Inside the same AS, FRPs cluster heavily "
    "(§III); yet the Transformer decoder attends only within a prefix and never "
    "compares bits across sibling FRPs of the same AS in a batch."
))
add_bullet(doc, (
    "L2: Quadratic attention scales poorly. Self-attention is O(L²) in sequence "
    "length. For long contextual sequences (BGP prefix + AS + organization + "
    "category + sub-category + protocol + 128 bits), this overhead shows up "
    "directly in pre-training and inference time."
))
add_bullet(doc, (
    "L3: Generation is open-loop. Once trained, Gungnir produces the entire "
    "candidate list before any probing happens; probing feedback only enters at "
    "the post-hoc granularity step. Misleading prefixes are therefore "
    "re-generated round after round until the operator manually refreshes the "
    "seed set."
))

add_body_text(doc, (
    "Our approach. We present Heimdall, a closed-loop FRP discovery system in "
    "which the predictor is a Dual-Scan Hierarchical Mamba (DS-HMamba). "
    "Heimdall makes three deliberate departures from Gungnir."
))

add_body_text(doc, (
    "(i) Two orthogonal selective scans. Every DS-HMamba layer runs two Mamba "
    "SSM blocks [14]: an outer scan along the 128-bit axis of one prefix (the "
    "same axis Gungnir uses), and an inner scan along the sample axis grouping "
    "all FRPs of the same AS in a batch. The inner scan injects peer-AS "
    "evidence—common nibble allocations, prefix-length distributions, "
    "sub-prefix patterns—into every bit position, recovering the cross-FRP "
    "signal that single-axis attention discards. An AS-conditioned softmax gate "
    "fuses the two scans, so the model can decide per-AS how much to rely on "
    "within-prefix versus cross-prefix structure."
))

add_body_text(doc, (
    "(ii) Linear-time, non-autoregressive generation. Mamba's selective scan "
    "has linear complexity in L, so DS-HMamba's runtime grows as O(LD) rather "
    "than O(L²D). We pair this with a coarse/medium/fine masked language "
    "modeling objective: bits 0–31 (routing/aggregation), 32–63 (subnetting), "
    "and 64–127 (interface) are masked layer-by-layer during training, and "
    "filled stage-by-stage at inference. This three-pass scheme captures the "
    "natural hierarchy of IPv6 addressing while permitting parallel filling "
    "within a stage. A per-bit Shannon-entropy weight biases learning toward "
    "genuinely variable bits and away from the near-constant high bits that "
    "BGP already pins."
))

add_body_text(doc, (
    "(iii) Closed-loop probing with bidirectional seed filtering. At deployment, "
    "Heimdall interleaves three steps per round: generate candidates with "
    "DS-HMamba; run APD on candidates that survive a bidirectional seed filter "
    "(Level-1 rejects exact duplicates of known seeds, Level-2 silently discards "
    "candidates that fall inside already-known seed ranges, since they cannot "
    "extend the FRP frontier); and fine-tune the model on the round's positive "
    "(confirmed FRP) and negative (failed APD) feedback. An adaptive temperature "
    "schedule reads the Level-2 ratio and the empirical FRP rate, ramping "
    "exploration up when the model spins inside known regions and reining it "
    "back in when a fresh vein is found. Together with APD's iterative "
    "granularity refinement, the loop turns probing feedback into an online "
    "training signal, so each round directly amends the next."
))

add_body_text(doc, "Contributions. This paper makes the following contributions:")

add_bullet(doc, (
    "We are the first to apply selective state-space models to IPv6 FRP "
    "discovery. The proposed Dual-Scan Mamba layer simultaneously captures "
    "within-prefix bit dependencies and cross-prefix peer-AS structure, and a "
    "learnable AS-conditioned gate adaptively fuses the two views."
))
add_bullet(doc, (
    "We design a coarse/medium/fine layer-wise MLM objective with per-bit "
    "entropy weighting, enabling non-autoregressive three-stage filling that "
    "is faithful to IPv6's allocation hierarchy. Inference is linear in "
    "sequence length."
))
add_bullet(doc, (
    "We turn FRP discovery into a closed loop. A two-level SeedFilter, an "
    "adaptive temperature controller and an online APD-feedback fine-tuning "
    "step let the model improve between rounds without operator intervention."
))
add_bullet(doc, (
    "We implement Heimdall on commodity CPU/GPU hardware and evaluate it on "
    "214,102 ICMPv6 seed FRPs across 299 ASes from a public BGP table. "
    "Heimdall improves discovery throughput by up to 1.9× and AS coverage by "
    "1.2× over Gungnir at the same probing budgets, while reducing wall-clock "
    "training time by 1.7× at equal model capacity. Code and the curated FRP "
    "list will be released."
))

# ============================================================
# II. Background and Related Work
# ============================================================
add_section_heading(doc, "II. Background and Related Work")

add_subsection_heading(doc, "A. Fully Responsive Prefixes")
add_body_text(doc, (
    "Murdock et al. [4] first observed that scans of certain IPv6 prefixes "
    "elicited a uniformly affirmative response, and called such ranges aliased "
    "prefixes. Zirngibl et al. [10] formalised Fully Responsive Prefixes (FRPs) "
    "as ranges with 100% probe-reply rate, dropping the assumption that all "
    "addresses must terminate at a single host. In IPv4, the analogous "
    "phenomenon is Highly Responsive Prefixes [11]. FRPs originate from CDN "
    "load balancers, honeypot mappings, IPv6 NAT/proxy address pools, and the "
    "net.ipv6.ip_nonlocal_bind sysctl which lets a single host bind any address "
    "inside a configured range."
))

add_subsection_heading(doc, "B. FRP Probing")
para = add_body_text(doc, "", first_indent=False)
run = para.add_run("APD/MAPD. ")
set_run_font(run, size=Pt(10), bold=True)
run2 = para.add_run(
    "APD [9] samples 16 random addresses inside a candidate prefix and labels "
    "it FRP iff all 16 reply. MAPD enumerates many candidate prefixes of a "
    "fixed length and runs APD on each; it is comprehensive at small lengths "
    "but combinatorially expensive at long ones."
)
set_run_font(run2, size=Pt(10))

para = add_body_text(doc, "", first_indent=False)
run = para.add_run("Luori. ")
set_run_font(run, size=Pt(10), bold=True)
run2 = para.add_run(
    "Luori [12] models FRP search as MCTS over the address tree. Each tree "
    "node represents a prefix; expansion adds a nibble to the right. Its "
    "layer-by-layer expansion couples decisions across depths and forces a "
    "separate tree per protocol/port."
)
set_run_font(run2, size=Pt(10))

para = add_body_text(doc, "", first_indent=False)
run = para.add_run("Gungnir. ")
set_run_font(run, size=Pt(10), bold=True)
run2 = para.add_run(
    "Gungnir [13] is the first generative-AI approach to FRP discovery. A "
    "Transformer decoder reads a sequence consisting of (protocol-port, "
    "category, sub-category, organization, AS, BGP prefix, FRP-pattern bits) "
    "and is trained autoregressively with cross-entropy. At inference, Top-p "
    "sampling produces candidate FRP patterns, which are then probed by an APD "
    "module and granularity-corrected. Gungnir reports SOTA results but suffers "
    "from quadratic self-attention, single-axis modeling, and an open-loop "
    "training/probing pipeline (§I)."
)
set_run_font(run2, size=Pt(10))

add_subsection_heading(doc, "C. Selective State-Space Models")
add_body_text(doc, (
    "Mamba [14] introduced selective state-space models (S6) with "
    "input-dependent state transition coefficients. An S6 block runs a "
    "recurrence of the form hₜ = Āₜhₜ₋₁ + B̄ₜuₜ, yₜ = Cₜhₜ with Āₜ, B̄ₜ "
    "generated from uₜ at runtime, executed in parallel via a hardware-aware "
    "scan. Mamba models match Transformers' quality on language and audio tasks "
    "at O(L) complexity, and have been extended along multiple axes (e.g., "
    "Vision Mamba [15]). We are, to our knowledge, the first to apply selective "
    "scans to IPv6 address generation."
))

# ============================================================
# III. Problem Analysis and Definitions
# ============================================================
add_section_heading(doc, "III. Problem Analysis and Definitions")

add_subsection_heading(doc, "A. Notation")
add_body_text(doc, (
    "We treat each FRP as a 128-bit binary string b ∈ {0,1}¹²⁸ paired with a "
    "prefix length ℓ ∈ {0,…,128}. A seed corpus S = {(bᵢ, ℓᵢ)}ᵢ₌₁ᴺ is "
    "annotated with metadata mᵢ = (ASᵢ, ORGᵢ, CAT1ᵢ, CAT2ᵢ, PROTOᵢ, GEOᵢ, "
    "RTTᵢ) obtained by longest-prefix matching against a public BGP table [17] "
    "and CAIDA's AS–organization mapping [18]."
))

add_subsection_heading(doc, "B. Why Dual-Axis Modeling")
add_body_text(doc, (
    "Following Gungnir, we measure the pattern entropy of FRPs across multiple "
    "slicings of the seed set. Let H_pattern(S, ℓ) = −Σ P(x)log P(x) be the "
    "empirical entropy of ℓ-bit prefix patterns in S. As shown in Fig. 1 "
    "(reproduced on our 214K seeds), H_pattern remains far below the log₂ "
    "theoretical maximum of a uniform distribution at every ℓ ∈ [1,20], and "
    "drops even further when the corpus is sliced by AS, organization, or "
    "business category. Two implications follow:"
))

add_bullet(doc, (
    "Within-prefix structure (the bit-by-bit autoregressive signal) is "
    "informative: bits are highly correlated with their predecessors. This "
    "motivates the outer scan."
))
add_bullet(doc, (
    "Cross-prefix structure (similarity between FRPs of the same AS) is also "
    "informative, and is multiplicatively stronger than within-prefix structure "
    "when conditioned on the AS. Yet a Transformer decoder operating on one "
    "prefix at a time cannot see other prefixes of the same AS at training time "
    "except through its embedding lookup. We argue that an additional inner scan "
    "along the sample axis is required to explicitly exploit this signal."
))

add_subsection_heading(doc, "C. Why Closed-Loop Generation")
add_body_text(doc, (
    "A static generator faces an unavoidable distributional shift: the seed "
    "corpus is sampled from past Internet measurements, but the FRP landscape "
    "drifts as ASes reconfigure load balancers, deploy IPv6 honeypots, or roll "
    "out new CDN ranges. Open-loop generation rediscovers known FRPs and "
    "re-emits known mistakes. We therefore frame FRP discovery as a sequential "
    "decision problem: at each round t, the agent emits a candidate set Cₜ, "
    "observes APD verdicts, and updates its policy. This is the optimization "
    "target the autoregressive generation literature already minimizes, but "
    "applied in an online setting. We instantiate it via per-round MLM "
    "fine-tuning on positive and negative samples (§IV-E)."
))

# ============================================================
# IV. Heimdall Design
# ============================================================
add_section_heading(doc, "IV. Heimdall Design")

add_subsection_heading(doc, "A. Overview")
add_body_text(doc, (
    "Heimdall has three modules (Fig. 1): (i) Semantic learning pre-trains the "
    "DS-HMamba generator offline on the curated seed corpus with layer-wise MLM "
    "(§IV-B, §IV-C). (ii) Prefix inference performs three-stage masked filling "
    "under metadata and BGP constraints to emit candidate FRPs (§IV-D). "
    "(iii) Closed-loop online scanning probes candidates with APD, applies "
    "bidirectional seed filtering, runs granularity refinement, and fine-tunes "
    "the model on the round's feedback before the next generation cycle (§IV-E)."
))

add_subsection_heading(doc, "B. Dual-Scan Hierarchical Mamba")

add_subsubsection_heading(doc, "1) Embedding")
add_body_text(doc, (
    "Each input prefix b ∈ {0, 1, MASK}¹²⁸ is embedded by a bit embedding "
    "E_bit ∈ ℝ³ˣᵈ, a positional embedding E_pos ∈ ℝ¹²⁸ˣᵈ, and a stage "
    "embedding E_stage ∈ ℝ³ˣᵈ that marks each bit as belonging to coarse / "
    "medium / fine. Metadata m = (AS, ORG, CAT1, CAT2, PROTO, GEO, RTT) is "
    "projected through a per-field embedding and concatenated, then linearly "
    "mapped to a single conditioning vector c ∈ ℝᵈ that is broadcast and added "
    "at every bit position:"
))
add_equation(doc, "X⁽⁰⁾ = E_bit(b) + E_pos + E_stage + 1₁₂₈ cᵀ")

add_subsubsection_heading(doc, "2) Outer scan (within-prefix)")
add_body_text(doc, (
    "A standard Mamba block [14] processes the bit axis. The recurrence runs "
    "once per bit, giving O(L·D·N) total work. Y_out ∈ ℝᴮˣ¹²⁸ˣᵈ captures "
    "within-prefix bit dependencies analogous to the masked self-attention of a "
    "causal Transformer, but in linear time."
))

add_subsubsection_heading(doc, "3) Inner scan (within-AS, cross-prefix)")
add_body_text(doc, (
    "The inner scan operates on the sample axis. Within a batch of B prefixes, "
    "samples are first grouped by AS; for each group gₐ = {i : ASᵢ = a} of "
    "size Gₐ, an inner Mamba block processes the Gₐ-element sequence at every "
    "bit position ℓ. This lets the model condition the bit at position ℓ in "
    "prefix i on the same position ℓ in other prefixes of the same AS. For "
    "singleton groups (Gₐ = 1) the inner scan degenerates to identity. The "
    "inner scan thus exposes shared nibble allocations, common prefix lengths, "
    "and characteristic sub-prefix patterns that the outer scan cannot see."
))

add_subsubsection_heading(doc, "4) AS-conditioned gating fusion")
add_body_text(doc, (
    "Given Y_out and Y_in, the AS conditioning vector c produces a 2-way "
    "softmax gate:"
))
add_equation(doc, "[g_out, g_in] = softmax(W_g · SiLU(W_g' c))")
add_body_text(doc, "broadcast across bit positions. The fused output is:", first_indent=False)
add_equation(doc, "X⁽ℓ⁺¹⁾ = LayerNorm(g_out · Y_out + g_in · Y_in + α · X⁽ℓ⁾)")
add_body_text(doc, (
    "with α a learnable residual scale initialized to 0.1. Stacking N DS-HMamba "
    "layers gives the final feature tensor; a linear head projects to bit logits "
    "ẑ ∈ ℝᴮˣ¹²⁸ˣ²."
))

add_subsection_heading(doc, "C. Layer-wise MLM Training")
add_body_text(doc, (
    "Inspired by IPv6's three-tier addressing (routing /48 / subnet /64 / "
    "interface /128 [16]) we partition bit indices into coarse [0,32), medium "
    "[32,64), and fine [64,128). At every step we sample a stage uniformly and "
    "replace the corresponding bit slice with the MASK token; the model is "
    "trained to reconstruct the masked bits. Only positions that fall within the "
    "prefix's declared length contribute to the loss."
))

para = add_paragraph_heading(doc, "Entropy-weighted cross-entropy.")
run2 = para.add_run(
    "We pre-compute the empirical Shannon entropy Hᵢ of every bit position i "
    "on the seed corpus and form bit weights wᵢ = Hᵢ/H̄. The training loss is:"
)
set_run_font(run2, size=Pt(10))

add_equation(doc, "L = (1/|M|) Σ_{(i,ℓ)∈M} w_ℓ · CE(ẑ_{i,ℓ}, b*_{i,ℓ})")

add_body_text(doc, (
    "where M is the set of (sample, bit) positions that are masked and inside "
    "the prefix. Bits that are nearly constant (e.g., the leading routing-prefix "
    "bits) thus receive less gradient than bits that genuinely vary, focusing "
    "capacity on informative positions."
))

add_subsection_heading(doc, "D. Three-Stage Prefix Inference")
add_body_text(doc, "At deployment we generate from coarse to fine:")

add_numbered_item(doc, "1", "Initialize b ← MASK¹²⁸.")
add_numbered_item(doc, "2", (
    "BGP locking: for every sample, the top-ℓ_bgp bits are replaced by the "
    "bits of the chosen BGP routing prefix and never masked, so generation is "
    "constrained to advertised ranges."
))
add_numbered_item(doc, "3", "Coarse fill: forward the model, sample bits in [0,32) from the temperature-softmaxed logits.")
add_numbered_item(doc, "4", "Medium fill: forward again, sample bits in [32,64).")
add_numbered_item(doc, "5", "Fine fill: forward again, sample bits in [64,128).")
add_numbered_item(doc, "6", "Trim to the target prefix length and emit.")

add_body_text(doc, (
    "Within each stage, all bits can be sampled in parallel (akin to "
    "non-autoregressive translation), in contrast to Gungnir's left-to-right "
    "token-by-token Top-p decoding. An optional per-bit temperature Tᵢ is "
    "supported, allowing the operator to dial up exploration in high-entropy "
    "positions while keeping low-entropy bits deterministic."
))

add_subsection_heading(doc, "E. Closed-Loop Online Scanning")

add_subsubsection_heading(doc, "1) Bidirectional SeedFilter")
add_body_text(doc, (
    "At round t, candidates Cₜ pass through two filtering levels:"
))
add_bullet(doc, (
    "L1 (exact-duplicate rejection). If a candidate equals a known seed, drop "
    "it—reprobing wastes budget."
))
add_bullet(doc, (
    "L2 (in-known-range silent discard). If a candidate is a strict subnet of "
    "a known seed, the candidate cannot enlarge the FRP frontier; we silently "
    "discard it without generating a negative training sample, since it might "
    "still be a genuine FRP. We do, however, log the count—this is the "
    "\"stuck-in-place\" signal that drives temperature adaptation."
))
add_body_text(doc, (
    "A candidate that covers a known seed (i.e., is a broader prefix) is kept: "
    "APD will either confirm it as a larger FRP or falsify it as a non-FRP, "
    "both of which are valid training signals."
))

add_subsubsection_heading(doc, "2) APD probing and granularity refinement")
add_body_text(doc, (
    "For each surviving candidate, the APD module sends ICMPv6 echo requests "
    "to 16 randomly chosen sub-addresses (per [9]) using raw sockets. If all "
    "16 reply, the candidate is provisionally an FRP and forwarded to the "
    "refinement step which iteratively halves the prefix length to seek the "
    "largest still-fully-responsive prefix, mirroring Gungnir's "
    "granularity-correction logic but inside the same closed loop."
))

add_subsubsection_heading(doc, "3) Online fine-tuning")
add_body_text(doc, (
    "After APD, round t yields a positive set F⁺ₜ (confirmed FRPs) and a "
    "negative set F⁻ₜ (failed candidates). We perform a small number of MLM "
    "updates:"
))
add_equation(doc, "θ ← θ − η ∇_θ [L_MLM(F⁺ₜ) − λ L_MLM(F⁻ₜ)]")
add_body_text(doc, (
    "with a small λ = 0.1 to discourage—without forbidding—the patterns "
    "observed in the negative set. The learning rate η is 10% of the "
    "pre-training rate and the gradient is clipped at norm 1 to keep the model "
    "close to its offline minimum."
))

add_subsubsection_heading(doc, "4) Adaptive temperature")
add_body_text(doc, (
    "Two signals modulate the sampling temperature T between rounds: "
    "r_L2⁽ᵗ⁾ = |{c ∈ Cₜ : c is L2-discarded}| / |Cₜ|, and "
    "r_FRP⁽ᵗ⁾ = |F⁺ₜ| / |Cₜ \\ filtered|. "
    "If r_L2⁽ᵗ⁾ > 0.3, the model is \"recycling\" known regions; we set "
    "T ← min(2.0, 1.1T). If r_FRP⁽ᵗ⁾ = 0, exploration is too narrow; same "
    "update rule. Conversely, if r_FRP⁽ᵗ⁾ > 0.5 we are in a vein and exploit "
    "it: T ← max(0.3, 0.95T)."
))

add_subsubsection_heading(doc, "5) Confirmed-FRP self-augmentation")
add_body_text(doc, (
    "Confirmed FRPs are appended to the SeedFilter at the end of every round. "
    "This grows the L1 set and prevents trivial re-discovery while expanding "
    "L2's reach over time."
))

# ============================================================
# V. Implementation
# ============================================================
add_section_heading(doc, "V. Implementation")

add_body_text(doc, (
    "We implement Heimdall in ~3k lines of Python. The DS-HMamba generator "
    "uses a CPU-friendly pure-PyTorch selective scan (replaceable by mamba-ssm "
    "CUDA kernels for higher throughput). The default model has d = 128, 4 "
    "DS-HMamba layers, SSM state size N = 16, and convolution kernel 4, "
    "totalling ~2.1M trainable parameters—an order of magnitude smaller than "
    "Gungnir's reported Transformer."
))

para = add_paragraph_heading(doc, "Data preparation.")
run2 = para.add_run(
    "The data_prep module merges RouteViews [17] BGP dumps with the CAIDA "
    "AS-to-organization map [18] and our seed FRP list. Longest-prefix matching "
    "gives every seed an AS, organization, country, primary category, and "
    "sub-category. Vocabularies are built in frequency order with id 0 reserved "
    "for \"unknown\"."
)
set_run_font(run2, size=Pt(10))

para = add_paragraph_heading(doc, "Entropy analysis.")
run2 = para.add_run(
    "The utils/entropy_analysis module computes per-bit Shannon entropy on the "
    "prepared corpus and exports a 128-dim weight vector consumed by the loss."
)
set_run_font(run2, size=Pt(10))

para = add_paragraph_heading(doc, "APD.")
run2 = para.add_run(
    "We reuse the ICMPv6 echo-request primitives of [9] via Python raw sockets, "
    "with concurrency 16 and a 2s per-attempt timeout. Granularity refinement "
    "halves the prefix length until APD fails. Hosts that cannot or should not "
    "be probed (per the operator's exclusion list) are blacklisted before probing."
)
set_run_font(run2, size=Pt(10))

para = add_paragraph_heading(doc, "Hardware.")
run2 = para.add_run(
    "All numbers in §VI were collected on a server with an Intel Xeon Gold "
    "6230R, 256 GB RAM, a single NVIDIA RTX-class GPU and a 100 Mbps uplink, "
    "matching the platform of [13]."
)
set_run_font(run2, size=Pt(10))

# ============================================================
# VI. Evaluation
# ============================================================
add_section_heading(doc, "VI. Evaluation")

add_subsection_heading(doc, "A. Setup")
add_body_text(doc, (
    "Our seed corpus consists of 214,102 ICMPv6 FRP prefixes covering 299 ASes, "
    "273 organizations, 51 countries, 15 business categories, and 24 "
    "sub-categories. We use 5% of the corpus for held-out validation. All "
    "probing experiments target the IPv6 BGP table snapshot of 2026-04-29 "
    "(227,894 prefixes)."
))

add_body_text(doc, "We compare four systems:")
add_bullet(doc, "MAPD [9]: enumerate /48 prefixes in the BGP table, run APD on each.")
add_bullet(doc, "Luori [12]: MCTS with the public hyper-parameters of the original paper.")
add_bullet(doc, (
    "Gungnir [13]: 6-layer Transformer decoder, d = 128, h = 8, Top-p = 0.6, "
    "granularity correction enabled."
))
add_bullet(doc, (
    "Heimdall (ours): 4-layer DS-HMamba, d = 128, N = 16, three-stage MLM, "
    "closed-loop online scanning with adaptive temperature, T₀ = 0.8."
))

para = add_paragraph_heading(doc, "Metrics.")
run2 = para.add_run(
    "We report Cov_Routing (number of distinct BGP prefixes that contain at "
    "least one discovered FRP), Cov_AS (number of distinct ASes covered), "
    "Cov_Space (Σ 2^(128−ℓ_p) over discovered FRPs p, normalised across prefix "
    "lengths so that one /48 counts the same as sixteen /52s), and the discovery "
    "rate ρ = |F⁺|/(|F⁺|+|F⁻|). These are the standard metrics used by "
    "Gungnir [13]."
)
set_run_font(run2, size=Pt(10))

add_subsection_heading(doc, "B. End-to-End Discovery")
add_body_text(doc, (
    "Table I reports ICMPv6 discovery against the three baselines. At every "
    "budget Heimdall wins both routing- and AS-coverage metrics; at B_prefix = 50 "
    "it discovers 32% more covered routing prefixes and 19% more ASes than "
    "Gungnir, and 2.8× more routing prefixes than MAPD. In terms of Cov_Space "
    "(omitted from the table for brevity), Heimdall is 1.4× Gungnir at "
    "B_prefix = 50 and ~5×10⁸ MAPD."
))

# Table I
add_table_caption(doc, "I. End-to-end ICMPv6 discovery at three probing budgets B_prefix ∈ {10, 30, 50}.")
table = doc.add_table(rows=5, cols=5)
table.alignment = WD_TABLE_ALIGNMENT.CENTER
# Header
headers = ['Method', 'B=10 Routing', 'B=10 AS', 'B=30 Routing', 'B=30 AS']
for i, h in enumerate(headers):
    cell = table.rows[0].cells[i]
    cell.text = h
    for run in cell.paragraphs[0].runs:
        run.font.size = Pt(8)
        run.bold = True

data = [
    ['MAPD', '660', '114', '809', '133'],
    ['Luori', '409', '95', '1202', '129'],
    ['Gungnir', '1158', '121', '1733', '156'],
    ['Heimdall', '1452', '147', '2284', '183'],
]
for r, row_data in enumerate(data):
    for c, val in enumerate(row_data):
        table.rows[r+1].cells[c].text = val
        for run in table.rows[r+1].cells[c].paragraphs[0].runs:
            run.font.size = Pt(8)

para = add_paragraph_heading(doc, "TCP/80.")
run2 = para.add_run(
    "On the same routing prefixes, Heimdall trained under PROTO=TCP/80 retains "
    "its lead, although the absolute numbers are lower because TCP/80 FRPs are "
    "sparser. Specifically, at B_prefix = 50 Heimdall reaches Cov_Routing = 1238 "
    "vs. Gungnir's 981 and Luori's 672."
)
set_run_font(run2, size=Pt(10))

add_subsection_heading(doc, "C. Operating Time")
add_body_text(doc, (
    "Table II shows that Heimdall further compresses the cost already reduced "
    "by Gungnir. Pre-training is 1.7× faster because Mamba's selective scan is "
    "linear in the (long) input sequence. Inference is 1.5× faster per protocol "
    "because three parallel masked-fill passes replace 128 autoregressive steps. "
    "Across N = 5 protocols, Heimdall completes the full pipeline in 12.0 hours, "
    "vs. Gungnir's 19.3 hours and Luori's 449.5 hours—a 37× speed-up over the "
    "MCTS baseline."
))

# Table II
add_table_caption(doc, "II. Wall-clock cost on N protocol/port targets.")
table2 = doc.add_table(rows=6, cols=4)
table2.alignment = WD_TABLE_ALIGNMENT.CENTER
headers2 = ['', 'Luori', 'Gungnir', 'Heimdall']
for i, h in enumerate(headers2):
    table2.rows[0].cells[i].text = h
    for run in table2.rows[0].cells[i].paragraphs[0].runs:
        run.font.size = Pt(8)
        run.bold = True

data2 = [
    ['Pre-training (h)', '≈4.9N', '6.8', '4.0'],
    ['Inference (h)', '≈85N', '≈2.5N', '1.6N'],
    ['Total (N=1)', '89.9', '9.3', '5.6'],
    ['Total (N=3)', '269.7', '14.3', '8.8'],
    ['Total (N=5)', '449.5', '19.3', '12.0'],
]
for r, row_data in enumerate(data2):
    for c, val in enumerate(row_data):
        table2.rows[r+1].cells[c].text = val
        for run in table2.rows[r+1].cells[c].paragraphs[0].runs:
            run.font.size = Pt(8)

add_subsection_heading(doc, "D. Closed-Loop Improvement")
add_body_text(doc, (
    "Table III tracks how the closed loop affects per-round discovery. Without "
    "online fine-tuning the rate steadily decays as the model exhausts its "
    "training distribution. With online fine-tuning the rate grows from round 1 "
    "to round 10, because every confirmed FRP joins the next training mini-batch "
    "and reshapes the AS-conditioned gating. After 10 rounds the closed-loop "
    "variant has discovered 1.47× more FRPs than the open-loop variant under "
    "the same probing budget."
))

# Table III
add_table_caption(doc, "III. Per-round discovery rate ρ and cumulative confirmed FRPs over a 10-round closed-loop run.")
table3 = doc.add_table(rows=3, cols=5)
table3.alignment = WD_TABLE_ALIGNMENT.CENTER
headers3 = ['Variant', 'Round 1', 'Round 3', 'Round 5', 'Round 10']
for i, h in enumerate(headers3):
    table3.rows[0].cells[i].text = h
    for run in table3.rows[0].cells[i].paragraphs[0].runs:
        run.font.size = Pt(8)
        run.bold = True

data3 = [
    ['w/o ft', '0.34 / 17', '0.30 / 47', '0.27 / 71', '0.24 / 119'],
    ['full', '0.34 / 17', '0.42 / 56', '0.48 / 91', '0.55 / 175'],
]
for r, row_data in enumerate(data3):
    for c, val in enumerate(row_data):
        table3.rows[r+1].cells[c].text = val
        for run in table3.rows[r+1].cells[c].paragraphs[0].runs:
            run.font.size = Pt(8)

add_subsection_heading(doc, "E. Ablations")
add_body_text(doc, (
    "Table IV ablates the four design choices that separate Heimdall from a "
    "vanilla Mamba. Each component contributes:"
))

# Table IV
add_table_caption(doc, "IV. Held-out bit-accuracy at each MLM stage.")
table4 = doc.add_table(rows=6, cols=4)
table4.alignment = WD_TABLE_ALIGNMENT.CENTER
headers4 = ['Variant', 'coarse', 'medium', 'fine']
for i, h in enumerate(headers4):
    table4.rows[0].cells[i].text = h
    for run in table4.rows[0].cells[i].paragraphs[0].runs:
        run.font.size = Pt(8)
        run.bold = True

data4 = [
    ['Outer-only Mamba', '0.881', '0.882', '0.911'],
    ['Inner-only Mamba', '0.864', '0.873', '0.901'],
    ['DS-HMamba w/o gate', '0.901', '0.905', '0.928'],
    ['DS-HMamba w/o entropy', '0.911', '0.913', '0.930'],
    ['Heimdall (full)', '0.927', '0.929', '0.943'],
]
for r, row_data in enumerate(data4):
    for c, val in enumerate(row_data):
        table4.rows[r+1].cells[c].text = val
        for run in table4.rows[r+1].cells[c].paragraphs[0].runs:
            run.font.size = Pt(8)

add_bullet(doc, (
    "Adding the inner scan to the outer scan (full DS-HMamba w/o gate versus "
    "Outer-only) lifts coarse accuracy by +2.0 points and fine by +1.7, "
    "confirming that peer-AS structure carries information not available to a "
    "single-prefix model."
))
add_bullet(doc, (
    "Replacing mean-pool fusion with the AS-conditioned gate lifts another +2.6 "
    "at coarse and +1.5 at fine, showing that which of the two scans matters "
    "depends on the AS."
))
add_bullet(doc, (
    "Per-bit entropy weighting yields a final +1.3 on average, consistent with "
    "its role of redirecting capacity to the genuinely variable bits."
))

para = add_paragraph_heading(doc, "Sampling temperature.")
run2 = para.add_run(
    "Following Gungnir we sweep T ∈ {0.2, 0.4, 0.6, 0.8}. Best end-to-end "
    "coverage is reached at T = 0.6, matching Gungnir's recommendation. However, "
    "when the adaptive controller is enabled the closed-loop variant becomes "
    "robust to T₀ in [0.4, 1.0]: starting at any value in this band, the "
    "temperature converges to ~0.55 within five rounds."
)
set_run_font(run2, size=Pt(10))

# ============================================================
# VII. Long-Term Multi-Protocol Probing and FRP Analysis
# ============================================================
add_section_heading(doc, "VII. Long-Term Multi-Protocol Probing and FRP Analysis")

add_body_text(doc, (
    "We ran Heimdall continuously for three months over 23 active protocols "
    "(ICMPv6, TCP, UDP). Compared against the publicly released hitlist [19] "
    "and Luori's FRP list, our crawl extends Cov_AS from 335 / 530 to 684 and "
    "Cov_Routing from 3,237 / 7,324 to 9,512. Three findings echo and refine "
    "the observations of [13]:"
))

para = add_paragraph_heading(doc, "Concentration.")
run2 = para.add_run(
    "Three organisations—Cloudflare, Amazon, and Akamai—contribute >2M ICMPv6 "
    "FRPs, dominated by fine-granularity /64–/56 prefixes consistent with "
    "ip_nonlocal_bind mass-binding."
)
set_run_font(run2, size=Pt(10))

para = add_paragraph_heading(doc, "Service vs. port mismatch.")
run2 = para.add_run(
    "Of 1.6M generated addresses in HTTP-FRPs, only 15% actually serve HTTP; "
    "the remainder respond at the TCP layer but provide no application content. "
    "We hypothesise an outbound-proxy IP pool design intended to defeat per-IP "
    "rate limits, with the side effect of evading single-IP-based DDoS defences."
)
set_run_font(run2, size=Pt(10))

para = add_paragraph_heading(doc, "Security tells.")
run2 = para.add_run(
    "Among SSH-FRPs we observed OpenSSH 7.2p2 retaining "
    "diffie-hellman-group14-sha1 in ~11% of fingerprints, strongly suggesting "
    "honeypot deployments. Among SSL/TLS-FRPs, ~23% share a single Cloudflare "
    "wildcard certificate; private-key compromise would simultaneously expose "
    "every covered FRP, motivating automated rotation."
)
set_run_font(run2, size=Pt(10))

# ============================================================
# VIII. Ethical Considerations
# ============================================================
add_section_heading(doc, "VIII. Ethical Considerations")

add_body_text(doc, (
    "We follow the Menlo report [20] and the community guidance of [21]. All "
    "scanning runs from a single vantage point at ≤1 Mbps with randomised "
    "destination ordering and no single batch exceeding 10 minutes. We maintain "
    "an opt-out list and honor every received exclusion request within 24 hours. "
    "No penetration testing is performed: vulnerabilities are inferred only from "
    "banners and version strings, never validated against live hosts."
))

# ============================================================
# IX. Conclusion
# ============================================================
add_section_heading(doc, "IX. Conclusion")

add_body_text(doc, (
    "We presented Heimdall, a closed-loop FRP discovery system that replaces "
    "Transformer-based autoregression with a Dual-Scan Hierarchical Mamba. By "
    "scanning bits within a prefix and samples within an AS, gating their fusion "
    "on AS metadata, training with entropy-weighted layer-wise MLM, and running "
    "an online APD-feedback loop with bidirectional seed filtering and adaptive "
    "temperature, Heimdall improves over the SOTA Gungnir by up to 1.9× in "
    "discovery throughput and 1.2× in AS coverage at identical probing budgets, "
    "while halving wall-clock time. We hope that the open release of the system, "
    "the curated FRP list, and the closed-loop methodology will accelerate "
    "measurement and defence research on the IPv6 ecosystem."
))

# ============================================================
# References
# ============================================================
add_section_heading(doc, "References")

references = [
    'Z. Durumeric, E. Wustrow, and J. A. Halderman, "ZMap: Fast Internet-wide scanning and its security applications," in USENIX Security, 2013.',
    'E. Rye and D. Levin, "IPv6 hitlists at scale: Be careful what you wish for," in ACM SIGCOMM, 2023.',
    'K. Vermeulen et al., "Diamond-Miner: Comprehensive discovery of the Internet\'s topology diamonds," in NSDI, 2020.',
    'A. Murdock, F. Li, P. Bramsen, Z. Durumeric, and V. Paxson, "Target generation for Internet-wide IPv6 scanning," in IMC, 2017.',
    'T. Yang, Z. Cai, B. Hou, and T. Zhou, "6Forest: An ensemble learning-based approach to target generation for Internet-wide IPv6 scanning," in IEEE INFOCOM, 2022.',
    'B. Hou, Z. Cai, K. Wu, J. Su, and Y. Xiong, "6Hit: A reinforcement learning-based approach to target generation for Internet-wide IPv6 scanning," in IEEE INFOCOM, 2021.',
    'G. Song et al., "DET: Enabling efficient probing of IPv6 active addresses," IEEE/ACM Trans. Netw., 2022.',
    'T. Yang, B. Hou, Z. Cai, K. Wu, T. Zhou, and C. Wang, "6Graph: A graph-theoretic approach to address pattern mining for Internet-wide IPv6 scanning," Computer Networks, 2022.',
    'O. Gasser et al., "Clusters in the expanse: Understanding and unbiasing IPv6 hitlists," in IMC, 2018.',
    'J. Zirngibl, L. Steger, P. Sattler, O. Gasser, and G. Carle, "Rusty clusters? Dusting an IPv6 research foundation," in IMC, 2022.',
    'P. Sattler, J. Zirngibl, M. Jonker, O. Gasser, G. Carle, and R. Holz, "Packed to the brim: Investigating the impact of highly responsive prefixes on Internet-wide measurement campaigns," Proc. ACM Networking, vol. 1, 2023.',
    'D. Cheng et al., "Luori: Active probing and evaluation of Internet-wide IPv6 fully responsive prefixes," in IEEE ICNP, 2024.',
    'C. Wei, Y. Liu, L. He, D. Cheng, and J. Zhou, "Gungnir: Autoregressive model for unified generation of IPv6 fully responsive prefixes," in IEEE ICNP, 2025.',
    'A. Gu and T. Dao, "Mamba: Linear-time sequence modeling with selective state spaces," arXiv:2312.00752, 2023.',
    'L. Zhu et al., "Vision Mamba: Efficient visual representation learning with bidirectional state-space model," in ICML, 2024.',
    'R. Hinden and S. Deering, "IP version 6 addressing architecture," RFC 4291, 2006.',
    'University of Oregon, "Route Views project," https://www.routeviews.org/, 2024.',
    'CAIDA, "AS-to-organization mapping dataset," https://www.caida.org/catalog/datasets/as-organizations/, 2024.',
    'TUM, "IPv6 hitlist service," https://ipv6hitlist.github.io/, 2025.',
    'E. Kenneally and D. Dittrich, "The Menlo report: Ethical principles guiding information and communication technology research," SSRN, 2014.',
    'C. Partridge and M. Allman, "Ethical considerations in network measurement papers," Commun. ACM, 2016.',
]

for i, ref in enumerate(references, 1):
    add_reference(doc, i, ref)

# ============================================================
# Step 5: Save
# ============================================================
doc.save(OUTPUT_PATH)
print(f"\nDone! Output saved to: {OUTPUT_PATH}")

# Clean up temp file
if os.path.exists(FIXED_TEMPLATE):
    os.remove(FIXED_TEMPLATE)
