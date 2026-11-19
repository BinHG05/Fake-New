"""
Generate enterprise-grade architecture diagram matching the user's reference format.
Aesthetic: Light theme, 5-column layout, Medallion Lakehouse cylinders,
branded badges, custom vector icons, and bottom guardrail banner.
100% Vector drawing with Segoe UI typography and zero font glyph warnings.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
from pathlib import Path

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "images" / "architecture_overview.png"
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

def create_enterprise_diagram():
    # Set Windows native clean font
    plt.rcParams['font.sans-serif'] = ['Segoe UI', 'Arial', 'DejaVu Sans']
    plt.rcParams['font.family'] = 'sans-serif'

    # 2:1 aspect ratio matching reference (24 x 12 inches @ 200 DPI = 4800 x 2400 px)
    fig = plt.figure(figsize=(24, 12), dpi=200, facecolor="#FFFFFF")
    ax = fig.add_subplot(111)
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 52)
    ax.axis("off")

    # =========================================================================
    # COLOR PALETTE (Strictly aligned with Smart Campus Reference)
    # =========================================================================
    c_navy = "#0F3C6E"          # Solid dark navy for main headers & brand
    c_header_sub = "#93C5FD"    # Light blue text in column headers
    c_border = "#CBD5E1"        # Soft border for column outer bodies
    c_card_bg = "#FFFFFF"       # Crisp white card fill
    c_card_border = "#E2E8F0"   # Light card border (Slate 200)
    c_text_title = "#0F172A"    # Dark slate title text (Slate 900)
    c_text_body = "#334155"     # Slate 700
    c_text_muted = "#64748B"    # Slate 500
    c_blue_arrow = "#2563EB"    # Royal blue for inter-column arrows
    c_green = "#059669"         # Emerald green for guardrails
    c_green_bg = "#ECFDF5"      # Light green fill
    c_green_border = "#6EE7B7"  # Green border

    # =========================================================================
    # VECTOR ICON HELPERS (Zero font glyph dependencies, crisp vector geometry)
    # =========================================================================
    def draw_check_circle(cx, cy, r, bg_color="#059669", check_color="#FFFFFF", lw=2.2, zorder=6):
        circle = patches.Circle((cx, cy), r, facecolor=bg_color, edgecolor="none", zorder=zorder)
        ax.add_patch(circle)
        # Vector checkmark coordinates
        chk_x = [cx - r * 0.42, cx - r * 0.10, cx + r * 0.45]
        chk_y = [cy - r * 0.02, cy - r * 0.35, cy + r * 0.38]
        ax.plot(chk_x, chk_y, color=check_color, linewidth=lw, solid_capstyle='round', solid_joinstyle='round', zorder=zorder+1)

    def draw_badge(x, y, w, h, text, bg_color, text_color="#FFFFFF", font_size=7.5, bold=True):
        badge = patches.FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.0,rounding_size=0.35",
            facecolor=bg_color, edgecolor="none", zorder=4
        )
        ax.add_patch(badge)
        weight = "bold" if bold else "normal"
        ax.text(x + w/2, y + h/2, text, color=text_color, fontsize=font_size,
                fontweight=weight, ha="center", va="center", zorder=5)

    def draw_cylinder(cx, cy, r, h, color, edge_color="#334155", zorder=4):
        # Cylinder body
        cyl = patches.Rectangle((cx - r, cy - h/2), 2*r, h, facecolor=color, edgecolor=edge_color, linewidth=0.9, zorder=zorder)
        ax.add_patch(cyl)
        # Bottom ellipse
        bot_e = patches.Ellipse((cx, cy - h/2), 2*r, r*0.8, facecolor=color, edgecolor=edge_color, linewidth=0.9, zorder=zorder)
        ax.add_patch(bot_e)
        # Top ellipse (drawn last so it covers body top)
        top_e = patches.Ellipse((cx, cy + h/2), 2*r, r*0.8, facecolor=color, edgecolor=edge_color, linewidth=0.9, zorder=zorder+1)
        ax.add_patch(top_e)

    def draw_database_icon(cx, cy, s=0.9, color="#2563EB", zorder=4):
        draw_cylinder(cx, cy + 0.4*s, 0.7*s, 0.4*s, color, edge_color="#1E293B", zorder=zorder)
        draw_cylinder(cx, cy - 0.3*s, 0.7*s, 0.4*s, color, edge_color="#1E293B", zorder=zorder)

    def draw_tree_icon(cx, cy, s=0.9, color="#0284C7", zorder=4):
        # Root node
        r_node = patches.Circle((cx, cy + 0.6*s), 0.24*s, facecolor=color, edgecolor="none", zorder=zorder+1)
        c1_node = patches.Circle((cx - 0.5*s, cy - 0.4*s), 0.22*s, facecolor=color, edgecolor="none", zorder=zorder+1)
        c2_node = patches.Circle((cx + 0.5*s, cy - 0.4*s), 0.22*s, facecolor=color, edgecolor="none", zorder=zorder+1)
        ax.add_patch(r_node)
        ax.add_patch(c1_node)
        ax.add_patch(c2_node)
        ax.plot([cx, cx - 0.5*s], [cy + 0.6*s, cy - 0.4*s], color=color, lw=1.5, zorder=zorder)
        ax.plot([cx, cx + 0.5*s], [cy + 0.6*s, cy - 0.4*s], color=color, lw=1.5, zorder=zorder)

    def draw_image_icon(cx, cy, s=0.9, color="#7C3AED", zorder=4):
        box = patches.FancyBboxPatch((cx - 0.7*s, cy - 0.5*s), 1.4*s, 1.0*s,
                                     boxstyle=f"round,pad=0.0,rounding_size={0.15*s}",
                                     facecolor="#F3E8FF", edgecolor=color, lw=1.2, zorder=zorder)
        ax.add_patch(box)
        sun = patches.Circle((cx - 0.35*s, cy + 0.15*s), 0.16*s, facecolor=color, edgecolor="none", zorder=zorder+1)
        ax.add_patch(sun)
        # Mountain polygon
        mtn_x = [cx - 0.6*s, cx - 0.1*s, cx + 0.2*s, cx + 0.6*s]
        mtn_y = [cy - 0.45*s, cy + 0.1*s, cy - 0.1*s, cy - 0.45*s]
        ax.fill(mtn_x, mtn_y, color=color, zorder=zorder+1)

    def draw_chart_icon(cx, cy, s=0.9, color="#059669", zorder=4):
        # 3 bars
        b1 = patches.Rectangle((cx - 0.6*s, cy - 0.5*s), 0.3*s, 0.4*s, facecolor=color, edgecolor="none", zorder=zorder)
        b2 = patches.Rectangle((cx - 0.15*s, cy - 0.5*s), 0.3*s, 0.7*s, facecolor=color, edgecolor="none", zorder=zorder)
        b3 = patches.Rectangle((cx + 0.30*s, cy - 0.5*s), 0.3*s, 1.0*s, facecolor=color, edgecolor="none", zorder=zorder)
        ax.add_patch(b1)
        ax.add_patch(b2)
        ax.add_patch(b3)

    def draw_network_icon(cx, cy, s=0.9, color="#2563EB", zorder=4):
        # 4 connected graph nodes
        coords = [(cx - 0.5*s, cy - 0.3*s), (cx - 0.4*s, cy + 0.4*s), (cx + 0.4*s, cy + 0.3*s), (cx + 0.5*s, cy - 0.4*s)]
        for x1, y1 in coords:
            for x2, y2 in coords:
                if x1 < x2:
                    ax.plot([x1, x2], [y1, y2], color="#94A3B8", lw=1.1, zorder=zorder)
        for x, y in coords:
            n = patches.Circle((x, y), 0.18*s, facecolor=color, edgecolor="#FFFFFF", lw=0.8, zorder=zorder+1)
            ax.add_patch(n)

    def draw_mail_icon(cx, cy, s=0.9, color="#2563EB", zorder=4):
        rect = patches.FancyBboxPatch((cx - 0.65*s, cy - 0.45*s), 1.3*s, 0.9*s,
                                      boxstyle=f"round,pad=0.0,rounding_size={0.15*s}",
                                      facecolor="#EFF6FF", edgecolor=color, lw=1.2, zorder=zorder)
        ax.add_patch(rect)
        # V flap
        ax.plot([cx - 0.6*s, cx, cx + 0.6*s], [cy + 0.4*s, cy - 0.05*s, cy + 0.4*s], color=color, lw=1.2, zorder=zorder+1)

    def draw_book_icon(cx, cy, s=0.9, color="#059669", zorder=4):
        # Open book pages
        p1 = patches.Rectangle((cx - 0.65*s, cy - 0.45*s), 0.6*s, 0.9*s, facecolor="#ECFDF5", edgecolor=color, lw=1.2, zorder=zorder)
        p2 = patches.Rectangle((cx + 0.05*s, cy - 0.45*s), 0.6*s, 0.9*s, facecolor="#ECFDF5", edgecolor=color, lw=1.2, zorder=zorder)
        ax.add_patch(p1)
        ax.add_patch(p2)
        ax.plot([cx - 0.5*s, cx - 0.15*s], [cy + 0.15*s, cy + 0.15*s], color=color, lw=1.0, zorder=zorder+1)
        ax.plot([cx - 0.5*s, cx - 0.15*s], [cy - 0.15*s, cy - 0.15*s], color=color, lw=1.0, zorder=zorder+1)
        ax.plot([cx + 0.15*s, cx + 0.5*s], [cy + 0.15*s, cy + 0.15*s], color=color, lw=1.0, zorder=zorder+1)
        ax.plot([cx + 0.15*s, cx + 0.5*s], [cy - 0.15*s, cy - 0.15*s], color=color, lw=1.0, zorder=zorder+1)

    # =========================================================================
    # 1. TOP HEADER BAR (Matching Smart Campus Reference)
    # =========================================================================
    # Header Graduation / Verification Cap Icon
    cap_x, cap_y = 3.2, 48.2
    # Cap diamond
    cap_diamond_x = [cap_x - 1.8, cap_x, cap_x + 1.8, cap_x]
    cap_diamond_y = [cap_y, cap_y + 1.1, cap_y, cap_y - 1.1]
    ax.fill(cap_diamond_x, cap_diamond_y, color=c_navy, zorder=4)
    # Cap skull base
    cap_base_x = [cap_x - 1.0, cap_x - 1.0, cap_x + 1.0, cap_x + 1.0]
    cap_base_y = [cap_y - 0.3, cap_y - 1.5, cap_y - 1.5, cap_y - 0.3]
    ax.fill(cap_base_x, cap_base_y, color=c_navy, zorder=3)
    # Tassel
    ax.plot([cap_x, cap_x + 1.9, cap_x + 2.0], [cap_y, cap_y - 0.6, cap_y - 1.8], color="#D97706", lw=2.0, zorder=5)

    # Main Title (2 Bold Lines)
    ax.text(6.0, 49.3, "MULTIMODAL DATA LAKEHOUSE", fontsize=19, fontweight="black",
            color=c_navy, va="center", ha="left")
    ax.text(6.0, 47.1, "CASCADE GNN INTELLIGENCE PLATFORM", fontsize=19, fontweight="black",
            color=c_navy, va="center", ha="left")

    # Thin Vertical Divider Line
    ax.plot([54.0, 54.0], [45.8, 50.4], color="#94A3B8", linewidth=1.5, zorder=2)

    # Subtitle / Tagline on Right (Predict • Understand • Support style)
    ax.text(56.0, 48.9, "Ingest  •  Transform  •  Propagate  •  Detect", fontsize=13.5, fontweight="bold",
            fontstyle="italic", color=c_navy, va="center", ha="left")
    ax.text(56.0, 46.9, "Production-grade Lakehouse with Multimodal Cascade Graph Neural Networks",
            fontsize=10.5, color=c_text_muted, va="center", ha="left")

    # Horizontal Rule below Header
    ax.plot([1.5, 98.5], [44.8, 44.8], color="#CBD5E1", linewidth=1.2, zorder=2)

    # =========================================================================
    # 2. COLUMN DEFINITIONS (5 COLUMNS)
    # =========================================================================
    cols = [
        {"id": 1, "x": 1.5,  "w": 16.5, "title": "1. DATA SOURCES", "sub": "Multi-source Reddit Data"},
        {"id": 2, "x": 20.0, "w": 23.8, "title": "2. ORCHESTRATION & LAKEHOUSE", "sub": "Reliable Medallion Pipeline"},
        {"id": 3, "x": 45.8, "w": 15.6, "title": "3. MULTIMODAL FEATURE STORE", "sub": "Dense Semantic & Vision Embeddings"},
        {"id": 4, "x": 63.4, "w": 16.8, "title": "4. ML & EXPLAINABILITY", "sub": "Accurate & Interpretable"},
        {"id": 5, "x": 82.2, "w": 16.3, "title": "5. ADVISOR PORTAL & INTERVENTION", "sub": "Proactive Verification Support"},
    ]

    col_bottom = 6.2
    col_top = 43.5
    col_h = col_top - col_bottom

    # Draw Column Outer Bodies & Headers
    for c in cols:
        x, w = c["x"], c["w"]
        # Outer container for column body
        body_box = patches.FancyBboxPatch(
            (x, col_bottom), w, col_h - 3.8,
            boxstyle="round,pad=0.0,rounding_size=0.8",
            facecolor="#F8FAFC", edgecolor=c_border, linewidth=1.2, zorder=1
        )
        ax.add_patch(body_box)

        # Header pill (Solid Dark Navy)
        hdr_box = patches.FancyBboxPatch(
            (x, col_top - 3.8), w, 3.8,
            boxstyle="round,pad=0.0,rounding_size=0.8",
            facecolor=c_navy, edgecolor=c_navy, linewidth=1.0, zorder=2
        )
        ax.add_patch(hdr_box)
        # Header text
        ax.text(x + w / 2, col_top - 1.4, c["title"], fontsize=10.5, fontweight="bold",
                color="#FFFFFF", ha="center", va="center", zorder=3)
        ax.text(x + w / 2, col_top - 2.8, c["sub"], fontsize=8.2,
                color=c_header_sub, ha="center", va="center", zorder=3)

    # Inter-column Connecting Arrows (Prominent blue arrows between columns)
    arrow_y = 25.0
    for i in range(len(cols) - 1):
        x_from = cols[i]["x"] + cols[i]["w"] + 0.3
        x_to = cols[i+1]["x"] - 0.3
        ax.annotate(
            "", xy=(x_to, arrow_y), xytext=(x_from, arrow_y),
            arrowprops=dict(arrowstyle="-|>", color=c_blue_arrow, lw=2.4, mutation_scale=18),
            zorder=6
        )

    # =========================================================================
    # 3. COLUMN 1: DATA SOURCES (5 Cards with Custom Vector Icons)
    # =========================================================================
    c1_x = 2.3
    c1_w = 14.9

    c1_items = [
        {"draw_icon": draw_database_icon, "color": "#2563EB", "title": "Reddit Posts JSONL", "desc": "10k+ Canonical Posts\nMetadata & Main Claims"},
        {"draw_icon": draw_tree_icon, "color": "#0284C7", "title": "Cascade Comment Trees", "desc": "Nested Reply Chains\nDepth 1–10+ Discussions"},
        {"draw_icon": draw_image_icon, "color": "#7C3AED", "title": "Raw Image Assets", "desc": "8,700+ Visual Files\nOriginal JPEG/PNG Formats"},
        {"draw_icon": draw_network_icon, "color": "#0D9488", "title": "Multi-Source Feeds", "desc": "Fakeddit, Reddit, Subreddits\nCross-domain Diversity"},
        {"draw_icon": draw_chart_icon, "color": "#16A34A", "title": "Binary Ground Truth", "desc": "Strict REAL (0) & FAKE (1)\nTrain / Val / Test Integrity"},
    ]

    curr_y = 34.0
    for itm in c1_items:
        card = patches.FancyBboxPatch(
            (c1_x, curr_y), c1_w, 5.0,
            boxstyle="round,pad=0.0,rounding_size=0.6",
            facecolor=c_card_bg, edgecolor=c_card_border, linewidth=1.0, zorder=2
        )
        ax.add_patch(card)
        # Draw custom vector icon on left
        itm["draw_icon"](c1_x + 1.8, curr_y + 2.5, s=1.1, color=itm["color"])

        ax.text(c1_x + 3.8, curr_y + 3.4, itm["title"], fontsize=8.8, fontweight="bold", color=c_text_title, va="center", zorder=3)
        ax.text(c1_x + 3.8, curr_y + 1.7, itm["desc"], fontsize=7.1, color=c_text_muted, va="center", zorder=3)
        curr_y -= 6.3

    # =========================================================================
    # 4. COLUMN 2: ORCHESTRATION & LAKEHOUSE (Dagster/DVC, Storage Cylinders, PySpark)
    # =========================================================================
    c2_x = 20.8
    c2_w = 22.2

    # Card 2.1: Orchestration Banner (DVC & Make / DagsHub - matching Dagster in ref)
    orch_card = patches.FancyBboxPatch(
        (c2_x, 34.8), c2_w, 4.2,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor="#EFF6FF", edgecolor="#BFDBFE", linewidth=1.1, zorder=2
    )
    ax.add_patch(orch_card)
    # DVC Octopus / Badge
    draw_badge(c2_x + 1.2, 35.8, 3.4, 2.2, "DVC", "#945DD6", font_size=8.5)
    ax.text(c2_x + 5.2, 37.6, "DVC Workflow Orchestration", fontsize=10.2, fontweight="bold", color=c_navy, va="center", zorder=3)
    ax.text(c2_x + 5.2, 36.0, "Reproducible 4-Stage DAG & DagsHub Remote Tracking", fontsize=7.6, color=c_text_muted, va="center", zorder=3)

    # Down arrow to storage container
    ax.annotate("", xy=(c2_x + c2_w/2, 34.3), xytext=(c2_x + c2_w/2, 34.8),
                arrowprops=dict(arrowstyle="-|>", color=c_blue_arrow, lw=1.6, mutation_scale=11), zorder=5)

    # Card 2.2: Lakehouse Storage Container with 3 3D-like Cylinders
    med_box = patches.FancyBboxPatch(
        (c2_x, 19.8), c2_w, 14.2,
        boxstyle="round,pad=0.0,rounding_size=0.8",
        facecolor="#FEFCE8", edgecolor="#FDE047", linewidth=1.2, zorder=2
    )
    ax.add_patch(med_box)
    ax.text(c2_x + c2_w/2, 32.8, "STORAGE  |  S3 / Lakehouse Partitioning", fontsize=8.4, fontweight="bold",
            color="#A16207", ha="center", va="center", zorder=3)

    # 3 Medallion Cylinders (Bronze, Silver, Gold)
    tiers = [
        {"name": "Bronze", "sub": "Raw JSONL\n& Raw Media", "x": c2_x + 1.2, "w": 5.9, "color": "#B45309", "bg": "#FFFBEB", "bd": "#FCD34D"},
        {"name": "Silver", "sub": "Cleaned Parquet\nDeduplicated", "x": c2_x + 8.1, "w": 5.9, "color": "#2563EB", "bg": "#EFF6FF", "bd": "#93C5FD"},
        {"name": "Gold", "sub": "PyG Graph\nFeature Store", "x": c2_x + 15.1, "w": 5.9, "color": "#EAB308", "bg": "#FEFCE8", "bd": "#FDE047"},
    ]
    for t in tiers:
        t_card = patches.FancyBboxPatch(
            (t["x"], 21.6), t["w"], 9.5,
            boxstyle="round,pad=0.0,rounding_size=0.6",
            facecolor=t["bg"], edgecolor=t["bd"], linewidth=1.0, zorder=3
        )
        ax.add_patch(t_card)
        draw_cylinder(t["x"] + t["w"]/2, 28.5, 0.95, 1.4, t["color"])
        ax.text(t["x"] + t["w"]/2, 25.8, t["name"], fontsize=9.2, fontweight="bold", color=t["color"], ha="center", va="center", zorder=4)
        ax.text(t["x"] + t["w"]/2, 23.4, t["sub"], fontsize=6.8, color=c_text_muted, ha="center", va="center", zorder=4)

    # Transition arrows between cylinders
    ax.annotate("", xy=(c2_x + 7.9, 26.5), xytext=(c2_x + 7.2, 26.5),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=1.6, mutation_scale=10), zorder=5)
    ax.annotate("", xy=(c2_x + 14.9, 26.5), xytext=(c2_x + 14.2, 26.5),
                arrowprops=dict(arrowstyle="-|>", color="#94A3B8", lw=1.6, mutation_scale=10), zorder=5)

    ax.text(c2_x + c2_w/2, 20.6, "Parquet Format  •  Medallion Architecture (source, event_date)", fontsize=7.4,
            fontstyle="italic", color="#78350F", ha="center", va="center", zorder=4)

    # Card 2.3: Two Engine Boxes Side-by-Side (matching dbt Core and DuckDB in reference)
    # Left box: PySpark
    pyspark_box = patches.FancyBboxPatch(
        (c2_x, 12.8), 10.8, 6.2,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor=c_card_bg, edgecolor=c_card_border, linewidth=1.0, zorder=2
    )
    ax.add_patch(pyspark_box)
    draw_badge(c2_x + 0.8, 16.5, 3.2, 1.8, "SPARK", "#EA580C", font_size=7.2)
    ax.text(c2_x + 4.4, 17.4, "PySpark 3.5", fontsize=9.2, fontweight="bold", color="#EA580C", va="center", zorder=3)
    ax.text(c2_x + 5.4, 15.0, "Broadcast Hash Join\nExploded reply tree", fontsize=7.0, color=c_text_body, ha="center", va="center", zorder=3)

    # Right box: PyArrow / Columnar Analytics
    pyarrow_box = patches.FancyBboxPatch(
        (c2_x + 11.4, 12.8), 10.8, 6.2,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor=c_card_bg, edgecolor=c_card_border, linewidth=1.0, zorder=2
    )
    ax.add_patch(pyarrow_box)
    draw_badge(c2_x + 12.2, 16.5, 3.4, 1.8, "ARROW", "#0284C7", font_size=7.0)
    ax.text(c2_x + 16.0, 17.4, "PyArrow Engine", fontsize=9.2, fontweight="bold", color="#0284C7", va="center", zorder=3)
    ax.text(c2_x + 16.8, 15.0, "Zero-copy columnar I/O\nSnappy compression", fontsize=7.0, color=c_text_body, ha="center", va="center", zorder=3)

    # Card 2.4: Data Quality Gate (Pydantic & Quarantine - matching Soda Core in reference)
    pyd_card = patches.FancyBboxPatch(
        (c2_x, 6.8), c2_w, 5.2,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor=c_green_bg, edgecolor=c_green_border, linewidth=1.2, zorder=2
    )
    ax.add_patch(pyd_card)
    draw_badge(c2_x + 1.2, 8.2, 3.6, 2.4, "QUALITY", "#059669", font_size=7.5)
    ax.text(c2_x + 5.4, 10.4, "Pydantic Data Contracts & Quarantine Gate", fontsize=9.2, fontweight="bold", color="#065F46", va="center", zorder=3)
    ax.text(c2_x + 5.4, 8.8, "• Schema validation: CanonicalPostRecord & split verification", fontsize=7.2, color="#047857", va="center", zorder=3)
    ax.text(c2_x + 5.4, 7.5, "• Auto-quarantine: isolates malformed JSON into rejected_records.jsonl", fontsize=6.8, color="#065F46", va="center", zorder=3)

    # =========================================================================
    # 5. COLUMN 3: MULTIMODAL FEATURE STORE (Dense Embeddings & Graph Topologies)
    # =========================================================================
    c3_x = 46.5
    c3_w = 14.2

    c3_items = [
        {"draw_icon": draw_network_icon, "color": "#2563EB", "title": "XLM-RoBERTa (Text)", "desc": "768-d Semantic Embeddings\nMulti-lingual Contextual NLP"},
        {"draw_icon": draw_image_icon, "color": "#10A37F", "title": "OpenAI CLIP (Vision)", "desc": "ViT-B/32 Visual Projections (512-d)\nCross-Modal Text Alignment (512-d)"},
        {"draw_icon": draw_chart_icon, "color": "#D946EF", "title": "Semantic Mismatch Score", "desc": "1-d Cosine Similarity (Text vs Image)\nDetects Out-of-Context Miscaptioning"},
        {"draw_icon": draw_tree_icon, "color": "#EA580C", "title": "Cascade Tree Topology", "desc": "Reply Tree Structure (Edge Index)\nParent-to-Child Information Flow"},
        {"draw_icon": draw_database_icon, "color": "#4F46E5", "title": "1,793-d Node Assembly", "desc": "Root Post: [768d Text | 1025d CLIP]\nComments: [768d Text | 1025d Zero-pad]"},
    ]

    curr_y = 34.0
    for itm in c3_items:
        card = patches.FancyBboxPatch(
            (c3_x, curr_y), c3_w, 5.0,
            boxstyle="round,pad=0.0,rounding_size=0.6",
            facecolor=c_card_bg, edgecolor=c_card_border, linewidth=1.0, zorder=2
        )
        ax.add_patch(card)
        itm["draw_icon"](c3_x + 1.8, curr_y + 2.5, s=1.1, color=itm["color"])

        ax.text(c3_x + 3.8, curr_y + 3.4, itm["title"], fontsize=8.8, fontweight="bold", color=c_text_title, va="center", zorder=3)
        ax.text(c3_x + 3.8, curr_y + 1.7, itm["desc"], fontsize=7.1, color=c_text_muted, va="center", zorder=3)
        curr_y -= 6.3

    # =========================================================================
    # 6. COLUMN 4: ML & EXPLAINABILITY (GNN, Tracking, Calibration, Survival Chart)
    # =========================================================================
    c4_x = 64.1
    c4_w = 15.4

    # Top Row: Two Model Boxes Side-by-Side (matching LightGBM and CatBoost in reference)
    m1_box = patches.FancyBboxPatch(
        (c4_x, 34.2), 7.4, 4.8,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor=c_card_bg, edgecolor=c_card_border, linewidth=1.0, zorder=2
    )
    ax.add_patch(m1_box)
    draw_badge(c4_x + 0.6, 36.8, 2.6, 1.6, "GATv2", "#8B5CF6", font_size=6.8)
    ax.text(c4_x + 3.7, 37.6, "PyG GATv2", fontsize=8.6, fontweight="bold", color=c_navy, va="center", zorder=3)
    ax.text(c4_x + 3.7, 35.4, "Attention heads on\npropagation graph", fontsize=6.6, color=c_text_body, va="center", zorder=3)

    m2_box = patches.FancyBboxPatch(
        (c4_x + 8.0, 34.2), 7.4, 4.8,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor=c_card_bg, edgecolor=c_card_border, linewidth=1.0, zorder=2
    )
    ax.add_patch(m2_box)
    draw_badge(c4_x + 8.6, 36.8, 2.6, 1.6, "Bi-GCN", "#D97706", font_size=6.6)
    ax.text(c4_x + 11.7, 37.6, "Bi-Directional", fontsize=8.4, fontweight="bold", color=c_navy, va="center", zorder=3)
    ax.text(c4_x + 11.7, 35.4, "Top-down / Bottom-up\nmessage passing", fontsize=6.6, color=c_text_body, va="center", zorder=3)

    # Down arrow in col 4
    ax.annotate("", xy=(c4_x + c4_w/2, 33.7), xytext=(c4_x + c4_w/2, 34.2),
                arrowprops=dict(arrowstyle="-|>", color=c_blue_arrow, lw=1.5, mutation_scale=10), zorder=5)

    # Middle Card: DagsHub / DVC Experiment Tracking (matching MLflow in reference)
    track_card = patches.FancyBboxPatch(
        (c4_x, 26.5), c4_w, 6.7,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor="#EFF6FF", edgecolor="#BFDBFE", linewidth=1.0, zorder=2
    )
    ax.add_patch(track_card)
    draw_badge(c4_x + 1.0, 29.8, 2.8, 2.2, "DVC", "#2563EB", font_size=7.5)
    ax.text(c4_x + 4.4, 31.5, "DagsHub Experiment Tracking", fontsize=9.2, fontweight="bold", color=c_navy, va="center", zorder=3)
    ax.text(c4_x + 4.4, 29.8, "• DVC pipeline lineage & remote metrics", fontsize=7.2, color=c_text_body, va="center", zorder=3)
    ax.text(c4_x + 4.4, 28.3, "• Single-archive remote caching (930MB)", fontsize=7.0, color=c_text_muted, va="center", zorder=3)

    # Down arrow in col 4
    ax.annotate("", xy=(c4_x + c4_w/2, 26.0), xytext=(c4_x + c4_w/2, 26.5),
                arrowprops=dict(arrowstyle="-|>", color=c_blue_arrow, lw=1.5, mutation_scale=10), zorder=5)

    # Third Card: Cross-Attention Fusion & SHAP / Explainability (matching SHAP in reference)
    shap_card = patches.FancyBboxPatch(
        (c4_x, 18.0), c4_w, 7.5,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor="#FDF4FF", edgecolor="#F0ABFC", linewidth=1.0, zorder=2
    )
    ax.add_patch(shap_card)
    draw_badge(c4_x + 1.0, 22.0, 2.8, 2.2, "ATTN", "#C026D3", font_size=7.2)
    ax.text(c4_x + 4.4, 23.8, "Cross-Attention Fusion", fontsize=9.2, fontweight="bold", color="#86198F", va="center", zorder=3)
    ax.text(c4_x + 4.4, 22.1, "• Multi-head attention: Text queries Vision", fontsize=7.2, color=c_text_body, va="center", zorder=3)
    ax.text(c4_x + 4.4, 20.6, "• Gated residual connection to preserve text", fontsize=7.0, color=c_text_muted, va="center", zorder=3)
    ax.text(c4_x + 4.4, 19.1, "• Edge-level attention weights for explainability", fontsize=7.0, color=c_text_muted, va="center", zorder=3)

    # Down arrow in col 4
    ax.annotate("", xy=(c4_x + c4_w/2, 17.5), xytext=(c4_x + c4_w/2, 18.0),
                arrowprops=dict(arrowstyle="-|>", color=c_blue_arrow, lw=1.5, mutation_scale=10), zorder=5)

    # Bottom Container: Optional / Deep Cascade & ROC Metrics (matching Survival Analysis in ref)
    surv_box = patches.FancyBboxPatch(
        (c4_x, 6.8), c4_w, 10.2,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor="#F8FAFC", edgecolor="#CBD5E1", linewidth=1.0, linestyle="--", zorder=2
    )
    ax.add_patch(surv_box)
    ax.text(c4_x + c4_w/2, 16.0, "Deep Cascade & Test Performance:", fontsize=7.4, fontstyle="italic", color=c_text_muted, ha="center", va="center", zorder=3)

    # Mini ROC / Cascade curve drawing
    curve_x = [c4_x + 1.2, c4_x + 2.4, c4_x + 3.8, c4_x + 5.2]
    curve_y = [10.0, 11.8, 13.5, 14.5]
    ax.plot(curve_x, curve_y, color="#2563EB", lw=2.2, zorder=4)
    # Coordinate axes
    ax.plot([c4_x + 1.0, c4_x + 5.4], [9.8, 9.8], color="#94A3B8", lw=1.0, zorder=3)
    ax.plot([c4_x + 1.0, c4_x + 1.0], [9.8, 14.8], color="#94A3B8", lw=1.0, zorder=3)

    # Performance stats text on the right
    ax.text(c4_x + 6.2, 14.2, "83.1% Test Accuracy", fontsize=8.6, fontweight="bold", color="#15803D", va="center", zorder=3)
    ax.text(c4_x + 6.2, 12.6, "0.9052 ROC-AUC", fontsize=8.0, fontweight="bold", color=c_navy, va="center", zorder=3)
    ax.text(c4_x + 6.2, 11.0, "Deep Cascades (>10):", fontsize=7.4, color=c_text_body, va="center", zorder=3)
    ax.text(c4_x + 6.2, 9.5, "93.3% Accuracy", fontsize=8.6, fontweight="black", color="#EA580C", va="center", zorder=3)
    ax.text(c4_x + c4_w/2, 7.8, "Fake Recall: 88.9% @ t = 0.38", fontsize=7.4, fontweight="bold", color="#166534", ha="center", va="center", zorder=3)

    # =========================================================================
    # 7. COLUMN 5: ADVISOR PORTAL & INTERVENTION (Flask, Docker, Action Cards)
    # =========================================================================
    c5_x = 83.0
    c5_w = 14.7

    # Card 5.1: Flask & Web Dashboard (matching Streamlit Advisor Dashboard in ref)
    dash_card = patches.FancyBboxPatch(
        (c5_x, 24.8), c5_w, 14.2,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor="#FEF2F2", edgecolor="#FECACA", linewidth=1.2, zorder=2
    )
    ax.add_patch(dash_card)
    # Crown / Red badge
    draw_badge(c5_x + 1.2, 35.8, 3.4, 2.2, "FLASK", "#DC2626", font_size=7.5)
    ax.text(c5_x + 5.2, 37.5, "Flask Inference API", fontsize=10.2, fontweight="bold", color="#991B1B", va="center", zorder=3)
    ax.text(c5_x + 5.2, 36.0, "Interactive Verification Portal", fontsize=7.6, color=c_text_muted, va="center", zorder=3)

    # Feature List with custom vector checkmark circles (blue circle with white check)
    dash_features = [
        "Real-Time Propagation Graph Builder",
        "Cascade Risk Scoring & Uncertainty Bounds",
        "Calibrated Decision Gate (t = 0.38)",
        "Single-Post & Batch Stream Prediction"
    ]
    f_y = 33.2
    for f in dash_features:
        draw_check_circle(c5_x + 1.8, f_y, 0.45, bg_color="#2563EB", check_color="#FFFFFF", lw=1.6, zorder=4)
        ax.text(c5_x + 2.8, f_y, f, fontsize=7.0, color="#7F1D1D", va="center", zorder=3)
        f_y -= 2.0

    # Downstream Action Cards (Side by side like Email Alerts & Recommended Catch-up Materials in ref)
    act1_card = patches.FancyBboxPatch(
        (c5_x, 14.2), 6.9, 9.6,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor=c_card_bg, edgecolor=c_card_border, linewidth=1.0, zorder=2
    )
    ax.add_patch(act1_card)
    draw_mail_icon(c5_x + 3.45, 21.6, s=1.1, color="#2563EB")
    ax.text(c5_x + 3.45, 19.4, "Alert Routing", fontsize=8.4, fontweight="bold", color=c_navy, ha="center", va="center", zorder=3)
    ax.text(c5_x + 3.45, 16.8, "Real-time Webhook\nFlagged to Reviewers\nHigh Risk (>0.38)", fontsize=6.8, color=c_text_muted, ha="center", va="center", zorder=3)

    act2_card = patches.FancyBboxPatch(
        (c5_x + 7.8, 14.2), 6.9, 9.6,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor=c_card_bg, edgecolor=c_card_border, linewidth=1.0, zorder=2
    )
    ax.add_patch(act2_card)
    draw_book_icon(c5_x + 11.25, 21.6, s=1.1, color="#059669")
    ax.text(c5_x + 11.25, 19.4, "Fact-Check DB", fontsize=8.4, fontweight="bold", color=c_navy, ha="center", va="center", zorder=3)
    ax.text(c5_x + 11.25, 16.8, "Automated Evidence\nContext Dispatch\nVerified Sources", fontsize=6.8, color=c_text_muted, ha="center", va="center", zorder=3)

    # Bottom Area: Banner graphic matching student at laptop / Better Support. Higher Retention in ref
    user_banner = patches.FancyBboxPatch(
        (c5_x, 6.8), c5_w, 6.4,
        boxstyle="round,pad=0.0,rounding_size=0.6",
        facecolor="#F0FDF4", edgecolor="#86EFAC", linewidth=1.1, zorder=2
    )
    ax.add_patch(user_banner)
    draw_chart_icon(c5_x + 12.8, 10.0, s=1.1, color="#16A34A")
    ax.text(c5_x + 1.2, 11.4, "Better Precision.", fontsize=8.8, fontweight="black", color="#15803D", va="center", zorder=3)
    ax.text(c5_x + 1.2, 9.8, "Safer Social Feeds.", fontsize=8.8, fontweight="black", color="#15803D", va="center", zorder=3)
    ax.text(c5_x + 1.2, 8.2, "Offline Dockerized Stack", fontsize=7.2, color="#166534", va="center", zorder=3)

    # =========================================================================
    # 8. BOTTOM GUARDRAIL BANNER (Matching Green Pill Banner in Reference)
    # =========================================================================
    guard_box = patches.FancyBboxPatch(
        (1.5, 1.4), 97.0, 3.6,
        boxstyle="round,pad=0.0,rounding_size=1.2",
        facecolor=c_green_bg, edgecolor=c_green_border, linewidth=1.5, zorder=2
    )
    ax.add_patch(guard_box)

    # Custom vector checkmark circle (no font dependency)
    draw_check_circle(3.5, 3.2, 1.0, bg_color="#059669", check_color="#FFFFFF", lw=2.4, zorder=4)

    # Guardrail Text (Bold & formatted with highlighted keywords)
    ax.text(5.4, 3.2,
            "Guardrail: Decision threshold (t = 0.38) is calibrated strictly on the Validation Set to prevent target leakage.",
            fontsize=10.5, fontweight="bold", color="#065F46", va="center", ha="left", zorder=3)

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight", facecolor="#FFFFFF", edgecolor="none")
    plt.close()
    print(f"Enterprise architecture diagram successfully rendered at {OUTPUT_PATH}")

if __name__ == "__main__":
    create_enterprise_diagram()
