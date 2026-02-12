"""
Cascade Graph Visualizer v2
============================
Đọc tất cả file .pt và tạo HTML tương tác với vis.js
để xem cấu trúc đồ thị: node nào nối với node nào.

Chạy: python visualize_graphs.py
Output: data/cascade_visualization.html
"""

import torch
import os
import json
import glob

# --- CONFIG ---
INPUT_FOLDER = "data/processed_graphs"
OUTPUT_HTML = "data/cascade_visualization.html"

def load_all_graphs(folder):
    """Đọc tất cả file .pt và trích xuất thông tin."""
    files = sorted(glob.glob(os.path.join(folder, "*.pt")))
    summaries = []
    
    for filepath in files:
        try:
            data = torch.load(filepath, map_location='cpu', weights_only=False)
            
            post_id = getattr(data, 'post_id', os.path.basename(filepath).replace('.pt', ''))
            num_nodes = int(data.num_nodes) if hasattr(data, 'num_nodes') else int(data.x.shape[0])
            feat_dim = int(data.x.shape[1]) if data.x is not None else 0
            num_edges = int(data.edge_index.shape[1]) if data.edge_index is not None and data.edge_index.numel() > 0 else 0
            file_size_kb = round(os.path.getsize(filepath) / 1024, 1)
            
            # Trích xuất TOÀN BỘ danh sách cạnh (không giới hạn)
            edges = []
            if num_edges > 0:
                ei = data.edge_index
                for i in range(ei.shape[1]):
                    edges.append([int(ei[0][i]), int(ei[1][i])])
            
            summaries.append({
                "post_id": str(post_id),
                "num_nodes": num_nodes,
                "feat_dim": feat_dim,
                "num_edges": num_edges,
                "file_size_kb": file_size_kb,
                "edges": edges,
                "filename": os.path.basename(filepath)
            })
        except Exception as e:
            summaries.append({
                "post_id": os.path.basename(filepath).replace('.pt', ''),
                "num_nodes": 0,
                "feat_dim": 0,
                "num_edges": 0,
                "file_size_kb": round(os.path.getsize(filepath) / 1024, 1),
                "edges": [],
                "filename": os.path.basename(filepath),
                "error": str(e)
            })
    
    return summaries

def generate_html(summaries):
    """Tạo dashboard HTML tương tác với vis.js."""
    
    total = len(summaries)
    total_nodes = sum(s['num_nodes'] for s in summaries)
    total_edges = sum(s['num_edges'] for s in summaries)
    avg_nodes = round(total_nodes / total, 1) if total > 0 else 0
    max_nodes = max(s['num_nodes'] for s in summaries) if summaries else 0
    single_node = sum(1 for s in summaries if s['num_nodes'] == 1)
    multi_node = total - single_node
    
    size_buckets = {"1 node": 0, "2-5 nodes": 0, "6-10 nodes": 0, "11-20 nodes": 0, "21-50 nodes": 0, "50+ nodes": 0}
    for s in summaries:
        n = s['num_nodes']
        if n <= 1: size_buckets["1 node"] += 1
        elif n <= 5: size_buckets["2-5 nodes"] += 1
        elif n <= 10: size_buckets["6-10 nodes"] += 1
        elif n <= 20: size_buckets["11-20 nodes"] += 1
        elif n <= 50: size_buckets["21-50 nodes"] += 1
        else: size_buckets["50+ nodes"] += 1
    
    data_json = json.dumps(summaries, ensure_ascii=False)
    buckets_json = json.dumps(size_buckets)
    
    html = f"""<!DOCTYPE html>
<html lang="vi">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🔍 Cascade Graph Inspector v2</title>
    <script src="https://unpkg.com/vis-network/standalone/umd/vis-network.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{
            font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
            background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
            color: #e0e0e0;
            min-height: 100vh;
        }}
        .header {{
            background: rgba(255,255,255,0.05);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255,255,255,0.1);
            padding: 20px 30px;
            text-align: center;
        }}
        .header h1 {{
            font-size: 28px;
            background: linear-gradient(90deg, #00d2ff, #3a7bd5);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }}
        .header p {{ color: #999; margin-top: 5px; font-size: 14px; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
            gap: 15px;
            padding: 20px 30px;
            max-width: 1200px;
            margin: 0 auto;
        }}
        .stat-card {{
            background: rgba(255,255,255,0.06);
            border: 1px solid rgba(255,255,255,0.1);
            border-radius: 12px;
            padding: 16px;
            text-align: center;
            transition: transform 0.2s;
        }}
        .stat-card:hover {{ transform: translateY(-3px); }}
        .stat-card .value {{
            font-size: 30px; font-weight: 700;
            background: linear-gradient(90deg, #00d2ff, #7b68ee);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .stat-card .label {{ font-size: 11px; color: #888; margin-top: 4px; text-transform: uppercase; letter-spacing: 1px; }}
        
        .section {{ padding: 20px 30px; max-width: 1200px; margin: 0 auto; }}
        .section h2 {{ font-size: 20px; margin-bottom: 15px; color: #7b68ee; }}
        
        .distribution {{ display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 20px; }}
        .dist-bar {{
            background: rgba(123,104,238,0.15);
            border: 1px solid rgba(123,104,238,0.3);
            border-radius: 8px; padding: 10px 16px;
            text-align: center; min-width: 110px;
        }}
        .dist-bar .count {{ font-size: 22px; font-weight: 700; color: #7b68ee; }}
        .dist-bar .range {{ font-size: 11px; color: #888; margin-top: 3px; }}
        
        .controls {{ display: flex; gap: 12px; margin-bottom: 15px; flex-wrap: wrap; align-items: center; }}
        .controls input, .controls select {{
            background: rgba(255,255,255,0.08);
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 8px; padding: 10px 14px;
            color: #e0e0e0; font-size: 14px; outline: none;
        }}
        .controls input:focus, .controls select:focus {{ border-color: #7b68ee; }}
        .controls input {{ flex: 1; min-width: 200px; }}
        
        .table-wrapper {{ overflow-x: auto; border-radius: 12px; border: 1px solid rgba(255,255,255,0.1); }}
        table {{ width: 100%; border-collapse: collapse; }}
        th {{
            background: rgba(123,104,238,0.15); padding: 12px 16px;
            text-align: left; font-size: 11px; text-transform: uppercase;
            letter-spacing: 1px; color: #aaa; white-space: nowrap;
        }}
        td {{ padding: 10px 16px; border-top: 1px solid rgba(255,255,255,0.05); font-size: 13px; }}
        tr:hover {{ background: rgba(255,255,255,0.03); }}
        .clickable {{ cursor: pointer; color: #00d2ff; text-decoration: underline; }}
        
        .badge {{ display: inline-block; padding: 2px 8px; border-radius: 10px; font-size: 11px; font-weight: 600; }}
        .badge-single {{ background: rgba(255,107,107,0.2); color: #ff6b6b; }}
        .badge-small {{ background: rgba(255,171,0,0.2); color: #ffab00; }}
        .badge-medium {{ background: rgba(0,210,255,0.2); color: #00d2ff; }}
        .badge-large {{ background: rgba(46,213,115,0.2); color: #2ed573; }}
        
        /* ===== MODAL XEM ĐỒ THỊ ===== */
        .modal-overlay {{
            display: none;
            position: fixed; top: 0; left: 0; width: 100%; height: 100%;
            background: rgba(0,0,0,0.85);
            z-index: 1000;
            justify-content: center; align-items: center;
        }}
        .modal-overlay.open {{ display: flex; }}
        .modal {{
            background: #1a1a2e;
            border: 1px solid rgba(255,255,255,0.15);
            border-radius: 16px;
            width: 95vw; max-width: 1100px;
            height: 85vh;
            display: flex; flex-direction: column;
            overflow: hidden;
        }}
        .modal-header {{
            display: flex; justify-content: space-between; align-items: center;
            padding: 18px 24px;
            border-bottom: 1px solid rgba(255,255,255,0.1);
        }}
        .modal-header h3 {{
            font-size: 18px; color: #00d2ff;
        }}
        .modal-close {{
            background: rgba(255,255,255,0.1); border: none; color: #ccc;
            width: 34px; height: 34px; border-radius: 50%;
            font-size: 18px; cursor: pointer; transition: background 0.2s;
        }}
        .modal-close:hover {{ background: rgba(255,107,107,0.3); color: #fff; }}
        
        .modal-body {{
            display: flex; flex: 1; overflow: hidden;
        }}
        
        /* Panel trái: đồ thị vis.js */
        .graph-panel {{
            flex: 1; position: relative;
            border-right: 1px solid rgba(255,255,255,0.1);
        }}
        #visGraph {{
            width: 100%; height: 100%;
            background: #0d0d1a;
        }}
        
        /* Panel phải: thông tin + danh sách cạnh */
        .info-panel {{
            width: 320px; overflow-y: auto; padding: 20px;
        }}
        .info-row {{ margin-bottom: 8px; }}
        .info-row .lbl {{ color: #888; font-size: 12px; }}
        .info-row .val {{ color: #e0e0e0; font-size: 15px; font-weight: 600; }}
        
        .edge-section {{ margin-top: 20px; }}
        .edge-section h4 {{ color: #7b68ee; font-size: 14px; margin-bottom: 10px; }}
        .edge-list {{
            max-height: 300px; overflow-y: auto;
            background: rgba(255,255,255,0.03);
            border-radius: 8px; padding: 8px;
            font-family: 'Consolas', monospace; font-size: 12px;
        }}
        .edge-item {{
            padding: 4px 8px;
            border-bottom: 1px solid rgba(255,255,255,0.05);
            display: flex; align-items: center; gap: 8px;
        }}
        .edge-item:hover {{ background: rgba(123,104,238,0.1); }}
        .edge-arrow {{ color: #00d2ff; }}
        .node-highlight {{ color: #7b68ee; font-weight: 600; }}
        
        .degree-section {{ margin-top: 20px; }}
        .degree-section h4 {{ color: #7b68ee; font-size: 14px; margin-bottom: 10px; }}
        .degree-list {{
            max-height: 200px; overflow-y: auto;
            background: rgba(255,255,255,0.03);
            border-radius: 8px; padding: 8px;
            font-size: 12px;
        }}
        .degree-item {{
            padding: 3px 8px; display: flex; justify-content: space-between;
        }}
        .degree-bar {{
            height: 6px; background: #7b68ee; border-radius: 3px; margin-top: 2px;
        }}

        .legend {{
            margin-top: 16px; padding: 12px;
            background: rgba(255,255,255,0.03); border-radius: 8px;
        }}
        .legend h4 {{ color: #888; font-size: 12px; margin-bottom: 8px; text-transform: uppercase; }}
        .legend-item {{ display: flex; align-items: center; gap: 8px; margin-bottom: 4px; font-size: 12px; }}
        .legend-dot {{ width: 12px; height: 12px; border-radius: 50%; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🔍 Cascade Graph Inspector v2</h1>
        <p>Trực quan hóa cấu trúc {total} đồ thị cascade (.pt) — xem node nào liên kết với node nào</p>
    </div>
    
    <div class="stats-grid">
        <div class="stat-card"><div class="value">{total}</div><div class="label">Tổng đồ thị</div></div>
        <div class="stat-card"><div class="value">{total_nodes:,}</div><div class="label">Tổng node</div></div>
        <div class="stat-card"><div class="value">{total_edges:,}</div><div class="label">Tổng cạnh</div></div>
        <div class="stat-card"><div class="value">{avg_nodes}</div><div class="label">TB node/graph</div></div>
        <div class="stat-card"><div class="value">{max_nodes}</div><div class="label">Max nodes</div></div>
        <div class="stat-card"><div class="value">{multi_node}</div><div class="label">Có cascade</div></div>
    </div>
    
    <div class="section">
        <h2>📊 Phân bố kích thước</h2>
        <div class="distribution" id="distBars"></div>
    </div>
    
    <div class="section">
        <h2>📋 Danh sách đồ thị — bấm "👁️ Xem" để xem cấu trúc liên kết</h2>
        <div class="controls">
            <input type="text" id="searchInput" placeholder="🔍 Tìm theo Post ID..." oninput="filterTable()">
            <select id="sizeFilter" onchange="filterTable()">
                <option value="all">Tất cả kích thước</option>
                <option value="1">Chỉ 1 node</option>
                <option value="2-5">2-5 nodes</option>
                <option value="6-20">6-20 nodes</option>
                <option value="20+">20+ nodes</option>
            </select>
            <select id="sortSelect" onchange="sortTable()">
                <option value="nodes_desc">Nodes ↓</option>
                <option value="nodes_asc">Nodes ↑</option>
                <option value="edges_desc">Edges ↓</option>
                <option value="id_asc">ID A-Z</option>
                <option value="size_desc">File size ↓</option>
            </select>
        </div>
        <div class="table-wrapper">
            <table>
                <thead><tr>
                    <th>#</th><th>Post ID</th><th>Nodes</th><th>Edges</th>
                    <th>Feature Dim</th><th>File Size</th><th>Loại</th><th>Xem cấu trúc</th>
                </tr></thead>
                <tbody id="tableBody"></tbody>
            </table>
        </div>
        <p id="resultCount" style="margin-top:10px; color:#888; font-size:13px;"></p>
    </div>
    
    <!-- ===== MODAL XEM ĐỒ THỊ ===== -->
    <div class="modal-overlay" id="modalOverlay" onclick="if(event.target===this)closeModal()">
        <div class="modal">
            <div class="modal-header">
                <h3 id="modalTitle">📊 Chi tiết đồ thị</h3>
                <button class="modal-close" onclick="closeModal()">✕</button>
            </div>
            <div class="modal-body">
                <div class="graph-panel">
                    <div id="visGraph"></div>
                </div>
                <div class="info-panel">
                    <div class="info-row"><div class="lbl">Post ID</div><div class="val" id="mId">-</div></div>
                    <div class="info-row"><div class="lbl">Số node</div><div class="val" id="mNodes">-</div></div>
                    <div class="info-row"><div class="lbl">Số cạnh</div><div class="val" id="mEdges">-</div></div>
                    <div class="info-row"><div class="lbl">Feature Dim</div><div class="val" id="mDim">-</div></div>
                    <div class="info-row"><div class="lbl">File</div><div class="val" id="mFile">-</div></div>
                    <div class="info-row"><div class="lbl">Kích thước</div><div class="val" id="mSize">-</div></div>
                    
                    <div class="legend">
                        <h4>Chú thích</h4>
                        <div class="legend-item"><div class="legend-dot" style="background:#00d2ff;"></div> Node 0 = ROOT (bài viết gốc)</div>
                        <div class="legend-item"><div class="legend-dot" style="background:#7b68ee;"></div> Node 1+ = Comment (bình luận)</div>
                        <div class="legend-item"><div class="legend-dot" style="background:#2ed573;"></div> Mũi tên = hướng lan truyền (cha → con)</div>
                    </div>
                    
                    <div class="degree-section">
                        <h4>📈 Bậc (degree) của từng node</h4>
                        <div class="degree-list" id="degreeList"></div>
                    </div>
                    
                    <div class="edge-section">
                        <h4>🔗 Danh sách tất cả cạnh</h4>
                        <div class="edge-list" id="edgeList"></div>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div style="text-align:center; padding:30px; color:#555; font-size:12px;">
        Cascade Graph Inspector v2 | vis.js Interactive Viewer | Phase 2
    </div>

    <script>
    const ALL_DATA = {data_json};
    const BUCKETS = {buckets_json};
    let currentData = [...ALL_DATA];
    let visNetwork = null;
    
    // Phân bố
    (function() {{
        const c = document.getElementById('distBars');
        for (const [range, count] of Object.entries(BUCKETS)) {{
            c.innerHTML += '<div class="dist-bar"><div class="count">'+count+'</div><div class="range">'+range+'</div></div>';
        }}
    }})();
    
    function getBadge(n) {{
        if (n<=1) return '<span class="badge badge-single">Chỉ root</span>';
        if (n<=5) return '<span class="badge badge-small">Nhỏ</span>';
        if (n<=20) return '<span class="badge badge-medium">TB</span>';
        return '<span class="badge badge-large">Lớn</span>';
    }}
    
    function renderTable(data) {{
        const tb = document.getElementById('tableBody');
        tb.innerHTML = '';
        data.forEach((item, i) => {{
            const tr = document.createElement('tr');
            tr.innerHTML = '<td>'+(i+1)+'</td>'
                +'<td style="font-family:monospace;font-weight:600;">'+item.post_id+'</td>'
                +'<td>'+item.num_nodes+'</td>'
                +'<td>'+item.num_edges+'</td>'
                +'<td>'+item.feat_dim+'</td>'
                +'<td>'+item.file_size_kb+' KB</td>'
                +'<td>'+getBadge(item.num_nodes)+'</td>'
                +'<td><span class="clickable" onclick="openModal('+ALL_DATA.indexOf(item)+')">👁️ Xem</span></td>';
            tb.appendChild(tr);
        }});
        document.getElementById('resultCount').textContent = 'Hiển thị '+data.length+' / '+ALL_DATA.length+' đồ thị';
    }}
    
    function filterTable() {{
        const search = document.getElementById('searchInput').value.toLowerCase();
        const sf = document.getElementById('sizeFilter').value;
        currentData = ALL_DATA.filter(item => {{
            if (search && !item.post_id.toLowerCase().includes(search)) return false;
            const n = item.num_nodes;
            if (sf==='1' && n!==1) return false;
            if (sf==='2-5' && (n<2||n>5)) return false;
            if (sf==='6-20' && (n<6||n>20)) return false;
            if (sf==='20+' && n<20) return false;
            return true;
        }});
        sortTable();
    }}
    
    function sortTable() {{
        const s = document.getElementById('sortSelect').value;
        currentData.sort((a,b) => {{
            if (s==='nodes_desc') return b.num_nodes-a.num_nodes;
            if (s==='nodes_asc') return a.num_nodes-b.num_nodes;
            if (s==='edges_desc') return b.num_edges-a.num_edges;
            if (s==='id_asc') return a.post_id.localeCompare(b.post_id);
            if (s==='size_desc') return b.file_size_kb-a.file_size_kb;
            return 0;
        }});
        renderTable(currentData);
    }}
    
    function openModal(idx) {{
        const item = ALL_DATA[idx];
        document.getElementById('mId').textContent = item.post_id;
        document.getElementById('mNodes').textContent = item.num_nodes;
        document.getElementById('mEdges').textContent = item.num_edges;
        document.getElementById('mDim').textContent = item.feat_dim;
        document.getElementById('mFile').textContent = item.filename;
        document.getElementById('mSize').textContent = item.file_size_kb + ' KB';
        document.getElementById('modalTitle').textContent = '� ' + item.post_id;
        document.getElementById('modalOverlay').classList.add('open');
        
        // Tính bậc cho từng node
        const degrees = {{}};
        for (let i = 0; i < item.num_nodes; i++) degrees[i] = 0;
        for (const [src, dst] of item.edges) {{
            degrees[src] = (degrees[src]||0) + 1;
            degrees[dst] = (degrees[dst]||0) + 1;
        }}
        const maxDeg = Math.max(1, ...Object.values(degrees));
        
        // Render degree list
        const dl = document.getElementById('degreeList');
        dl.innerHTML = '';
        const sortedNodes = Object.entries(degrees).sort((a,b) => b[1]-a[1]);
        for (const [nodeId, deg] of sortedNodes) {{
            const label = parseInt(nodeId) === 0 ? 'Node 0 (ROOT)' : 'Node ' + nodeId;
            dl.innerHTML += '<div class="degree-item"><span>'+label+'</span><span style="color:#7b68ee;font-weight:600;">'+deg+'</span></div>'
                + '<div class="degree-bar" style="width:'+Math.round(deg/maxDeg*100)+'%"></div>';
        }}
        
        // Render edge list
        const el = document.getElementById('edgeList');
        el.innerHTML = '';
        if (item.edges.length === 0) {{
            el.innerHTML = '<div style="color:#666;padding:10px;text-align:center;">Không có cạnh nào (chỉ 1 node)</div>';
        }} else {{
            for (const [src, dst] of item.edges) {{
                const srcLabel = src === 0 ? 'ROOT' : 'Node '+src;
                const dstLabel = dst === 0 ? 'ROOT' : 'Node '+dst;
                el.innerHTML += '<div class="edge-item">'
                    + '<span class="node-highlight">'+srcLabel+'</span>'
                    + '<span class="edge-arrow">→</span>'
                    + '<span class="node-highlight">'+dstLabel+'</span>'
                    + '</div>';
            }}
        }}
        
        // Vẽ đồ thị bằng vis.js
        renderVisGraph(item, degrees);
    }}
    
    function renderVisGraph(item, degrees) {{
        const container = document.getElementById('visGraph');
        
        if (item.num_nodes <= 1) {{
            container.innerHTML = '<div style="display:flex;justify-content:center;align-items:center;height:100%;color:#888;">📌 Chỉ có 1 node (bài gốc, không có comment)</div>';
            return;
        }}
        container.innerHTML = '';
        
        // Tạo nodes cho vis.js
        const nodes = [];
        for (let i = 0; i < item.num_nodes; i++) {{
            const deg = degrees[i] || 0;
            const size = i === 0 ? 25 : Math.max(8, 8 + deg * 2);
            nodes.push({{
                id: i,
                label: i === 0 ? 'ROOT' : '' + i,
                color: {{
                    background: i === 0 ? '#00d2ff' : '#7b68ee',
                    border: i === 0 ? '#0099cc' : '#5a4fcf',
                    highlight: {{ background: '#ff6b6b', border: '#cc4444' }},
                    hover: {{ background: '#ffab00', border: '#cc8800' }}
                }},
                size: size,
                font: {{
                    color: '#ffffff',
                    size: i === 0 ? 14 : 10,
                    face: 'Segoe UI'
                }},
                title: (i===0 ? 'ROOT (Bài gốc)' : 'Node '+i+' (Comment)') + '\\nBậc: ' + deg,
                shape: i === 0 ? 'dot' : 'dot'
            }});
        }}
        
        // Tạo edges cho vis.js
        const edges = [];
        for (let i = 0; i < item.edges.length; i++) {{
            const [src, dst] = item.edges[i];
            edges.push({{
                from: src,
                to: dst,
                arrows: 'to',
                color: {{ color: 'rgba(46,213,115,0.5)', highlight: '#2ed573', hover: '#ffab00' }},
                width: 1.5,
                smooth: {{ type: 'cubicBezier', roundness: 0.3 }}
            }});
        }}
        
        const data = {{
            nodes: new vis.DataSet(nodes),
            edges: new vis.DataSet(edges)
        }};
        
        const options = {{
            physics: {{
                enabled: true,
                solver: 'forceAtlas2Based',
                forceAtlas2Based: {{
                    gravitationalConstant: -40,
                    centralGravity: 0.005,
                    springLength: 100,
                    springConstant: 0.04,
                    damping: 0.5
                }},
                stabilization: {{ iterations: 150, fit: true }}
            }},
            interaction: {{
                hover: true,
                tooltipDelay: 100,
                zoomView: true,
                dragView: true,
                dragNodes: true,
                navigationButtons: false,
                keyboard: false
            }},
            layout: {{
                improvedLayout: item.num_nodes < 100
            }}
        }};
        
        if (visNetwork) {{
            visNetwork.destroy();
        }}
        visNetwork = new vis.Network(container, data, options);
    }}
    
    function closeModal() {{
        document.getElementById('modalOverlay').classList.remove('open');
        if (visNetwork) {{
            visNetwork.destroy();
            visNetwork = null;
        }}
    }}
    
    // Đóng modal bằng phím Escape
    document.addEventListener('keydown', function(e) {{
        if (e.key === 'Escape') closeModal();
    }});
    
    // Render ban đầu
    sortTable();
    </script>
</body>
</html>"""
    return html

def main():
    print("📊 Cascade Graph Inspector v2")
    print("="*40)
    
    if not os.path.exists(INPUT_FOLDER):
        print(f"❌ Không tìm thấy folder: {INPUT_FOLDER}")
        return
    
    print(f"📂 Đang đọc file từ {INPUT_FOLDER}...")
    summaries = load_all_graphs(INPUT_FOLDER)
    print(f"✅ Đã đọc {len(summaries)} file .pt")
    
    total_nodes = sum(s['num_nodes'] for s in summaries)
    total_edges = sum(s['num_edges'] for s in summaries)
    single = sum(1 for s in summaries if s['num_nodes'] == 1)
    
    print(f"\n📈 Thống kê nhanh:")
    print(f"   Tổng đồ thị: {len(summaries)}")
    print(f"   Tổng node:   {total_nodes:,}")
    print(f"   Tổng cạnh:   {total_edges:,}")
    print(f"   Có cascade:  {len(summaries) - single}")
    print(f"   Chỉ root:    {single}")
    
    print(f"\n🎨 Đang tạo HTML...")
    html = generate_html(summaries)
    
    os.makedirs(os.path.dirname(OUTPUT_HTML), exist_ok=True)
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Đã tạo xong: {OUTPUT_HTML}")
    print(f"👉 Mở file này trong trình duyệt để xem!")

if __name__ == "__main__":
    main()
