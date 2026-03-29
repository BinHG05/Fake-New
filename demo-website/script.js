/* ═══════════════════════════════════════════════
   Fake News Detection Dashboard — JavaScript
   ═══════════════════════════════════════════════ */

// ── State ──
let currentTaskId = null;
let currentEventSource = null;

// ════════════════════════════════════════════════
// Particles Background
// ════════════════════════════════════════════════

(function initParticles() {
    const canvas = document.getElementById('particles');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let particles = [];
    const COUNT = 60;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }
    window.addEventListener('resize', resize);
    resize();

    for (let i = 0; i < COUNT; i++) {
        particles.push({
            x: Math.random() * canvas.width,
            y: Math.random() * canvas.height,
            vx: (Math.random() - 0.5) * 0.3,
            vy: (Math.random() - 0.5) * 0.3,
            r: Math.random() * 2 + 0.5,
            alpha: Math.random() * 0.5 + 0.1,
        });
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach(p => {
            p.x += p.vx; p.y += p.vy;
            if (p.x < 0) p.x = canvas.width;
            if (p.x > canvas.width) p.x = 0;
            if (p.y < 0) p.y = canvas.height;
            if (p.y > canvas.height) p.y = 0;

            ctx.beginPath();
            ctx.arc(p.x, p.y, p.r, 0, Math.PI * 2);
            ctx.fillStyle = `rgba(129,140,248,${p.alpha})`;
            ctx.fill();
        });

        // Draw connections
        for (let i = 0; i < particles.length; i++) {
            for (let j = i + 1; j < particles.length; j++) {
                const dx = particles[i].x - particles[j].x;
                const dy = particles[i].y - particles[j].y;
                const dist = Math.sqrt(dx * dx + dy * dy);
                if (dist < 120) {
                    ctx.beginPath();
                    ctx.moveTo(particles[i].x, particles[i].y);
                    ctx.lineTo(particles[j].x, particles[j].y);
                    ctx.strokeStyle = `rgba(129,140,248,${0.06 * (1 - dist / 120)})`;
                    ctx.stroke();
                }
            }
        }
        requestAnimationFrame(draw);
    }
    draw();
})();


// ════════════════════════════════════════════════
// Navigation
// ════════════════════════════════════════════════

document.querySelectorAll('.nav-link').forEach(link => {
    link.addEventListener('click', function () {
        document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
        this.classList.add('active');
    });
});

// Intersection observer for active nav
const sections = document.querySelectorAll('.section');
const observer = new IntersectionObserver(entries => {
    entries.forEach(entry => {
        if (entry.isIntersecting) {
            const id = entry.target.id;
            document.querySelectorAll('.nav-link').forEach(l => {
                l.classList.toggle('active', l.getAttribute('href') === `#${id}`);
            });
        }
    });
}, { threshold: 0.3 });
sections.forEach(s => observer.observe(s));


// ════════════════════════════════════════════════
// API Helpers
// ════════════════════════════════════════════════

async function api(url, method = 'GET', body = null) {
    const opts = { method, headers: { 'Content-Type': 'application/json' } };
    if (body) opts.body = JSON.stringify(body);
    const res = await fetch(url, opts);
    return res.json();
}

function setNavStatus(text, running = false) {
    const dot = document.querySelector('.status-dot');
    const txt = document.querySelector('.status-text');
    dot.className = `status-dot ${running ? 'running' : 'idle'}`;
    txt.textContent = text;
}


// ════════════════════════════════════════════════
// Log Streaming (SSE)
// ════════════════════════════════════════════════

function streamLog(taskId, logContainerId, onDone = null) {
    const container = document.getElementById(logContainerId);
    container.classList.remove('hidden');
    container.classList.add('active');
    const pre = container.querySelector('.log-content');
    pre.textContent = '';

    currentTaskId = taskId;
    if (currentEventSource) currentEventSource.close();

    const es = new EventSource(`/api/stream/${taskId}`);
    currentEventSource = es;

    es.onmessage = (event) => {
        const data = JSON.parse(event.data);
        pre.textContent += data.line + '\n';
        pre.scrollTop = pre.scrollHeight;

        if (data.done) {
            es.close();
            currentEventSource = null;
            currentTaskId = null;
            setNavStatus('Sẵn sàng', false);
            loadDataStats();
            loadRunHistory();
            renderCharts();
            loadResultsOverview();
            if (onDone) onDone(data);
        }
    };

    es.onerror = () => {
        es.close();
        currentEventSource = null;
        currentTaskId = null;
        setNavStatus('Sẵn sàng', false);
        if (onDone) onDone({ done: true });
    };
}

function clearLog(logId) {
    const container = document.getElementById(logId);
    container.classList.add('hidden');
    const pre = container.querySelector('.log-content');
    pre.textContent = '';
}


// ════════════════════════════════════════════════
// Data Pipeline Actions
// ════════════════════════════════════════════════

async function startCrawl() {
    const limit = parseInt(document.getElementById('crawlLimit').value) || 25;
    const imagesOnly = document.getElementById('crawlImagesOnly').checked;

    setNavStatus('Đang cào dữ liệu...', true);
    const res = await api('/api/crawl/start', 'POST', { limit, images_only: imagesOnly });
    streamLog(res.task_id, 'crawlLog');
}

async function startEnrich() {
    setNavStatus('Đang enrich data...', true);
    const res = await api('/api/enrich/start', 'POST');
    streamLog(res.task_id, 'enrichLog');
}

async function startBuildGraphs() {
    setNavStatus('Đang build graphs...', true);
    const res = await api('/api/build-graphs/start', 'POST');
    streamLog(res.task_id, 'graphLog');
}

async function startVisualize() {
    setNavStatus('Đang tạo visualization...', true);
    const res = await api('/api/visualize/start', 'POST');
    streamLog(res.task_id, 'vizLog', () => {
        document.getElementById('viewVizBtn').style.display = 'inline-flex';
    });
}

function viewVisualization() {
    window.open('/api/visualize/view', '_blank');
}


// New pipeline actions

function togglePipelineManual() {
    const mode = document.getElementById('pipelineMode').value;
    document.getElementById('pipelineManualOpts').classList.toggle('hidden', mode !== 'manual');
}

async function startRedditPipeline() {
    const mode = document.getElementById('pipelineMode').value;
    const body = { mode };
    if (mode === 'manual') {
        body.start = parseInt(document.getElementById('pipelineStart').value) || 0;
        body.count = parseInt(document.getElementById('pipelineCount').value) || 50;
    }
    setNavStatus('Reddit Pipeline...', true);
    const res = await api('/api/reddit-pipeline/start', 'POST', body);
    streamLog(res.task_id, 'redditPipeLog');
}

async function startBatchPipeline() {
    const start = parseInt(document.getElementById('batchStart').value) || 0;
    const count = parseInt(document.getElementById('batchCount').value) || 200;
    setNavStatus('Fakeddit Batch...', true);
    const res = await api('/api/batch-pipeline/start', 'POST', { start, count });
    streamLog(res.task_id, 'batchPipeLog');
}

async function startConvertLS() {
    const input = document.getElementById('lsInputFile').value;
    if (!input) {
        alert('Vui lòng chọn file export!');
        return;
    }
    const mergeMaster = document.getElementById('lsMergeMaster').checked;
    setNavStatus('Convert LS...', true);
    const res = await api('/api/convert-ls/start', 'POST', { input, merge_master: mergeMaster });
    streamLog(res.task_id, 'convertLSLog');
}

async function loadLSExports() {
    try {
        const files = await api('/api/ls-exports');
        const sel = document.getElementById('lsInputFile');
        sel.innerHTML = '';
        if (files.length === 0) {
            sel.innerHTML = '<option value="">Không tìm thấy file export</option>';
        } else {
            files.forEach(f => {
                const opt = document.createElement('option');
                opt.value = f.path;
                opt.textContent = `${f.path} (${f.size_kb} KB)`;
                sel.appendChild(opt);
            });
        }
    } catch (e) {
        console.warn('Could not load LS exports:', e);
    }
}


// ════════════════════════════════════════════════
// Auto-Labeling
// ════════════════════════════════════════════════

async function startAutoLabel() {
    const input = document.getElementById('autoLabelInput').value;
    if (!input) {
        alert('Vui lòng chọn file JSONL!');
        return;
    }
    const method = document.getElementById('autoLabelMethod').value;
    const mode = document.getElementById('autoLabelMode').value;
    const threshold = parseFloat(document.getElementById('autoLabelThreshold').value) || 0.3;
    const limitVal = document.getElementById('autoLabelLimit').value;
    const limit = limitVal ? parseInt(limitVal) : null;
    const lsPredictions = document.getElementById('autoLabelLS').checked;

    const methodLabels = { clip: 'CLIP', text: 'Text-only', groq: 'Groq LLM' };
    setNavStatus(`🤖 Auto-labeling (${methodLabels[method] || method})...`, true);
    const res = await api('/api/auto-label/start', 'POST', {
        input, method, mode, threshold, limit, ls_predictions: lsPredictions
    });
    streamLog(res.task_id, 'autoLabelLog');
}

async function loadAutoLabelFiles() {
    try {
        const files = await api('/api/auto-label/files');
        const sel = document.getElementById('autoLabelInput');
        sel.innerHTML = '';
        if (files.length === 0) {
            sel.innerHTML = '<option value="">Không tìm thấy file JSONL</option>';
        } else {
            files.forEach(f => {
                const opt = document.createElement('option');
                opt.value = f.path;
                opt.textContent = `${f.path} (${f.lines} dòng, ${f.size_kb} KB)`;
                sel.appendChild(opt);
            });
        }
    } catch (e) {
        console.warn('Could not load auto-label files:', e);
    }
}


// ════════════════════════════════════════════════
// Training
// ════════════════════════════════════════════════

function updateTrainingForm() {
    const type = document.getElementById('modelType').value;

    // Show/hide model-specific fields
    document.getElementById('gnnTypeGroup').classList.toggle('hidden', type !== 'gnn');
    document.getElementById('fusionTypeGroup').classList.toggle('hidden', type !== 'multimodal');
    document.getElementById('fusionDimGroup').classList.toggle('hidden', type !== 'multimodal');
    document.getElementById('batchSizeGroup').classList.toggle('hidden', type === 'gnn');

    // Update default LR
    const lrInput = document.getElementById('trainLR');
    if (type === 'image' || type === 'gnn') {
        lrInput.value = '0.001';
    } else {
        lrInput.value = '0.0002';
    }
}

async function startTraining() {
    const type = document.getElementById('modelType').value;
    const body = {
        model_type: type,
        epochs: parseInt(document.getElementById('trainEpochs').value),
        lr: parseFloat(document.getElementById('trainLR').value),
        batch_size: parseInt(document.getElementById('trainBatch').value),
        patience: parseInt(document.getElementById('trainPatience').value),
    };

    if (type === 'gnn') {
        body.gnn_type = document.getElementById('gnnType').value;
    }
    if (type === 'multimodal') {
        body.fusion_type = document.getElementById('fusionType').value;
        body.fusion_dim = parseInt(document.getElementById('fusionDim').value);
    }

    document.getElementById('trainBtn').disabled = true;
    document.getElementById('stopTrainBtn').classList.remove('hidden');
    document.getElementById('trainingStatus').classList.remove('hidden');
    document.getElementById('progressText').textContent = `Training ${type}...`;
    document.getElementById('progressFill').style.width = '10%';

    setNavStatus(`Training ${type}...`, true);
    const res = await api('/api/train/start', 'POST', body);

    // Animate progress bar
    let progress = 10;
    const progressInterval = setInterval(() => {
        if (progress < 90) {
            progress += Math.random() * 2;
            document.getElementById('progressFill').style.width = progress + '%';
        }
    }, 1000);

    streamLog(res.task_id, 'trainLog', () => {
        clearInterval(progressInterval);
        document.getElementById('progressFill').style.width = '100%';
        document.getElementById('progressText').textContent = 'Hoàn thành!';
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopTrainBtn').classList.add('hidden');
        setTimeout(() => {
            document.getElementById('trainingStatus').classList.add('hidden');
        }, 3000);
    });
}

async function stopCurrentTask() {
    if (currentTaskId) {
        await api(`/api/task/${currentTaskId}/stop`, 'POST');
        setNavStatus('Đã dừng', false);
        document.getElementById('trainBtn').disabled = false;
        document.getElementById('stopTrainBtn').classList.add('hidden');
        document.getElementById('trainingStatus').classList.add('hidden');
    }
}


// ════════════════════════════════════════════════
// Charts (Canvas-based)
// ════════════════════════════════════════════════

function drawBarChart(canvasId, labels, values, colors, maxVal = null) {
    const canvas = document.getElementById(canvasId);
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const W = canvas.width;
    const H = canvas.height;
    const padding = { top: 30, bottom: 60, left: 60, right: 20 };
    const chartW = W - padding.left - padding.right;
    const chartH = H - padding.top - padding.bottom;

    ctx.clearRect(0, 0, W, H);

    if (!maxVal) maxVal = Math.max(...values) * 1.25;
    const barW = chartW / labels.length * 0.6;
    const gap = chartW / labels.length;

    // Grid lines
    ctx.strokeStyle = 'rgba(148,163,184,0.1)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 5; i++) {
        const y = padding.top + chartH - (chartH * i / 5);
        ctx.beginPath();
        ctx.moveTo(padding.left, y);
        ctx.lineTo(W - padding.right, y);
        ctx.stroke();
        // Label
        ctx.fillStyle = '#64748b';
        ctx.font = '11px Inter';
        ctx.textAlign = 'right';
        ctx.fillText((maxVal * i / 5).toFixed(2), padding.left - 8, y + 4);
    }

    // Bars
    labels.forEach((label, i) => {
        const x = padding.left + i * gap + (gap - barW) / 2;
        const barH = (values[i] / maxVal) * chartH;
        const y = padding.top + chartH - barH;

        // Bar gradient
        const grad = ctx.createLinearGradient(x, y, x, y + barH);
        grad.addColorStop(0, colors[i] || '#818cf8');
        grad.addColorStop(1, colors[i] ? colors[i] + '88' : '#818cf888');
        ctx.fillStyle = grad;

        // Rounded rect
        const radius = 4;
        ctx.beginPath();
        ctx.moveTo(x + radius, y);
        ctx.lineTo(x + barW - radius, y);
        ctx.quadraticCurveTo(x + barW, y, x + barW, y + radius);
        ctx.lineTo(x + barW, y + barH);
        ctx.lineTo(x, y + barH);
        ctx.lineTo(x, y + radius);
        ctx.quadraticCurveTo(x, y, x + radius, y);
        ctx.fill();

        // Value on top
        ctx.fillStyle = '#e2e8f0';
        ctx.font = '600 12px Inter';
        ctx.textAlign = 'center';
        ctx.fillText(values[i].toFixed(3), x + barW / 2, y - 8);

        // Label below
        ctx.fillStyle = '#94a3b8';
        ctx.font = '11px Inter';
        ctx.save();
        ctx.translate(x + barW / 2, padding.top + chartH + 12);
        ctx.rotate(-0.3);
        ctx.fillText(label, 0, 0);
        ctx.restore();
    });
}

async function renderCharts() {
    try {
        const data = await api('/api/metrics/best');
        const colors = ['#818cf8', '#34d399', '#f87171', '#fbbf24', '#a78bfa'];

        drawBarChart('chartF1', data.labels, (data.binF1Values || []).map(v => (v || 0) * 100), colors, 100);
        drawBarChart('chartAcc', data.labels, data.binAccValues, colors, 100);
    } catch (e) {
        console.warn('Could not load chart metrics:', e);
    }
}

function formatPct(value) {
    if (value === undefined || value === null || Number.isNaN(value)) return '-';
    return `${Number(value).toFixed(2)}%`;
}

function formatPp(value) {
    if (value === undefined || value === null || Number.isNaN(value)) return '-';
    return `${value >= 0 ? '+' : ''}${Number(value).toFixed(2)} pp`;
}

function renderSummaryMarkdown(markdown) {
    if (!markdown) return 'Chưa có summary markdown.';
    return escapeHtml(markdown)
        .replace(/^# (.+)$/gm, '<h4>$1</h4>')
        .replace(/^\- (.+)$/gm, '<div class="summary-line">• $1</div>')
        .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
}

function renderResultFigures(figures) {
    const grid = document.getElementById('resultsFigureGrid');
    if (!grid) return;

    if (!figures || figures.length === 0) {
        grid.innerHTML = '<div class="empty-state">Chưa có hình kết quả.</div>';
        return;
    }

    grid.innerHTML = figures.map(fig => `
        <a class="result-figure-card" href="${fig.url}" target="_blank" rel="noopener noreferrer">
            <img src="${fig.url}" alt="${fig.title}" loading="lazy">
            <div class="result-figure-meta">
                <div class="result-figure-title">${fig.title}</div>
                <div class="result-figure-link">Mở hình gốc</div>
            </div>
        </a>
    `).join('');
}

async function loadResultsOverview() {
    try {
        const data = await api('/api/results/overview');
        const summary = data.summary || {};
        const best = summary.best_baseline || {};
        const full = summary.full_model || {};
        const delta = summary.delta_vs_baseline || {};

        document.getElementById('bestBaselineAcc').textContent = formatPct(best.binary_accuracy_pct);
        document.getElementById('bestBaselineF1').textContent = formatPct(best.binary_f1_pct);
        document.getElementById('fullModelAcc').textContent = formatPct(full.binary_accuracy_pct);
        document.getElementById('deltaAcc').textContent = formatPp(delta.accuracy_pp);

        const summaryBox = document.getElementById('resultsSummaryText');
        if (summaryBox) {
            summaryBox.innerHTML = renderSummaryMarkdown(data.summary_markdown || '');
        }

        const legacyWrap = document.querySelector('#results .results-table');
        if (legacyWrap && data.summary_exists) {
            legacyWrap.closest('.results-table-wrap').style.display = 'none';
        }

        renderResultFigures(data.figures || []);
    } catch (e) {
        console.warn('Could not load experiment results:', e);
    }
}


// ════════════════════════════════════════════════
// Data Stats
// ════════════════════════════════════════════════

async function loadDataStats() {
    try {
        const stats = await api('/api/data/stats');
        document.getElementById('statRaw').textContent = stats.raw_reddit_count?.toLocaleString() || '0';
        document.getElementById('statEnriched').textContent = stats.enriched_count?.toLocaleString() || '0';
        document.getElementById('statGraphs').textContent = stats.graph_count?.toLocaleString() || '0';
        document.getElementById('statModels').textContent = stats.checkpoints?.length?.toString() || '0';

        // Show view viz button if exists
        if (stats.visualization_exists) {
            document.getElementById('viewVizBtn').style.display = 'inline-flex';
        }
    } catch (e) {
        console.warn('Could not load stats:', e);
    }
}


// ════════════════════════════════════════════════
// Run History
// ════════════════════════════════════════════════

const TYPE_LABELS = {
    crawl: '🕷️ Crawl',
    enrich: '💬 Enrich',
    build_graphs: '🔗 Build Graphs',
    visualize: '👁️ Visualize',
    reddit_pipeline: '🔄 Reddit Pipeline',
    batch_pipeline: '📦 Fakeddit Batch',
    convert_ls: '🏷️ LS Convert',
    auto_label: '🤖 Auto-Label',
    train_text: '📝 Train Text',
    train_image: '🖼️ Train Image',
    train_fusion: '🔀 Train Fusion',
    train_gnn: '🔗 Train GNN',
    train_multimodal: '🧠 Train Multimodal',
};

async function loadRunHistory() {
    try {
        const runs = await api('/api/runs');
        const filter = document.getElementById('historyFilter').value;
        const tbody = document.getElementById('historyBody');

        let filtered = runs;
        if (filter === 'train') {
            filtered = runs.filter(r => r.task_type.startsWith('train'));
        } else if (filter !== 'all') {
            filtered = runs.filter(r => r.task_type === filter);
        }

        if (filtered.length === 0) {
            tbody.innerHTML = '<tr><td colspan="8" class="empty-state">Chưa có lần chạy nào</td></tr>';
            return;
        }

        tbody.innerHTML = filtered.map(run => {
            const params = tryParse(run.params);
            const metrics = tryParse(run.metrics);
            const statusClass = `badge-${run.status}`;
            const typeLabel = TYPE_LABELS[run.task_type] || run.task_type;
            const startTime = formatTime(run.started_at);
            const endTime = run.finished_at ? formatTime(run.finished_at) : '—';
            const paramsStr = Object.entries(params).slice(0, 3).map(([k, v]) => `${k}=${v}`).join(', ') || '—';
            const metricsStr = formatMetrics(metrics);

            return `<tr>
                <td><code style="font-size:0.72rem;color:var(--accent)">${run.id}</code></td>
                <td><span class="type-badge">${typeLabel}</span></td>
                <td><span class="badge ${statusClass}">${run.status}</span></td>
                <td style="white-space:nowrap">${startTime}</td>
                <td style="white-space:nowrap">${endTime}</td>
                <td class="params-cell">${paramsStr}</td>
                <td class="metrics-cell">${metricsStr}</td>
                <td>
                    <button class="btn btn-sm btn-secondary" onclick="viewRunDetail('${run.id}')">📋</button>
                    <button class="btn btn-sm btn-danger" onclick="deleteRun('${run.id}')" style="margin-left:4px">🗑</button>
                </td>
            </tr>`;
        }).join('');
    } catch (e) {
        console.warn('Could not load history:', e);
    }
}

function tryParse(val) {
    if (!val) return {};
    if (typeof val === 'object') return val;
    try { return JSON.parse(val); } catch { return {}; }
}

function formatTime(iso) {
    if (!iso) return '—';
    const d = new Date(iso);
    return d.toLocaleString('vi-VN', { hour: '2-digit', minute: '2-digit', day: '2-digit', month: '2-digit' });
}

function formatMetrics(metrics) {
    if (!metrics || Object.keys(metrics).length === 0) return '—';
    const parts = [];
    if (metrics.test_f1_macro_6 !== undefined) parts.push(`F1=${metrics.test_f1_macro_6.toFixed(3)}`);
    if (metrics.test_acc_bin !== undefined) parts.push(`BinAcc=${(metrics.test_acc_bin * 100).toFixed(1)}%`);
    if (metrics.new_items !== undefined) parts.push(`New=${metrics.new_items}`);
    if (metrics.graphs_created !== undefined) parts.push(`Created=${metrics.graphs_created}`);
    return parts.join(', ') || '—';
}

async function viewRunDetail(runId) {
    const run = await api(`/api/runs/${runId}`);
    document.getElementById('modalTitle').textContent = `Run ${run.id} — ${TYPE_LABELS[run.task_type] || run.task_type}`;

    const params = tryParse(run.params);
    const metrics = tryParse(run.metrics);

    let html = `<strong>Status:</strong> ${run.status}\n`;
    html += `<strong>Started:</strong> ${run.started_at}\n`;
    html += `<strong>Finished:</strong> ${run.finished_at || '—'}\n\n`;
    html += `<strong>Params:</strong>\n${JSON.stringify(params, null, 2)}\n\n`;
    html += `<strong>Metrics:</strong>\n${JSON.stringify(metrics, null, 2)}\n\n`;
    html += `<strong>Log:</strong>\n${run.log || '(empty)'}`;

    document.getElementById('modalBody').textContent = '';
    document.getElementById('modalBody').innerHTML = `<pre style="white-space:pre-wrap;word-break:break-all">${escapeHtml(html)}</pre>`;
    document.getElementById('runModal').classList.remove('hidden');
}

function escapeHtml(str) {
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

function closeModal() {
    document.getElementById('runModal').classList.add('hidden');
}

async function deleteRun(runId) {
    if (confirm('Xóa lần chạy này?')) {
        await api(`/api/runs/${runId}`, 'DELETE');
        loadRunHistory();
    }
}

// Close modal on outside click
document.getElementById('runModal')?.addEventListener('click', function (e) {
    if (e.target === this) closeModal();
});


// ════════════════════════════════════════════════
// Init
// ════════════════════════════════════════════════

window.addEventListener('DOMContentLoaded', () => {
    const mergeCheckbox = document.getElementById('lsMergeMaster');
    if (mergeCheckbox) {
        mergeCheckbox.checked = true;
        mergeCheckbox.disabled = true;
    }
    loadDataStats();
    loadRunHistory();
    renderCharts();
    loadResultsOverview();
    loadLSExports();
    loadAutoLabelFiles();
});
