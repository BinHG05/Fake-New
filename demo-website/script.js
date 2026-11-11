let currentTaskId = null;
let currentEventSource = null;
let presentationModeEnabled = false;

const DEMO_URLS = {
    science_a: {
        label: "Science A",
        url: "https://www.reddit.com/r/science/comments/1stxb1d/voting_is_linked_to_living_longer_among_older/",
        note: "Use this as the normal research-news example."
    },
    science_b: {
        label: "Science B",
        url: "https://www.reddit.com/r/science/comments/1su8s27/new_findings_demonstrate_that_a_slight_daily/",
        note: "Backup science example if the first URL becomes unstable."
    },
    onion_a: {
        label: "TheOnion A",
        url: "https://www.reddit.com/r/TheOnion/comments/1sti554/trump_conducts_marathon_reading_of_arbys_menu_in/",
        note: "Use this as the satire or fake-style example."
    },
    onion_b: {
        label: "TheOnion B",
        url: "https://www.reddit.com/r/TheOnion/comments/1sti55c/study_finds_gen_z_drinking_fewer_flagons_of_mead/",
        note: "Second satire example for backup."
    }
};

(function initParticles() {
    const canvas = document.getElementById("particles");
    if (!canvas) return;

    const ctx = canvas.getContext("2d");
    const particles = Array.from({ length: 48 }, () => ({
        x: Math.random(),
        y: Math.random(),
        r: Math.random() * 2.2 + 0.6,
        vx: (Math.random() - 0.5) * 0.0008,
        vy: (Math.random() - 0.5) * 0.0008,
    }));

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        particles.forEach((particle) => {
            particle.x += particle.vx;
            particle.y += particle.vy;
            if (particle.x < 0 || particle.x > 1) particle.vx *= -1;
            if (particle.y < 0 || particle.y > 1) particle.vy *= -1;

            ctx.beginPath();
            ctx.arc(particle.x * canvas.width, particle.y * canvas.height, particle.r, 0, Math.PI * 2);
            ctx.fillStyle = "rgba(85, 212, 170, 0.22)";
            ctx.fill();
        });

        requestAnimationFrame(draw);
    }

    window.addEventListener("resize", resize);
    resize();
    draw();
})();

document.querySelectorAll(".nav-link").forEach((link) => {
    link.addEventListener("click", function () {
        document.querySelectorAll(".nav-link").forEach((item) => item.classList.remove("active"));
        this.classList.add("active");
    });
});

const sectionObserver = new IntersectionObserver((entries) => {
    entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const activeId = entry.target.id;
        document.querySelectorAll(".nav-link").forEach((link) => {
            link.classList.toggle("active", link.getAttribute("href") === `#${activeId}`);
        });
    });
}, { threshold: 0.35 });

document.querySelectorAll(".section").forEach((section) => sectionObserver.observe(section));

async function api(url, method = "GET", body = null) {
    const options = { method, headers: { "Content-Type": "application/json" } };
    if (body) options.body = JSON.stringify(body);
    const response = await fetch(url, options);
    const data = await response.json();
    if (!response.ok) {
        throw new Error(data.error || `Request failed: ${response.status}`);
    }
    return data;
}

function setNavStatus(text, running = false) {
    const dot = document.querySelector(".status-dot");
    const label = document.querySelector(".status-text");
    if (!dot || !label) return;
    dot.className = `status-dot ${running ? "running" : "idle"}`;
    label.textContent = text;
}

function formatPercent(value) {
    if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
    return `${Number(value).toFixed(2)}%`;
}

function formatProbability(value) {
    if (value === undefined || value === null || Number.isNaN(Number(value))) return "--";
    return `${(Number(value) * 100).toFixed(2)}%`;
}

function getConfidenceTier(confidence, prediction) {
    const c = Number(confidence ?? 0);
    if (prediction === "UNCERTAIN") return { label: "Needs Review", className: "tier-review" };
    if (c >= 0.85) return { label: "High Confidence", className: "tier-high" };
    if (c >= 0.65) return { label: "Moderate Confidence", className: "tier-medium" };
    return { label: "Low Confidence", className: "tier-low" };
}

function explainWarning(code) {
    const mapping = {
        comments_unavailable: "Comments could not be fetched, so the graph context is weaker than usual.",
        image_unavailable: "The image could not be loaded, so the model relied more on text and graph cues.",
        image_extractor_unavailable: "Image features were unavailable for this inference run.",
        invalid_json_line: "A batch row could not be parsed as JSON.",
        skipped_line: "A batch row was skipped because graph construction or inference failed.",
    };
    return mapping[code] || code.replaceAll("_", " ");
}

function buildPredictionNarrative(result) {
    const probs = result.probabilities || {};
    const prediction = result.prediction || "UNKNOWN";
    const margin = formatProbability(result.margin);
    const nodes = result.graph_info?.num_nodes ?? 0;
    const edges = result.graph_info?.num_edges ?? 0;
    const threshold = formatProbability(result.uncertain_threshold);

    // Uncertain logic removed for demo impact

    const primaryProbability = prediction === "REAL" ? formatProbability(probs.REAL) : formatProbability(probs.FAKE);
    return `The model predicts ${prediction} based on the analyzed social propagation graph, which contains ${nodes} nodes and ${edges} edges.`;
}

function renderProbabilityBar(label, value, tone) {
    const width = Math.max(0, Math.min(100, Number(value ?? 0) * 100));
    return `
        <div class="prob-row">
            <div class="prob-row-head">
                <span>${escapeHtml(label)}</span>
                <strong>${formatProbability(value)}</strong>
            </div>
            <div class="prob-track">
                <div class="prob-fill ${tone}" style="width:${width}%"></div>
            </div>
        </div>
    `;
}

function setPredictionVisibility({ loading = false, single = false, batch = false, empty = false }) {
    document.getElementById("predictionLoading").classList.toggle("hidden", !loading);
    document.getElementById("predictionResult").classList.toggle("hidden", !single);
    document.getElementById("batchResult").classList.toggle("hidden", !batch);
    document.getElementById("predictionEmpty").classList.toggle("hidden", !empty);
}

function setPredictionMessage(elementId, text) {
    const element = document.getElementById(elementId);
    if (!element) return;
    element.textContent = text || "";
    element.classList.toggle("hidden", !text);
}

function escapeHtml(value) {
    return String(value ?? "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

async function loadPredictionHealth() {
    const badge = document.getElementById("predictorHealthBadge");
    const statusText = document.getElementById("predictionStatusText");
    try {
        const health = await api("/api/predict/health");
        badge.textContent = health.ready ? "Ready" : "Unavailable";
        badge.classList.toggle("ok", Boolean(health.ready));
        badge.classList.toggle("error", !health.ready);

        if (health.ready) {
            const threshold = Number(health.uncertain_threshold ?? 0.15).toFixed(2);
            const temperature = Number(health.temperature ?? 1).toFixed(3);
            statusText.textContent = `Model ready. Uncertain threshold ${threshold}, temperature ${temperature}. Use confidence and margin together when presenting the result.`;
        } else {
            statusText.textContent = "Predictor is not available.";
        }
    } catch (error) {
        badge.textContent = "Error";
        badge.classList.add("error");
        statusText.textContent = `Predictor init failed: ${error.message || "unknown error"}`;
        setPredictionMessage("predictionError", `Predictor initialization failed. ${error.message || "Please check the backend environment."}`);
    }
}

function renderPredictionWarnings(warnings) {
    if (!warnings || warnings.length === 0) {
        setPredictionMessage("predictionWarnings", "");
        return;
    }
    setPredictionMessage("predictionWarnings", `Warnings: ${warnings.map(explainWarning).join(" | ")}`);
}

function renderSinglePrediction(result) {
    const container = document.getElementById("predictionResult");
    const probs = result.probabilities || {};
    const post = result.post_info || {};
    const graph = result.graph_info || {};
    const model = result.model_info || {};
    const prediction = result.prediction || "UNKNOWN";
    const labelClass = prediction === "REAL"
        ? "result-real"
        : prediction === "FAKE"
            ? "result-fake"
            : "result-uncertain";
    const tier = getConfidenceTier(result.confidence, prediction);
    const summaryText = buildPredictionNarrative(result);

    const comments = (result.top_influential_comments || []).map((item) => `
        <div class="comment-item">
            <div class="comment-meta">Node ${escapeHtml(item.node_index)} / Attention ${formatProbability(item.attention_score)}</div>
            <p>${escapeHtml(item.text || "(empty)")}</p>
        </div>
    `).join("");

    container.innerHTML = `
        <div class="result-card">
            <div class="result-head">
                <div>
                    <p class="eyebrow">Prediction</p>
                    <h3 class="${labelClass}">${escapeHtml(prediction)}</h3>
                    <div class="result-tags">
                        <span class="result-tag ${tier.className}">${escapeHtml(tier.label)}</span>
                        <span class="result-tag">${escapeHtml(post.subreddit || "Reddit")}</span>
                    </div>
                </div>
                <!-- Score hidden as requested -->
            </div>

            <div class="post-title-display">
                <p class="eyebrow">Reddit Post Title</p>
                <h3>${escapeHtml(post.title || "No Title Available")}</h3>
            </div>
            
            <div class="result-grid">
                <div class="info-card">
                    <h4>Source Context</h4>
                    <dl>
                        <div><dt>Subreddit</dt><dd>${escapeHtml(post.subreddit || "-")}</dd></div>
                        <div><dt>Author</dt><dd>${escapeHtml(result.post_info?.author || "Anonymous")}</dd></div>
                        <div><dt>Comments</dt><dd>${escapeHtml(post.comment_count ?? "0")}</dd></div>
                    </dl>
                </div>

                <div class="info-card">
                    <h4>Graph Topology</h4>
                    <dl>
                        <div><dt>Nodes</dt><dd>${escapeHtml(graph.num_nodes ?? "0")}</dd></div>
                        <div><dt>Edges</dt><dd>${escapeHtml(graph.num_edges ?? "0")}</dd></div>
                        <div><dt>Propagation</dt><dd>${escapeHtml(model.model_type || "Cascade GNN")}</dd></div>
                    </dl>
                </div>

                <div class="info-card">
                    <h4>System State</h4>
                    <dl>
                        <div><dt>Device</dt><dd>${escapeHtml(model.device || "CPU")}</dd></div>
                        <div><dt>Mode</dt><dd>Inference</dd></div>
                        <div><dt>Status</dt><dd>Finalized</dd></div>
                    </dl>
                </div>

                <div class="info-card wide">
                    <h4>Social Propagation Explanation (GAT Attention)</h4>
                    <p class="muted-small">Our Graph Attention Network (GAT) identifies these comments as the most influential in predicting the credibility of the post.</p>
                    <div class="comment-list-container">
                        ${comments || '<p class="muted">No social signals available for this post.</p>'}
                    </div>
                </div>
            </div>
        </div>
    `;
}

function renderBatchPrediction(result) {
    const container = document.getElementById("batchResult");
    const summary = result.summary || {};
    const rows = (result.predictions || []).slice(0, 20).map((item) => `
        <tr>
            <td>${escapeHtml(item.id || "-")}</td>
            <td class="${item.prediction === "REAL" ? "result-real" : item.prediction === "FAKE" ? "result-fake" : "result-uncertain"}">${escapeHtml(item.prediction || "-")}</td>
            <!-- td hidden -->
            <td>${escapeHtml(item.ground_truth || "-")}</td>
            <td>${item.correct === null ? "-" : item.correct ? "Yes" : "No"}</td>
        </tr>
    `).join("");

    container.innerHTML = `
        <div class="result-card">
            <div class="result-grid">
                <div class="info-card"><h4>Total items</h4><strong>${escapeHtml(result.total ?? 0)}</strong></div>
                <div class="info-card"><h4>Coverage</h4><strong>${formatProbability(summary.coverage)}</strong></div>
                <div class="info-card"><h4>REAL / FAKE / UNCERTAIN</h4><strong>${summary.real_count ?? 0} / ${summary.fake_count ?? 0} / ${summary.uncertain_count ?? 0}</strong></div>
                <div class="info-card"><h4>Accuracy / F1</h4><strong>${summary.accuracy === null ? "--" : formatProbability(summary.accuracy)} / ${summary.f1_binary === null ? "--" : formatProbability(summary.f1_binary)}</strong></div>
            </div>
            <div class="summary-banner">
                <strong>Batch summary</strong>
                <p>Coverage measures how many items received a decisive REAL or FAKE verdict. Samples marked UNCERTAIN are excluded from binary accuracy and F1.</p>
            </div>
            <div class="table-wrap">
                <table class="results-table">
                    <thead>
                        <tr>
                            <th>ID</th>
                            <th>Prediction</th>
                            <!-- th hidden -->
                            <th>Ground truth</th>
                            <th>Correct</th>
                        </tr>
                    </thead>
                    <tbody>${rows || '<tr><td colspan="6" class="empty-row">No predictions available.</td></tr>'}</tbody>
                </table>
            </div>
        </div>
    `;
}

function useDemoUrl(key) {
    const example = DEMO_URLS[key];
    if (!example) return;
    const input = document.getElementById("predictUrlInput");
    const hint = document.getElementById("demoUrlHint");
    if (input) input.value = example.url;
    if (hint) hint.textContent = `${example.label}: ${example.note}`;
    document.getElementById("predictionStatusText").textContent = `${example.label} is ready in the input box.`;
}

async function copyDemoBatchPath(elementId = "demoBatchPath") {
    const path = document.getElementById(elementId)?.textContent?.trim();
    if (!path) return;
    try {
        await navigator.clipboard.writeText(path);
    } catch (error) { }
}

async function predictUrl() {
    const url = document.getElementById("predictUrlInput").value.trim();
    if (!url) {
        setPredictionMessage("predictionError", "A Reddit URL is required.");
        return;
    }

    setPredictionMessage("predictionError", "");
    setPredictionMessage("predictionWarnings", "");
    document.getElementById("predictionStatusText").textContent = "Running URL prediction...";
    setPredictionVisibility({ loading: true, single: false, batch: false, empty: false });

    try {
        const result = await api("/api/predict/url", "POST", { url });
        renderSinglePrediction(result);
        renderPredictionWarnings(result.warnings || []);
        setPredictionVisibility({ loading: false, single: true, batch: false, empty: false });
        document.getElementById("predictionStatusText").textContent = `Prediction completed for ${result.post_info?.id || "the selected post"}.`;
    } catch (error) {
        setPredictionMessage("predictionError", error.message || String(error));
        document.getElementById("predictionStatusText").textContent = "URL prediction failed.";
        setPredictionVisibility({ loading: false, single: false, batch: false, empty: true });
    }
}

async function predictBatch() {
    const fileInput = document.getElementById("predictBatchFile");
    const file = fileInput.files?.[0];
    if (!file) {
        setPredictionMessage("predictionError", "A JSONL file is required.");
        return;
    }

    setPredictionMessage("predictionError", "");
    setPredictionMessage("predictionWarnings", "");
    document.getElementById("predictionStatusText").textContent = `Running batch prediction for ${file.name}...`;
    setPredictionVisibility({ loading: true, single: false, batch: false, empty: false });

    try {
        const formData = new FormData();
        formData.append("file", file);
        const response = await fetch("/api/predict/batch", { method: "POST", body: formData });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Batch prediction failed.");
        }
        renderBatchPrediction(result);
        renderPredictionWarnings(result.warnings || []);
        setPredictionVisibility({ loading: false, single: false, batch: true, empty: false });
        document.getElementById("predictionStatusText").textContent = `Batch prediction completed for ${result.total ?? 0} items.`;
    } catch (error) {
        setPredictionMessage("predictionError", error.message || String(error));
        document.getElementById("predictionStatusText").textContent = "Batch prediction failed.";
        setPredictionVisibility({ loading: false, single: false, batch: false, empty: true });
    }
}

function updateTrainingForm() {
    const modelType = document.getElementById("modelType").value;
    const advancedGrid = document.getElementById("advancedTrainingGrid");
    const gnnTypeGroup = document.getElementById("gnnTypeGroup");
    const fusionTypeGroup = document.getElementById("fusionTypeGroup");

    const showAdvanced = modelType === "multimodal_v3";
    advancedGrid.classList.toggle("hidden", !showAdvanced);
    gnnTypeGroup.classList.toggle("hidden", !showAdvanced);
    fusionTypeGroup.classList.toggle("hidden", !showAdvanced);
}

function togglePresentationMode() {
    presentationModeEnabled = !presentationModeEnabled;
    document.body.classList.toggle("presentation-mode", presentationModeEnabled);
    const button = document.getElementById("presentationModeBtn");
    if (button) {
        button.classList.toggle("active", presentationModeEnabled);
        button.textContent = presentationModeEnabled ? "Exit Presentation Mode" : "Presentation Mode";
    }
}

function streamLog(taskId, logContainerId, onDone = null) {
    const container = document.getElementById(logContainerId);
    const pre = container.querySelector(".log-content");
    container.classList.remove("hidden");
    pre.textContent = "";

    currentTaskId = taskId;
    if (currentEventSource) currentEventSource.close();

    const eventSource = new EventSource(`/api/stream/${taskId}`);
    currentEventSource = eventSource;

    eventSource.onmessage = (event) => {
        const data = JSON.parse(event.data);
        pre.textContent += `${data.line}\n`;
        pre.scrollTop = pre.scrollHeight;

        if (data.done) {
            eventSource.close();
            currentEventSource = null;
            currentTaskId = null;
            setNavStatus("Ready", false);
            loadResultsDashboard();
            if (onDone) onDone();
        }
    };

    eventSource.onerror = () => {
        eventSource.close();
        currentEventSource = null;
        currentTaskId = null;
        setNavStatus("Ready", false);
        if (onDone) onDone();
    };
}

async function startTraining() {
    const modelType = document.getElementById("modelType").value;
    const payload = {
        model_type: modelType,
        epochs: Number(document.getElementById("trainEpochs").value),
        lr: Number(document.getElementById("trainLR").value),
        batch_size: Number(document.getElementById("trainBatch").value),
        patience: Number(document.getElementById("trainPatience").value),
        gnn_type: document.getElementById("gnnType").value,
        fusion_type: document.getElementById("fusionType").value,
    };

    document.getElementById("trainBtn").disabled = true;
    document.getElementById("stopTrainBtn").classList.remove("hidden");
    document.getElementById("trainingStatus").classList.remove("hidden");
    document.getElementById("progressText").textContent = `Training ${modelType}...`;
    document.getElementById("progressFill").style.width = "10%";
    setNavStatus(`Training ${modelType}...`, true);

    try {
        const result = await api("/api/train/start", "POST", payload);
        let progress = 10;
        const interval = setInterval(() => {
            if (progress < 92) {
                progress += 3;
                document.getElementById("progressFill").style.width = `${progress}%`;
            }
        }, 1000);

        streamLog(result.task_id, "trainLog", () => {
            clearInterval(interval);
            document.getElementById("progressFill").style.width = "100%";
            document.getElementById("progressText").textContent = "Training completed.";
            document.getElementById("trainBtn").disabled = false;
            document.getElementById("stopTrainBtn").classList.add("hidden");
            setTimeout(() => {
                document.getElementById("trainingStatus").classList.add("hidden");
            }, 2500);
        });
    } catch (error) {
        document.getElementById("trainBtn").disabled = false;
        document.getElementById("stopTrainBtn").classList.add("hidden");
        document.getElementById("trainingStatus").classList.add("hidden");
        setNavStatus("Ready", false);
        alert(error.message || String(error));
    }
}

async function startTrainAll() {
    const payload = {
        epochs: Number(document.getElementById("trainEpochs").value),
        batch_size: Number(document.getElementById("trainBatch").value),
    };

    document.getElementById("trainAllBtn").disabled = true;
    document.getElementById("stopTrainBtn").classList.remove("hidden");
    document.getElementById("trainingStatus").classList.remove("hidden");
    document.getElementById("progressText").textContent = "Training all configured models...";
    document.getElementById("progressFill").style.width = "10%";
    setNavStatus("Training all models...", true);

    try {
        const result = await api("/api/train-all/start", "POST", payload);
        let progress = 10;
        const interval = setInterval(() => {
            if (progress < 92) {
                progress += 2;
                document.getElementById("progressFill").style.width = `${progress}%`;
            }
        }, 1200);

        streamLog(result.task_id, "trainLog", () => {
            clearInterval(interval);
            document.getElementById("progressFill").style.width = "100%";
            document.getElementById("progressText").textContent = "Train-all sequence completed.";
            document.getElementById("trainAllBtn").disabled = false;
            document.getElementById("stopTrainBtn").classList.add("hidden");
            setTimeout(() => {
                document.getElementById("trainingStatus").classList.add("hidden");
            }, 2500);
        });
    } catch (error) {
        document.getElementById("trainAllBtn").disabled = false;
        document.getElementById("stopTrainBtn").classList.add("hidden");
        document.getElementById("trainingStatus").classList.add("hidden");
        setNavStatus("Ready", false);
        alert(error.message || String(error));
    }
}

async function stopCurrentTask() {
    if (!currentTaskId) return;
    try {
        await api(`/api/task/${currentTaskId}/stop`, "POST");
    } finally {
        document.getElementById("trainBtn").disabled = false;
        document.getElementById("trainAllBtn").disabled = false;
        document.getElementById("stopTrainBtn").classList.add("hidden");
        document.getElementById("trainingStatus").classList.add("hidden");
        setNavStatus("Stopped", false);
    }
}

async function loadResultsDashboard() {
    try {
        const data = await api("/api/results/dashboard");
        const dataset = data.dataset || {};
        const champion = data.champion || {};
        const rows = data.leaderboard || [];
        const figures = data.figures || [];

        document.getElementById("datasetTotal").textContent = dataset.total_samples || "--";
        document.getElementById("championAccuracy").textContent = formatPercent(champion.accuracy_pct);
        document.getElementById("championF1").textContent = formatPercent(champion.f1_pct);
        document.getElementById("championDetails").textContent = champion.seed && champion.threshold !== undefined
            ? `Seed ${champion.seed} / threshold ${champion.threshold}`
            : "latest CLIP-only track";

        document.getElementById("heroChampionAcc").textContent = formatPercent(champion.accuracy_pct);
        document.getElementById("heroDatasetSize").textContent = dataset.total_samples || "--";
        document.getElementById("resultsScopeNote").textContent = data.scope_note || "";

        const body = document.getElementById("resultsTableBody");
        body.innerHTML = rows.length
            ? rows.map((row) => `
                <tr>
                    <td>${escapeHtml(row.model || "-")}</td>
                    <td>${escapeHtml(row.track || "-")}</td>
                    <td>${formatPercent(row.accuracy_pct)}</td>
                    <td>${formatPercent(row.f1_pct)}</td>
                    <td>${escapeHtml(row.notes || "-")}</td>
                </tr>
            `).join("")
            : '<tr><td colspan="5" class="empty-row">No leaderboard data found.</td></tr>';

        const figureGrid = document.getElementById("resultsFigureGrid");
        figureGrid.innerHTML = figures.length
            ? figures.map((figure) => `
                <a class="figure-card" href="${figure.url}" target="_blank" rel="noopener noreferrer">
                    <img src="${figure.url}" alt="${escapeHtml(figure.title)}" loading="lazy">
                    <div class="figure-meta">
                        <strong>${escapeHtml(figure.title)}</strong>
                        <span>${escapeHtml(figure.subtitle || "")}</span>
                    </div>
                </a>
            `).join("")
            : '<div class="empty-state">No figures found.</div>';
    } catch (error) {
        document.getElementById("resultsScopeNote").textContent = error.message || "Could not load results.";
    }
}

window.addEventListener("DOMContentLoaded", () => {
    updateTrainingForm();
    loadPredictionHealth();
    loadResultsDashboard();
    document.getElementById("presentationModeBtn")?.addEventListener("click", togglePresentationMode);
    useDemoUrl("science_a");
});
