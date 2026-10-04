// ==========================================================================
// AI COMPLAINT OPERATIONS CENTER - JAVASCRIPT APPLICATION CONTROLLER
// ==========================================================================

document.addEventListener("DOMContentLoaded", () => {
    // --- STATE MANAGEMENT ---
    const state = {
        currentView: "new",
        lastAnalysisResult: null,
        queue: [], // Local in-memory session work order queue
        ticketCounter: 1001,
        metrics: null,
        speechRecognition: null,
        isListening: false
    };

    // --- DOM ELEMENTS ---
    const elements = {
        navItems: document.querySelectorAll(".nav-item"),
        viewSections: document.querySelectorAll(".view-section"),
        pageTitle: document.getElementById("pageTitle"),
        pageSubtitle: document.getElementById("pageSubtitle"),
        queueBadge: document.getElementById("queueBadge"),

        // New Complaint
        complaintInput: document.getElementById("complaintInput"),
        btnAnalyze: document.getElementById("btnAnalyze"),
        btnClearInput: document.getElementById("btnClearInput"),
        btnVoice: document.getElementById("btnVoice"),
        micIcon: document.getElementById("micIcon"),
        micText: document.getElementById("micText"),
        voiceStatus: document.getElementById("voiceStatus"),
        presetContainer: document.getElementById("presetContainer"),

        // Report Card
        analysisReportCard: document.getElementById("analysisReportCard"),
        reportTimestamp: document.getElementById("reportTimestamp"),
        reportPriority: document.getElementById("reportPriority"),
        reportCategory: document.getElementById("reportCategory"),
        reportPriorityText: document.getElementById("reportPriorityText"),
        reportSeverity: document.getElementById("reportSeverity"),
        reportConfidenceText: document.getElementById("reportConfidenceText"),
        reportConfidenceFill: document.getElementById("reportConfidenceFill"),
        reportDepartment: document.getElementById("reportDepartment"),
        reportLocation: document.getElementById("reportLocation"),
        reportObject: document.getElementById("reportObject"),
        reportProblem: document.getElementById("reportProblem"),
        reportTime: document.getElementById("reportTime"),
        reportAction: document.getElementById("reportAction"),
        reportSignals: document.getElementById("reportSignals"),
        btnAddToQueue: document.getElementById("btnAddToQueue"),
        btnCopyReport: document.getElementById("btnCopyReport"),
        btnNewAnalysis: document.getElementById("btnNewAnalysis"),
        btnQuickNew: document.getElementById("btnQuickNew"),

        // Overview
        metricTotal: document.getElementById("metricTotal"),
        metricHigh: document.getElementById("metricHigh"),
        metricCritical: document.getElementById("metricCritical"),
        metricResolved: document.getElementById("metricResolved"),
        overviewEmptyState: document.getElementById("overviewEmptyState"),
        overviewChartsArea: document.getElementById("overviewChartsArea"),
        priorityDistributionBars: document.getElementById("priorityDistributionBars"),
        categoryBreakdownList: document.getElementById("categoryBreakdownList"),
        btnGoToNew: document.getElementById("btnGoToNew"),
        btnLoadDemoQueue: document.getElementById("btnLoadDemoQueue"),

        // Queue
        queueSearch: document.getElementById("queueSearch"),
        filterCategory: document.getElementById("filterCategory"),
        filterPriority: document.getElementById("filterPriority"),
        filterStatus: document.getElementById("filterStatus"),
        sortQueue: document.getElementById("sortQueue"),
        queueCountText: document.getElementById("queueCountText"),
        queueTableBody: document.getElementById("queueTableBody"),
        btnClearQueue: document.getElementById("btnClearQueue"),

        // Analytics
        analyticsDeptWorkload: document.getElementById("analyticsDeptWorkload"),
        analyticsConfidence: document.getElementById("analyticsConfidence"),

        // Model Insights
        insightDatasetSize: document.getElementById("insightDatasetSize"),
        insightCatAcc: document.getElementById("insightCatAcc"),
        insightPrioAcc: document.getElementById("insightPrioAcc"),
        insightFeatures: document.getElementById("insightFeatures"),
        categoryKeywordsGrid: document.getElementById("categoryKeywordsGrid"),

        // Modal
        ticketModal: document.getElementById("ticketModal"),
        modalTicketId: document.getElementById("modalTicketId"),
        modalBody: document.getElementById("modalBody"),
        btnCloseModal: document.getElementById("btnCloseModal"),

        // Toast
        toastContainer: document.getElementById("toastContainer")
    };

    // --- INITIALIZATION ---
    init();

    function init() {
        setupNavigation();
        setupInputEvents();
        setupSpeechRecognition();
        fetchSamplePresets();
        fetchModelMetrics();
        setupQueueControls();
        setupModal();
        updateOverviewAndQueue();
    }

    // --- NAVIGATION CONTROLLER ---
    function setupNavigation() {
        const titles = {
            overview: { title: "Operations Dashboard", subtitle: "Session complaint metrics, priority distribution, and queue analytics." },
            new: { title: "Complaint Operations Center", subtitle: "Classify, prioritize and route complaints using NLP and Machine Learning." },
            queue: { title: "Work Order Complaint Queue", subtitle: "Manage active session tickets, status updates, and department routing." },
            analytics: { title: "Routing & Severity Analytics", subtitle: "Department workloads, ML confidence distributions, and queue metrics." },
            insights: { title: "Machine Learning & NLP Insights", subtitle: "Supervised model architecture, empirical accuracy scores, and TF-IDF features." }
        };

        elements.navItems.forEach(item => {
            item.addEventListener("click", (e) => {
                e.preventDefault();
                const viewTarget = item.getAttribute("data-view");
                switchView(viewTarget);
            });
        });

        if (elements.btnQuickNew) {
            elements.btnQuickNew.addEventListener("click", () => switchView("new"));
        }
        if (elements.btnGoToNew) {
            elements.btnGoToNew.addEventListener("click", () => switchView("new"));
        }
    }

    function switchView(viewName) {
        state.currentView = viewName;
        elements.navItems.forEach(nav => {
            if (nav.getAttribute("data-view") === viewName) {
                nav.classList.add("active");
            } else {
                nav.classList.remove("active");
            }
        });

        elements.viewSections.forEach(sec => {
            if (sec.id === `view-${viewName}`) {
                sec.classList.add("active");
            } else {
                sec.classList.remove("active");
            }
        });

        const viewInfo = {
            overview: { title: "Operations Dashboard", subtitle: "Session complaint metrics, priority distribution, and queue analytics." },
            new: { title: "Complaint Operations Center", subtitle: "Classify, prioritize and route complaints using NLP and Machine Learning." },
            queue: { title: "Work Order Complaint Queue", subtitle: "Manage active session tickets, status updates, and department routing." },
            analytics: { title: "Routing & Severity Analytics", subtitle: "Department workloads, ML confidence distributions, and queue metrics." },
            insights: { title: "Machine Learning & NLP Insights", subtitle: "Supervised model architecture, empirical accuracy scores, and TF-IDF features." }
        }[viewName] || { title: "Complaint Operations Center", subtitle: "Classify, prioritize and route complaints." };

        elements.pageTitle.textContent = viewInfo.title;
        elements.pageSubtitle.textContent = viewInfo.subtitle;

        if (viewName === "overview" || viewName === "queue" || viewName === "analytics") {
            renderQueueAndMetrics();
        }
    }

    // --- SAMPLE PRESETS & INPUT ---
    async function fetchSamplePresets() {
        try {
            const res = await fetch("/api/samples");
            if (res.ok) {
                const samples = await res.json();
                renderPresets(samples);
            }
        } catch (err) {
            console.error("Failed to load sample presets:", err);
        }
    }

    function renderPresets(samples) {
        elements.presetContainer.innerHTML = "";
        samples.forEach(sample => {
            const chip = document.createElement("button");
            chip.type = "button";
            chip.className = "preset-chip";
            chip.innerHTML = `<i class="fa-solid fa-tag"></i> ${sample.title}`;
            chip.addEventListener("click", () => {
                elements.complaintInput.value = sample.text;
                showToast(`Loaded preset: "${sample.title}"`);
            });
            elements.presetContainer.appendChild(chip);
        });
    }

    function setupInputEvents() {
        elements.btnAnalyze.addEventListener("click", handleAnalyze);
        elements.btnClearInput.addEventListener("click", () => {
            elements.complaintInput.value = "";
            elements.analysisReportCard.classList.add("hidden");
            state.lastAnalysisResult = null;
        });

        elements.btnAddToQueue.addEventListener("click", handleAddToQueue);
        elements.btnCopyReport.addEventListener("click", handleCopyReport);
        elements.btnNewAnalysis.addEventListener("click", () => {
            elements.complaintInput.value = "";
            elements.analysisReportCard.classList.add("hidden");
            elements.complaintInput.focus();
        });
    }

    // --- COMPLAINT ANALYSIS ENGINE ---
    async function handleAnalyze() {
        const text = elements.complaintInput.value.trim();
        if (!text) {
            showToast("Please enter a complaint description first.", "warning");
            return;
        }

        elements.btnAnalyze.disabled = true;
        elements.btnAnalyze.innerHTML = `<i class="fa-solid fa-spinner fa-spin"></i> Analyzing...`;

        try {
            const response = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ complaint: text })
            });

            const data = await response.json();

            if (!response.ok) {
                throw new Error(data.error || "Analysis failed.");
            }

            state.lastAnalysisResult = data;
            renderAnalysisReport(data);
            showToast(`Analysis Complete: Category '${data.category}' (${data.priority} Priority)`);
        } catch (err) {
            showToast(`Error: ${err.message}`, "error");
        } finally {
            elements.btnAnalyze.disabled = false;
            elements.btnAnalyze.innerHTML = `<i class="fa-solid fa-wand-magic-sparkles"></i> <span>Analyze Complaint</span>`;
        }
    }

    function renderAnalysisReport(data) {
        elements.analysisReportCard.className = `card report-card prio-${data.priority}`;
        elements.analysisReportCard.classList.remove("hidden");

        elements.reportTimestamp.textContent = `Generated ${new Date().toLocaleTimeString()}`;
        elements.reportPriority.textContent = data.priority;
        elements.reportPriority.className = `badge-priority ${data.priority}`;

        elements.reportCategory.textContent = data.category;
        elements.reportPriorityText.textContent = data.priority;
        elements.reportSeverity.textContent = data.severity;

        elements.reportConfidenceText.textContent = `${data.confidence.toFixed(1)}%`;
        elements.reportConfidenceFill.style.width = `${Math.min(data.confidence, 100)}%`;

        elements.reportDepartment.innerHTML = `<i class="fa-solid fa-building-user"></i> ${data.department}`;
        elements.reportLocation.textContent = data.location;
        elements.reportObject.textContent = data.object;
        elements.reportProblem.textContent = data.problem;
        elements.reportTime.textContent = data.time_reference;
        elements.reportAction.textContent = data.recommended_action;

        // Reasoning signals
        elements.reportSignals.innerHTML = "";
        if (data.reasoning_signals && data.reasoning_signals.length > 0) {
            data.reasoning_signals.forEach(sig => {
                const pill = document.createElement("span");
                pill.className = "signal-pill";
                pill.textContent = sig;
                elements.reportSignals.appendChild(pill);
            });
        } else {
            elements.reportSignals.innerHTML = `<span class="signal-pill">standard_tokens</span>`;
        }

        // Scroll cleanly to report card
        elements.analysisReportCard.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }

    // --- QUEUE MANAGEMENT ---
    function handleAddToQueue() {
        if (!state.lastAnalysisResult) return;

        const res = state.lastAnalysisResult;
        const ticketId = `CMP-${state.ticketCounter++}`;

        const ticket = {
            id: ticketId,
            text: res.complaint_text,
            shortTitle: res.complaint_text.length > 60 ? res.complaint_text.substring(0, 58) + "..." : res.complaint_text,
            category: res.category,
            priority: res.priority,
            severity: res.severity,
            location: res.location,
            department: res.department,
            confidence: res.confidence,
            recommended_action: res.recommended_action,
            status: "New",
            timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
        };

        state.queue.unshift(ticket);
        showToast(`Ticket ${ticketId} added to Operations Queue!`);
        updateOverviewAndQueue();

        // Prompt user option
        switchView("queue");
    }

    function setupQueueControls() {
        elements.queueSearch.addEventListener("input", renderQueueTable);
        elements.filterCategory.addEventListener("change", renderQueueTable);
        elements.filterPriority.addEventListener("change", renderQueueTable);
        elements.filterStatus.addEventListener("change", renderQueueTable);
        elements.sortQueue.addEventListener("change", renderQueueTable);

        elements.btnClearQueue.addEventListener("click", () => {
            if (state.queue.length === 0) return;
            if (confirm("Are you sure you want to clear all tickets from the session queue?")) {
                state.queue = [];
                updateOverviewAndQueue();
                showToast("Session queue cleared.");
            }
        });

        elements.btnLoadDemoQueue.addEventListener("click", loadDemoQueue);
    }

    async function loadDemoQueue() {
        try {
            const res = await fetch("/api/samples");
            const samples = await res.json();

            for (const sample of samples) {
                const analyzeRes = await fetch("/api/analyze", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ complaint: sample.text })
                });

                if (analyzeRes.ok) {
                    const data = await analyzeRes.json();
                    const ticketId = `CMP-${state.ticketCounter++}`;
                    state.queue.push({
                        id: ticketId,
                        text: data.complaint_text,
                        shortTitle: data.complaint_text.length > 60 ? data.complaint_text.substring(0, 58) + "..." : data.complaint_text,
                        category: data.category,
                        priority: data.priority,
                        severity: data.severity,
                        location: data.location,
                        department: data.department,
                        confidence: data.confidence,
                        recommended_action: data.recommended_action,
                        status: Math.random() > 0.6 ? "In Progress" : "New",
                        timestamp: new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })
                    });
                }
            }

            updateOverviewAndQueue();
            showToast(`Loaded ${samples.length} demo tickets into queue!`);
        } catch (err) {
            showToast("Failed to load demo queue.", "error");
        }
    }

    function updateOverviewAndQueue() {
        elements.queueBadge.textContent = state.queue.length;
        renderQueueTable();
        renderOverviewMetrics();
        renderAnalytics();
    }

    function renderQueueAndMetrics() {
        renderQueueTable();
        renderOverviewMetrics();
        renderAnalytics();
    }

    function getFilteredQueue() {
        let list = [...state.queue];
        const search = elements.queueSearch.value.toLowerCase().trim();
        const cat = elements.filterCategory.value;
        const prio = elements.filterPriority.value;
        const status = elements.filterStatus.value;
        const sort = elements.sortQueue.value;

        if (search) {
            list = list.filter(item =>
                item.id.toLowerCase().includes(search) ||
                item.text.toLowerCase().includes(search) ||
                item.location.toLowerCase().includes(search)
            );
        }

        if (cat) list = list.filter(item => item.category === cat);
        if (prio) list = list.filter(item => item.priority === prio);
        if (status) list = list.filter(item => item.status === status);

        // Sorting
        const prioOrder = { "CRITICAL": 4, "HIGH": 3, "MEDIUM": 2, "LOW": 1 };
        if (sort === "priority") {
            list.sort((a, b) => prioOrder[b.priority] - prioOrder[a.priority]);
        } else if (sort === "newest") {
            list.reverse();
        } else if (sort === "category") {
            list.sort((a, b) => a.category.localeCompare(b.category));
        }

        return list;
    }

    function renderQueueTable() {
        const filtered = getFilteredQueue();
        elements.queueCountText.textContent = `Showing ${filtered.length} of ${state.queue.length} complaints`;
        elements.queueTableBody.innerHTML = "";

        if (filtered.length === 0) {
            elements.queueTableBody.innerHTML = `
                <tr>
                    <td colspan="8" style="text-align: center; color: var(--text-muted); padding: 2rem;">
                        No tickets matching filter criteria.
                    </td>
                </tr>
            `;
            return;
        }

        filtered.forEach(ticket => {
            const tr = document.createElement("tr");

            tr.innerHTML = `
                <td class="ticket-id">${ticket.id}</td>
                <td title="${ticket.text}">${ticket.shortTitle}</td>
                <td><span class="preset-chip" style="font-size:0.75rem;">${ticket.category}</span></td>
                <td><span class="badge-priority ${ticket.priority}" style="padding:0.15rem 0.6rem; font-size:0.75rem;">${ticket.priority}</span></td>
                <td>${ticket.location}</td>
                <td><small style="color:var(--primary); font-weight:600;">${ticket.department}</small></td>
                <td>
                    <select class="status-select status-${ticket.status.replace(' ', '-')}" data-id="${ticket.id}">
                        <option value="New" ${ticket.status === "New" ? "selected" : ""}>New</option>
                        <option value="In Progress" ${ticket.status === "In Progress" ? "selected" : ""}>In Progress</option>
                        <option value="Resolved" ${ticket.status === "Resolved" ? "selected" : ""}>Resolved</option>
                    </select>
                </td>
                <td>
                    <button class="btn btn-ghost btn-sm btn-view-ticket" data-id="${ticket.id}"><i class="fa-solid fa-eye"></i></button>
                    <button class="btn btn-ghost btn-sm btn-del-ticket" data-id="${ticket.id}" style="color:var(--prio-critical);"><i class="fa-solid fa-trash"></i></button>
                </td>
            `;

            elements.queueTableBody.appendChild(tr);
        });

        // Event listeners for inline actions
        document.querySelectorAll(".status-select").forEach(sel => {
            sel.addEventListener("change", (e) => {
                const id = e.target.getAttribute("data-id");
                const newStatus = e.target.value;
                const t = state.queue.find(x => x.id === id);
                if (t) {
                    t.status = newStatus;
                    e.target.className = `status-select status-${newStatus.replace(' ', '-')}`;
                    showToast(`Updated ${id} status to '${newStatus}'`);
                    updateOverviewAndQueue();
                }
            });
        });

        document.querySelectorAll(".btn-view-ticket").forEach(btn => {
            btn.addEventListener("click", (e) => {
                const id = btn.getAttribute("data-id");
                openTicketModal(id);
            });
        });

        document.querySelectorAll(".btn-del-ticket").forEach(btn => {
            btn.addEventListener("click", (e) => {
                const id = btn.getAttribute("data-id");
                state.queue = state.queue.filter(x => x.id !== id);
                showToast(`Ticket ${id} removed.`);
                updateOverviewAndQueue();
            });
        });
    }

    // --- OVERVIEW & METRICS ---
    function renderOverviewMetrics() {
        const total = state.queue.length;
        const high = state.queue.filter(q => q.priority === "HIGH").length;
        const critical = state.queue.filter(q => q.priority === "CRITICAL").length;
        const resolved = state.queue.filter(q => q.status === "Resolved").length;

        elements.metricTotal.textContent = total;
        elements.metricHigh.textContent = high;
        elements.metricCritical.textContent = critical;
        elements.metricResolved.textContent = resolved;

        if (total === 0) {
            elements.overviewEmptyState.classList.remove("hidden");
            elements.overviewChartsArea.classList.add("hidden");
        } else {
            elements.overviewEmptyState.classList.add("hidden");
            elements.overviewChartsArea.classList.remove("hidden");

            renderPriorityBars(total);
            renderCategoryBars(total);
        }
    }

    function renderPriorityBars(total) {
        const counts = { CRITICAL: 0, HIGH: 0, MEDIUM: 0, LOW: 0 };
        state.queue.forEach(q => {
            if (counts[q.priority] !== undefined) counts[q.priority]++;
        });

        const colors = {
            CRITICAL: "var(--prio-critical)",
            HIGH: "var(--prio-high)",
            MEDIUM: "var(--prio-medium)",
            LOW: "var(--prio-low)"
        };

        elements.priorityDistributionBars.innerHTML = "";
        ["CRITICAL", "HIGH", "MEDIUM", "LOW"].forEach(prio => {
            const count = counts[prio];
            const pct = total > 0 ? Math.round((count / total) * 100) : 0;

            const div = document.createElement("div");
            div.className = "dist-bar-item";
            div.innerHTML = `
                <div class="dist-bar-header">
                    <span>${prio}</span>
                    <span>${count} (${pct}%)</span>
                </div>
                <div class="dist-track">
                    <div class="dist-fill" style="width: ${pct}%; background-color: ${colors[prio]};"></div>
                </div>
            `;
            elements.priorityDistributionBars.appendChild(div);
        });
    }

    function renderCategoryBars(total) {
        const catCounts = {};
        state.queue.forEach(q => {
            catCounts[q.category] = (catCounts[q.category] || 0) + 1;
        });

        elements.categoryBreakdownList.innerHTML = "";
        Object.entries(catCounts).forEach(([cat, count]) => {
            const pct = Math.round((count / total) * 100);
            const div = document.createElement("div");
            div.className = "dist-bar-item";
            div.innerHTML = `
                <div class="dist-bar-header">
                    <span>${cat}</span>
                    <span>${count} (${pct}%)</span>
                </div>
                <div class="dist-track">
                    <div class="dist-fill" style="width: ${pct}%; background-color: var(--primary);"></div>
                </div>
            `;
            elements.categoryBreakdownList.appendChild(div);
        });
    }

    // --- ANALYTICS ---
    function renderAnalytics() {
        if (state.queue.length === 0) {
            elements.analyticsDeptWorkload.innerHTML = `<p class="text-muted">No tickets in queue. Add complaints to see department routing analytics.</p>`;
            elements.analyticsConfidence.innerHTML = `<p class="text-muted">No tickets in queue. Add complaints to see confidence stats.</p>`;
            return;
        }

        const depts = {};
        let totalConf = 0;
        state.queue.forEach(q => {
            depts[q.department] = (depts[q.department] || 0) + 1;
            totalConf += q.confidence || 75;
        });

        const avgConf = (totalConf / state.queue.length).toFixed(1);

        let deptHtml = `<div class="distribution-bars">`;
        Object.entries(depts).forEach(([d, count]) => {
            const pct = Math.round((count / state.queue.length) * 100);
            deptHtml += `
                <div class="dist-bar-item">
                    <div class="dist-bar-header"><span>${d}</span><span>${count} tickets (${pct}%)</span></div>
                    <div class="dist-track"><div class="dist-fill" style="width:${pct}%; background-color:var(--accent-violet);"></div></div>
                </div>
            `;
        });
        deptHtml += `</div>`;
        elements.analyticsDeptWorkload.innerHTML = deptHtml;

        elements.analyticsConfidence.innerHTML = `
            <div class="stat-box" style="text-align:center;">
                <span class="stat-label">Average Session ML Confidence</span>
                <h2 style="font-size:2.2rem; color:var(--primary); font-weight:800;">${avgConf}%</h2>
                <small>Calculated across ${state.queue.length} analyzed complaints</small>
            </div>
        `;
    }

    // --- MODEL INSIGHTS & METRICS ---
    async function fetchModelMetrics() {
        try {
            const res = await fetch("/api/metrics");
            if (res.ok) {
                const metrics = await res.json();
                state.metrics = metrics;
                renderModelInsights(metrics);
            }
        } catch (err) {
            console.error("Failed to load model metrics:", err);
        }
    }

    function renderModelInsights(m) {
        if (!m) return;
        elements.insightDatasetSize.textContent = `${m.dataset_size} Samples`;
        elements.insightCatAcc.textContent = `${m.category_accuracy}%`;
        elements.insightPrioAcc.textContent = `${m.priority_accuracy}%`;
        elements.insightFeatures.textContent = `${m.feature_count} N-Grams`;

        if (m.top_keywords) {
            elements.categoryKeywordsGrid.innerHTML = "";
            Object.entries(m.top_keywords).forEach(([cat, keywords]) => {
                const card = document.createElement("div");
                card.className = "cat-kw-card";
                let pillsHtml = keywords.map(kw => `<span class="kw-pill">${kw}</span>`).join("");
                card.innerHTML = `
                    <h4><i class="fa-solid fa-tag"></i> ${cat}</h4>
                    <div class="kw-pills">${pillsHtml}</div>
                `;
                elements.categoryKeywordsGrid.appendChild(card);
            });
        }
    }

    // --- WEB SPEECH RECOGNITION ---
    function setupSpeechRecognition() {
        const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

        if (!SpeechRecognition) {
            elements.btnVoice.disabled = true;
            elements.voiceStatus.textContent = "Browser speech recognition unavailable";
            return;
        }

        state.speechRecognition = new SpeechRecognition();
        state.speechRecognition.continuous = false;
        state.speechRecognition.interimResults = false;
        state.speechRecognition.lang = 'en-US';

        state.speechRecognition.onstart = () => {
            state.isListening = true;
            elements.micIcon.className = "fa-solid fa-microphone-slash listening-pulse";
            elements.micText.textContent = "Stop Listening";
            elements.voiceStatus.textContent = "Listening... Speak now";
            elements.voiceStatus.className = "voice-status listening-pulse";
        };

        state.speechRecognition.onresult = (event) => {
            const transcript = event.results[0][0].transcript;
            elements.complaintInput.value += (elements.complaintInput.value ? " " : "") + transcript;
            showToast("Voice input transcribed!");
        };

        state.speechRecognition.onerror = (event) => {
            console.error("Speech recognition error:", event.error);
            showToast(`Voice error: ${event.error}`, "warning");
            stopListening();
        };

        state.speechRecognition.onend = () => {
            stopListening();
        };

        elements.btnVoice.addEventListener("click", () => {
            if (state.isListening) {
                state.speechRecognition.stop();
            } else {
                state.speechRecognition.start();
            }
        });
    }

    function stopListening() {
        state.isListening = false;
        elements.micIcon.className = "fa-solid fa-microphone";
        elements.micText.textContent = "Voice Input";
        elements.voiceStatus.textContent = "Microphone ready";
        elements.voiceStatus.className = "voice-status";
    }

    // --- MODAL CONTROLLER ---
    function setupModal() {
        elements.btnCloseModal.addEventListener("click", closeModal);
        elements.ticketModal.addEventListener("click", (e) => {
            if (e.target === elements.ticketModal) closeModal();
        });
    }

    function openTicketModal(ticketId) {
        const t = state.queue.find(x => x.id === ticketId);
        if (!t) return;

        elements.modalTicketId.textContent = t.id;
        elements.modalBody.innerHTML = `
            <div class="report-grid">
                <div class="report-item"><span class="label">Category</span><strong>${t.category}</strong></div>
                <div class="report-item"><span class="label">Priority</span><span class="badge-priority ${t.priority}">${t.priority}</span></div>
                <div class="report-item"><span class="label">Severity</span><strong>${t.severity}</strong></div>
                <div class="report-item"><span class="label">Confidence</span><strong>${t.confidence}%</strong></div>
            </div>
            <div style="background:var(--bg-main); padding:1rem; border-radius:var(--radius-md);">
                <small style="color:var(--text-muted); font-weight:700;">FULL COMPLAINT TEXT</small>
                <p style="font-size:0.95rem; margin-top:0.25rem;">"${t.text}"</p>
            </div>
            <div class="entity-grid">
                <div class="entity-card"><i class="fa-solid fa-location-dot"></i><div><small>Location</small><strong>${t.location}</strong></div></div>
                <div class="entity-card"><i class="fa-solid fa-building-user"></i><div><small>Department</small><strong>${t.department}</strong></div></div>
            </div>
            <div class="action-box">
                <i class="fa-solid fa-route"></i>
                <div><small>Action Protocol</small><p>${t.recommended_action}</p></div>
            </div>
        `;

        elements.ticketModal.classList.remove("hidden");
    }

    function closeModal() {
        elements.ticketModal.classList.add("hidden");
    }

    // --- COPY REPORT FUNCTION ---
    function handleCopyReport() {
        if (!state.lastAnalysisResult) return;
        const d = state.lastAnalysisResult;
        const text = `AI ANALYSIS REPORT
Category: ${d.category}
Priority: ${d.priority} | Severity: ${d.severity}
Confidence: ${d.confidence}%
Department: ${d.department}
Location: ${d.location}
Problem: ${d.problem}
Action: ${d.recommended_action}`;

        navigator.clipboard.writeText(text).then(() => {
            showToast("Analysis report copied to clipboard!");
        });
    }

    // --- TOAST NOTIFICATIONS ---
    function showToast(message, type = "info") {
        const toast = document.createElement("div");
        toast.className = `toast toast-${type}`;
        toast.innerHTML = `<i class="fa-solid fa-circle-info"></i> ${message}`;
        elements.toastContainer.appendChild(toast);

        setTimeout(() => {
            toast.style.opacity = "0";
            toast.style.transform = "translateX(50px)";
            setTimeout(() => toast.remove(), 300);
        }, 3000);
    }
});
