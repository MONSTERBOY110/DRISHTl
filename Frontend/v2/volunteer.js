/**
 * Drishti 2.0 — Volunteer Alert Radar (M4, Task D5)
 * Real-time SSE stream, Web Audio synthesizer chime, haptic vibration,
 * and Ack / Found / False Alarm case workflow.
 */

// ---------------------------------------------------------------------------
// State & Elements
// ---------------------------------------------------------------------------
const state = {
    alerts: new Map(),        // alert_id -> alert object
    currentFilter: 'all',     // 'all' | 'active' | 'resolved'
    alertsEnabled: false,
    audioCtx: null,
    sseSource: null,
    reconnectTimer: null,
    callsign: localStorage.getItem('drishti_volunteer_unit') || 'Volunteer 1',
    lastHeartbeat: null,
};

const elements = {
    connectionPill: document.getElementById('connection-pill'),
    connectionText: document.getElementById('connection-text'),
    callsignInput: document.getElementById('callsign-input'),
    btnAlertsToggle: document.getElementById('btn-alerts-toggle'),
    alertsToggleText: document.getElementById('alerts-toggle-text'),
    alertsFeed: document.getElementById('alerts-feed'),
    emptyState: document.getElementById('empty-state'),
    statActive: document.getElementById('stat-active'),
    statFound: document.getElementById('stat-found'),
    statTotal: document.getElementById('stat-total'),
    countAll: document.getElementById('count-all'),
    countActive: document.getElementById('count-active'),
    countResolved: document.getElementById('count-resolved'),
    heartbeatInfo: document.getElementById('heartbeat-info'),
    lightboxModal: document.getElementById('lightbox-modal'),
    lightboxImg: document.getElementById('lightbox-img'),
    btnCloseLightbox: document.getElementById('btn-close-lightbox'),
    btnTestAlert: document.getElementById('btn-test-alert'),
    filterBtns: document.querySelectorAll('.filter-btn'),
};

// ---------------------------------------------------------------------------
// Audio Synthesizer (Zero External Assets, 100% Offline)
// ---------------------------------------------------------------------------
function initAudio() {
    try {
        const AudioContextClass = window.AudioContext || window.webkitAudioContext;
        if (!AudioContextClass) return null;
        if (!state.audioCtx) {
            state.audioCtx = new AudioContextClass();
        }
        if (state.audioCtx.state === 'suspended') {
            state.audioCtx.resume();
        }
        return state.audioCtx;
    } catch (err) {
        console.warn('AudioContext init error:', err);
        return null;
    }
}

function playAlertChime() {
    if (!state.alertsEnabled) return;
    const ctx = initAudio();
    if (!ctx) return;

    try {
        const now = ctx.currentTime;
        // Two-tone rising radar alert pulse (880Hz -> 1320Hz)
        const osc1 = ctx.createOscillator();
        const osc2 = ctx.createOscillator();
        const gain = ctx.createGain();

        osc1.type = 'sine';
        osc1.frequency.setValueAtTime(880, now);
        osc1.frequency.exponentialRampToValueAtTime(1320, now + 0.15);

        osc2.type = 'triangle';
        osc2.frequency.setValueAtTime(1320, now + 0.2);
        osc2.frequency.setValueAtTime(1760, now + 0.35);

        gain.gain.setValueAtTime(0.001, now);
        gain.gain.exponentialRampToValueAtTime(0.5, now + 0.05);
        gain.gain.exponentialRampToValueAtTime(0.001, now + 0.5);

        osc1.connect(gain);
        osc2.connect(gain);
        gain.connect(ctx.destination);

        osc1.start(now);
        osc1.stop(now + 0.2);
        osc2.start(now + 0.2);
        osc2.stop(now + 0.5);
    } catch (e) {
        console.warn('Failed to play alert sound:', e);
    }
}

function triggerHaptics() {
    if (!state.alertsEnabled) return;
    if ('vibrate' in navigator) {
        try {
            // Urgent haptic pulse: buzz-buzz-pause-long buzz
            navigator.vibrate([200, 100, 200, 100, 400]);
        } catch (e) {
            console.warn('Haptic vibration failed:', e);
        }
    }
}

// ---------------------------------------------------------------------------
// Callsign & Audio Permissions Setup
// ---------------------------------------------------------------------------
function setupControls() {
    elements.callsignInput.value = state.callsign;
    elements.callsignInput.addEventListener('input', (e) => {
        state.callsign = e.target.value.trim() || 'Volunteer 1';
        localStorage.setItem('drishti_volunteer_unit', state.callsign);
    });

    elements.btnAlertsToggle.addEventListener('click', () => {
        initAudio();
        state.alertsEnabled = !state.alertsEnabled;

        if (state.alertsEnabled) {
            elements.btnAlertsToggle.classList.add('active');
            elements.alertsToggleText.textContent = 'ALERTS ACTIVE';
            elements.btnAlertsToggle.querySelector('i').className = 'fa-solid fa-volume-high';
            // Play a soft test chime to confirm audio is unlocked
            playAlertChime();
            triggerHaptics();
        } else {
            elements.btnAlertsToggle.classList.remove('active');
            elements.alertsToggleText.textContent = 'ENABLE ALERTS';
            elements.btnAlertsToggle.querySelector('i').className = 'fa-solid fa-bell';
        }
    });

    if (elements.btnTestAlert) {
        elements.btnTestAlert.addEventListener('click', async () => {
            initAudio();
            if (!state.alertsEnabled) {
                elements.btnAlertsToggle.click();
            }
            try {
                elements.btnTestAlert.disabled = true;
                elements.btnTestAlert.style.opacity = '0.5';
                const res = await fetch('/api/v2/alerts/test', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                        case_id: 'case_0001',
                        case_name: 'Riya',
                        camera_name: 'Phone Cam',
                        location: 'Stage Exit',
                        similarity: 0.72,
                        decision: 'strong_match',
                    }),
                });
                const data = await res.json();
                if (data.success && data.alert) {
                    handleIncomingAlert(data.alert);
                }
            } catch (err) {
                console.warn('Test alert failed:', err);
            } finally {
                elements.btnTestAlert.disabled = false;
                elements.btnTestAlert.style.opacity = '1';
            }
        });
    }

    elements.filterBtns.forEach(btn => {
        btn.addEventListener('click', () => {
            elements.filterBtns.forEach(b => b.classList.remove('active'));
            btn.classList.add('active');
            state.currentFilter = btn.dataset.filter;
            renderAllAlerts();
        });
    });

    // Lightbox modal handlers
    elements.btnCloseLightbox.addEventListener('click', () => {
        elements.lightboxModal.classList.remove('active');
    });
    elements.lightboxModal.addEventListener('click', (e) => {
        if (e.target === elements.lightboxModal) {
            elements.lightboxModal.classList.remove('active');
        }
    });
}

function openLightbox(imgSrc) {
    if (!imgSrc) return;
    elements.lightboxImg.src = imgSrc;
    elements.lightboxModal.classList.add('active');
}

// ---------------------------------------------------------------------------
// Server-Sent Events (SSE) & Polling Hydration
// ---------------------------------------------------------------------------
async function fetchInitialAlerts() {
    try {
        const res = await fetch('/api/v2/alerts');
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();
        if (data.success && Array.isArray(data.alerts)) {
            data.alerts.forEach(al => {
                state.alerts.set(al.id, al);
            });
            renderAllAlerts();
            updateStats();
        }
    } catch (err) {
        console.warn('Could not fetch initial alerts:', err);
    }
}

function connectSSE() {
    if (state.sseSource) {
        state.sseSource.close();
        state.sseSource = null;
    }

    setConnectionStatus('connecting');

    const sse = new EventSource('/api/v2/alerts/stream');
    state.sseSource = sse;

    sse.onopen = () => {
        setConnectionStatus('live');
    };

    sse.addEventListener('alert', (event) => {
        try {
            const alertData = JSON.parse(event.data);
            handleIncomingAlert(alertData);
        } catch (err) {
            console.error('Error parsing SSE alert payload:', err);
        }
    });

    sse.addEventListener('ping', (event) => {
        state.lastHeartbeat = new Date();
        const timeStr = state.lastHeartbeat.toLocaleTimeString();
        elements.heartbeatInfo.textContent = `Heartbeat: ${timeStr}`;
        setConnectionStatus('live');
    });

    sse.onerror = (err) => {
        console.warn('SSE stream disconnected, reconnecting in 3s...', err);
        setConnectionStatus('reconnecting');
        sse.close();
        state.sseSource = null;
        if (!state.reconnectTimer) {
            state.reconnectTimer = setTimeout(() => {
                state.reconnectTimer = null;
                connectSSE();
            }, 3000);
        }
    };
}

function setConnectionStatus(status) {
    elements.connectionPill.className = 'status-pill';
    if (status === 'live') {
        elements.connectionPill.classList.add('live');
        elements.connectionText.textContent = 'LIVE SSE';
    } else if (status === 'reconnecting') {
        elements.connectionPill.classList.add('reconnecting');
        elements.connectionText.textContent = 'RECONNECTING';
    } else {
        elements.connectionText.textContent = 'CONNECTING';
    }
}

// ---------------------------------------------------------------------------
// Alert Processing & Rendering
// ---------------------------------------------------------------------------
function handleIncomingAlert(alert) {
    const isNew = !state.alerts.has(alert.id);
    state.alerts.set(alert.id, alert);

    if (isNew) {
        playAlertChime();
        triggerHaptics();
    }

    renderAllAlerts();
    updateStats();

    // Scroll to top if new alert
    if (isNew) {
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
}

function updateStats() {
    let active = 0;
    let found = 0;
    let resolved = 0;

    state.alerts.forEach(al => {
        if (al.status === 'new') {
            active++;
        } else if (al.status === 'found') {
            found++;
            resolved++;
        } else {
            resolved++;
        }
    });

    const total = state.alerts.size;

    elements.statActive.textContent = active;
    elements.statFound.textContent = found;
    elements.statTotal.textContent = total;

    elements.countAll.textContent = total;
    elements.countActive.textContent = active;
    elements.countResolved.textContent = resolved;
}

function renderAllAlerts() {
    const list = Array.from(state.alerts.values()).sort((a, b) => {
        const tA = new Date(a.created_at || 0).getTime();
        const tB = new Date(b.created_at || 0).getTime();
        return tB - tA; // newest first
    });

    const filtered = list.filter(al => {
        if (state.currentFilter === 'active') return al.status === 'new';
        if (state.currentFilter === 'resolved') return al.status !== 'new';
        return true;
    });

    // Remove existing cards
    const existingCards = elements.alertsFeed.querySelectorAll('.alert-card');
    existingCards.forEach(c => c.remove());

    if (filtered.length === 0) {
        elements.emptyState.style.display = 'flex';
        return;
    }

    elements.emptyState.style.display = 'none';

    filtered.forEach(alert => {
        const card = createAlertCardElement(alert);
        elements.alertsFeed.appendChild(card);
    });
}

function createAlertCardElement(alert) {
    const card = document.createElement('article');
    card.className = `alert-card status-${alert.status || 'new'}`;
    card.dataset.id = alert.id;

    const timeAgo = formatTimeAgo(alert.created_at);
    const decisionText = alert.decision ? alert.decision.replace('_', ' ').toUpperCase() : 'MATCH';
    const isStrong = alert.decision === 'strong_match';
    const simScore = typeof alert.similarity === 'number' ? alert.similarity.toFixed(2) : '--';
    const isActionable = alert.status === 'new';

    const snapshotSrc = alert.snapshot_url || '/assets/cctv_ui.png';
    const cropSrc = alert.crop_url || '/assets/result_ui.png';

    card.innerHTML = `
        <div class="card-media-hero" data-img="${snapshotSrc}">
            <img class="hero-snapshot" src="${snapshotSrc}" alt="Camera snapshot" loading="lazy">
            <div class="hero-overlay-top">
                <div class="hero-badges">
                    <span class="badge badge-match">MATCH</span>
                    <span class="badge badge-decision ${isStrong ? 'strong' : ''}">[ ${decisionText} ]</span>
                    <span class="badge badge-live">● LIVE</span>
                </div>
                <div class="card-time">${timeAgo}</div>
            </div>
            <div class="face-inset" data-img="${cropSrc}">
                <img src="${cropSrc}" alt="Face crop" loading="lazy">
                <span class="face-inset-label">FACE CROP</span>
            </div>
        </div>

        <div class="card-info-strip">
            <div class="info-left">
                <div class="case-name">${escapeHtml(alert.case_name || 'Missing Person')}</div>
                <div class="case-location">
                    <i class="fa-solid fa-location-dot"></i>
                    <span>${escapeHtml(alert.location || 'Camera Area')} · ${escapeHtml(alert.camera_name || alert.camera_id || 'CCTV')}</span>
                </div>
            </div>
            <div class="sim-pill ${isStrong ? 'strong' : ''}">${simScore}</div>
        </div>

        ${isActionable ? `
            <div class="card-actions">
                <button class="btn-action btn-ack" data-action="ack" title="Acknowledge sighting">
                    <i class="fa-solid fa-check"></i> ACK
                </button>
                <button class="btn-action btn-found" data-action="found" title="Person safely located">
                    <i class="fa-solid fa-shield-heart"></i> FOUND!
                </button>
                <button class="btn-action btn-false" data-action="false_alarm" title="False alarm">
                    <i class="fa-solid fa-xmark"></i> DISMISS
                </button>
            </div>
        ` : `
            <div class="card-status-strip">
                <div class="status-msg ${alert.status}">
                    ${getStatusIcon(alert.status)}
                    <span>${getStatusText(alert)}</span>
                </div>
                <div style="font-size: 8px; color: var(--muted); font-family: monospace;">
                    ${alert.acked_at ? new Date(alert.acked_at).toLocaleTimeString() : ''}
                </div>
            </div>
        `}
    `;

    // Lightbox — hero snapshot click
    const heroEl = card.querySelector('.card-media-hero');
    if (heroEl) {
        heroEl.addEventListener('click', (e) => {
            // Don't trigger if clicking the face inset
            if (e.target.closest('.face-inset')) return;
            openLightbox(snapshotSrc);
        });
    }

    // Lightbox — face inset click
    const faceEl = card.querySelector('.face-inset');
    if (faceEl) {
        faceEl.addEventListener('click', (e) => {
            e.stopPropagation();
            openLightbox(cropSrc);
        });
    }

    // Action button handlers
    if (isActionable) {
        card.querySelectorAll('.btn-action').forEach(btn => {
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const action = btn.dataset.action;
                handleAlertAction(alert.id, action);
            });
        });
    }

    return card;
}

function getStatusIcon(status) {
    if (status === 'found') return '<i class="fa-solid fa-circle-check"></i>';
    if (status === 'ack') return '<i class="fa-solid fa-user-check"></i>';
    return '<i class="fa-solid fa-ban"></i>';
}

function getStatusText(alert) {
    const by = alert.acked_by || 'Volunteer';
    if (alert.status === 'found') return `PERSON FOUND! Marked by ${by}`;
    if (alert.status === 'ack') return `Acknowledged by ${by}`;
    if (alert.status === 'false_alarm') return `Dismissed as false alarm by ${by}`;
    return alert.status.toUpperCase();
}

// ---------------------------------------------------------------------------
// Ack / Found Actions (TRD §6.6 Contract)
// ---------------------------------------------------------------------------
async function handleAlertAction(alertId, action) {
    const card = document.querySelector(`.alert-card[data-id="${alertId}"]`);
    if (card) {
        const buttons = card.querySelectorAll('.btn-action');
        buttons.forEach(b => {
            b.disabled = true;
            b.style.opacity = '0.5';
        });
    }

    try {
        const res = await fetch(`/api/v2/alerts/${alertId}/ack`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                by: state.callsign,
                action: action,
            }),
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        if (data.success && data.alert) {
            state.alerts.set(alertId, data.alert);
            renderAllAlerts();
            updateStats();

            // Extra celebratory vibration if person was found!
            if (action === 'found' && 'vibrate' in navigator) {
                navigator.vibrate([100, 50, 100, 50, 300]);
            }
        }
    } catch (err) {
        console.error('Failed to submit alert action:', err);
        alert(`Action failed: ${err.message}`);
        if (card) {
            const buttons = card.querySelectorAll('.btn-action');
            buttons.forEach(b => {
                b.disabled = false;
                b.style.opacity = '1';
            });
        }
    }
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------
function formatTimeAgo(isoString) {
    if (!isoString) return 'RECENT';
    const date = new Date(isoString);
    const now = new Date();
    const diffSec = Math.floor((now - date) / 1000);

    if (isNaN(diffSec) || diffSec < 5) return 'JUST NOW';
    if (diffSec < 60) return `${diffSec}s AGO`;
    const diffMin = Math.floor(diffSec / 60);
    if (diffMin < 60) return `${diffMin}m AGO`;
    return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

function escapeHtml(str) {
    if (!str) return '';
    return String(str)
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
window.addEventListener('DOMContentLoaded', () => {
    setupControls();
    fetchInitialAlerts();
    connectSSE();

    // Periodic background sync fallback (every 3s) ensuring zero missed alerts
    setInterval(async () => {
        try {
            const res = await fetch('/api/v2/alerts');
            if (res.ok) {
                const data = await res.json();
                if (data.success && Array.isArray(data.alerts)) {
                    let hasNew = false;
                    data.alerts.forEach(al => {
                        if (!state.alerts.has(al.id)) {
                            state.alerts.set(al.id, al);
                            hasNew = true;
                        }
                    });
                    if (hasNew) {
                        playAlertChime();
                        triggerHaptics();
                        renderAllAlerts();
                        updateStats();
                    }
                }
            }
        } catch (_) {}
    }, 3000);

    // Periodic time label updater
    setInterval(() => {
        document.querySelectorAll('.alert-card').forEach(card => {
            const id = card.dataset.id;
            const al = state.alerts.get(id);
            if (al && al.created_at) {
                const timeEl = card.querySelector('.card-time');
                if (timeEl) timeEl.textContent = formatTimeAgo(al.created_at);
            }
        });
    }, 10000);
});
