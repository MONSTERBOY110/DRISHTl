import { MOCK_MODE, getCameras, getHealth, getIndexStatus, rebuildIndex, searchPerson, transcribeAudio } from "./api.js?v=2.2";

const cameraGrid = document.getElementById("cameraGrid");
const healthLine = document.getElementById("healthLine");
const reindexButton = document.getElementById("reindexButton");
const uploadArea = document.getElementById("uploadArea");
const photoInput = document.getElementById("photoInput");
const uploadEmpty = document.getElementById("uploadEmpty");
const uploadPreview = document.getElementById("uploadPreview");
const previewImage = document.getElementById("previewImage");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const descriptionInput = document.getElementById("descriptionInput");
const micButton = document.getElementById("micButton");
const cameraFilter = document.getElementById("cameraFilter");
const findButton = document.getElementById("findButton");
const statusText = document.getElementById("statusText");
const timingText = document.getElementById("timingText");
const verifiedList = document.getElementById("verifiedList");
const leadList = document.getElementById("leadList");
const venueMap = document.getElementById("venueMap");
const trailList = document.getElementById("trailList");
const lastSeenText = document.getElementById("lastSeenText");

let selectedFile = null;
let cameras = [];
let speechRecognition = null;

function setStatus(message, type = "") {
    statusText.textContent = message;
    statusText.className = `status-text ${type}`.trim();
}

function emptyState(text) {
    return `<div class="empty-state">${text}</div>`;
}

function formatSeconds(seconds) {
    if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return "";
    const total = Math.max(0, Math.floor(Number(seconds)));
    const h = String(Math.floor(total / 3600)).padStart(2, "0");
    const m = String(Math.floor((total % 3600) / 60)).padStart(2, "0");
    const s = String(total % 60).padStart(2, "0");
    return `${h}:${m}:${s}`;
}

function shortWallTime(value) {
    if (!value) return "";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return value;
    return date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

function renderCameras() {
    if (!cameras.length) {
        cameraGrid.innerHTML = emptyState("No cameras found. Add videos to CCTVS/ or use ?mock=1.");
        cameraFilter.innerHTML = "";
        return;
    }

    cameraGrid.innerHTML = cameras.map((camera) => `
        <article class="camera-card" data-camera-id="${camera.id}">
            ${camera.video_url && !MOCK_MODE
                ? `<video src="${camera.video_url}" muted autoplay loop playsinline></video>`
                : `<div class="camera-placeholder"></div>`}
            <div class="camera-overlay">
                <div class="camera-chip">${camera.name || camera.id}</div>
                <div class="match-chip">MATCH</div>
            </div>
        </article>
    `).join("");

    cameraFilter.innerHTML = cameras.map((camera) => `
        <label class="camera-toggle">
            <input type="checkbox" value="${camera.id}" checked>
            <span>${camera.name || camera.id}</span>
        </label>
    `).join("");

    document.querySelectorAll(".camera-card video").forEach((video) => {
        const play = video.play();
        if (play && typeof play.catch === "function") play.catch(() => {});
    });
}

function markMatchedCameras(result) {
    document.querySelectorAll(".camera-card").forEach((card) => {
        card.classList.remove("match");
        const chip = card.querySelector(".match-chip");
        if (chip) chip.textContent = "MATCH";
    });

    [...(result.verified || []), ...(result.leads || [])].forEach((sighting) => {
        const card = document.querySelector(`.camera-card[data-camera-id="${CSS.escape(sighting.camera_id)}"]`);
        if (!card) return;
        card.classList.add("match");
        const chip = card.querySelector(".match-chip");
        if (chip) chip.textContent = sighting.face ? "VERIFIED" : "LEAD";
    });
}

function renderCard(item, type) {
    const image = item.best_frame || item.best_crop || "";
    const time = item.t_start_hms || formatSeconds(item.t_start);
    const label = type === "verified" ? "Verified" : "Lead (unverified)";
    const chips = [];
    if (item.face && item.face.decision) chips.push(item.face.decision.replace("_", " "));
    if (item.face && item.face.best_similarity !== undefined) chips.push(`face ${Number(item.face.best_similarity).toFixed(2)}`);
    if (item.appearance && item.appearance.z !== undefined) chips.push(`z ${Number(item.appearance.z).toFixed(2)}`);
    if (item.appearance && Array.isArray(item.appearance.attribute_hits)) chips.push(...item.appearance.attribute_hits);
    if (!chips.length && item.evidence) chips.push(item.evidence);

    return `
        <article class="result-card ${type}">
            ${image ? `<img src="${image}" alt="${label} crop" onerror="this.style.opacity='0.35'">` : `<div class="empty-state">No image</div>`}
            <div class="result-body">
                <div class="result-title">
                    <span>${item.camera_name || item.camera_id || "Unknown camera"}</span>
                    <span>${time}</span>
                </div>
                <div class="result-meta">
                    <span class="pill label">${label}</span>
                    ${item.location ? `<span class="pill">${item.location}</span>` : ""}
                    ${item.clip_url ? `<a class="pill" href="${item.clip_url}" target="_blank" rel="noopener">Open clip</a>` : ""}
                </div>
                <div class="why-chips">
                    ${chips.map((chip) => `<span class="pill">${chip}</span>`).join("")}
                </div>
            </div>
        </article>
    `;
}

function renderTrail(result) {
    const trail = result.trail || [];
    if (!trail.length) {
        trailList.innerHTML = emptyState("No trail yet.");
        venueMap.innerHTML = "";
        lastSeenText.textContent = "No last seen point.";
        return;
    }

    const last = result.last_seen || trail[trail.length - 1];
    lastSeenText.textContent = `Last seen ${shortWallTime(last.wall_time) || formatSeconds(last.t_start)} at ${last.location || last.camera_name || last.camera_id}.`;
    trailList.innerHTML = trail.map((point, index) => `
        <li>
            <strong>${index + 1}. ${point.location || point.camera_name || point.camera_id}</strong><br>
            ${shortWallTime(point.wall_time) || formatSeconds(point.t_start)} - ${point.evidence}
        </li>
    `).join("");

    const points = trail
        .filter((point) => point.map && Number.isFinite(Number(point.map.x)) && Number.isFinite(Number(point.map.y)))
        .map((point, index) => ({ ...point, index, x: Number(point.map.x) * 100, y: Number(point.map.y) * 100 }));

    const lines = points.slice(1).map((point, index) => {
        const prev = points[index];
        return `<line x1="${prev.x}" y1="${prev.y}" x2="${point.x}" y2="${point.y}" stroke="#ef233c" stroke-width="1.4" />`;
    }).join("");
    const pins = points.map((point) => `
        <g>
            <circle cx="${point.x}" cy="${point.y}" r="4.5" fill="#fff3d7" stroke="#ef233c" stroke-width="1.5"></circle>
            <text x="${point.x}" y="${point.y + 1.5}" text-anchor="middle" font-size="4" fill="#252734" font-weight="800">${point.index + 1}</text>
        </g>
    `).join("");

    venueMap.innerHTML = `<rect x="4" y="4" width="92" height="92" rx="4" fill="#12141d" stroke="rgba(154,164,184,.45)" />${lines}${pins}`;
}

function renderSearchResult(result) {
    markMatchedCameras(result);
    timingText.textContent = result.timings
        ? `${result.timings.search_ms} ms - ${result.timings.tracks_scanned} tracks scanned`
        : (result.message || "Search complete");
    verifiedList.innerHTML = (result.verified || []).length
        ? result.verified.map((item) => renderCard(item, "verified")).join("")
        : emptyState("No face-verified match.");
    leadList.innerHTML = (result.leads || []).length
        ? result.leads.map((item) => renderCard(item, "lead")).join("")
        : emptyState("No appearance-only lead.");
    renderTrail(result);
}

function handleFile(file) {
    if (!file.type.startsWith("image/")) {
        setStatus("Please select an image file.", "error");
        return;
    }
    selectedFile = file;
    previewImage.src = URL.createObjectURL(file);
    fileName.textContent = file.name;
    fileSize.textContent = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    uploadEmpty.classList.add("hidden");
    uploadPreview.classList.remove("hidden");
    setStatus("Photo ready.", "success");
}

async function submitSearch() {
    const description = descriptionInput.value.trim();
    if (!selectedFile && !description) {
        setStatus("Add a photo, a description, or both.", "error");
        return;
    }

    const selectedCameras = Array.from(cameraFilter.querySelectorAll("input:checked")).map((input) => input.value);
    const formData = new FormData();
    if (selectedFile) formData.append("photo", selectedFile);
    if (description) formData.append("description", description);
    if (selectedCameras.length && selectedCameras.length !== cameras.length) {
        formData.append("cameras", selectedCameras.join(","));
    }

    findButton.disabled = true;
    setStatus("Searching selected cameras...", "");
    try {
        const result = await searchPerson(formData);
        renderSearchResult(result);
        setStatus(result.message || "Search complete.", result.decision === "no_reliable_match" ? "" : "success");
    } catch (error) {
        setStatus(error.message, "error");
    } finally {
        findButton.disabled = false;
    }
}

let mediaRecorder = null;
let audioChunks = [];

function setupSpeechInput() {
    let isRecording = false;

    // Web Speech API fallback for local recognition
    function startWebSpeechFallback() {
        const Recognition = window.SpeechRecognition || window.webkitSpeechRecognition;
        if (!Recognition) {
            setStatus("Microphone access and speech recognition not supported in this browser.", "error");
            return;
        }
        speechRecognition = new Recognition();
        speechRecognition.lang = "en-IN";
        speechRecognition.interimResults = true;
        speechRecognition.continuous = false;
        speechRecognition.onstart = () => {
            micButton.classList.add("listening");
            setStatus("Listening (browser local)...", "");
        };
        speechRecognition.onresult = (event) => {
            let transcript = "";
            for (let i = event.resultIndex; i < event.results.length; i += 1) {
                transcript += event.results[i][0].transcript;
            }
            descriptionInput.value = transcript.trim();
        };
        speechRecognition.onerror = (event) => setStatus(`Voice input failed: ${event.error}`, "error");
        speechRecognition.onend = () => {
            micButton.classList.remove("listening");
            if (descriptionInput.value.trim()) setStatus("Description captured. Ready to search.", "success");
        };
        try {
            speechRecognition.start();
        } catch (e) {
            speechRecognition.stop();
        }
    }

    async function sendAudioForTranscription(blob) {
        micButton.classList.remove("listening");
        setStatus("Transcribing voice description...", "");
        try {
            const formData = new FormData();
            formData.append("audio", blob, "recording.webm");
            formData.append("lang", "auto");
            const res = await transcribeAudio(formData);
            if (res && res.description_en) {
                descriptionInput.value = res.description_en;
                setStatus(`Captured: "${res.transcript || res.description_en}"`, "success");
            } else {
                setStatus("Could not detect description from audio.", "error");
            }
        } catch (err) {
            setStatus(`Server transcription error: ${err.message}. Trying browser speech...`, "error");
            startWebSpeechFallback();
        }
    }

    micButton.addEventListener("click", async () => {
        if (isRecording) {
            // Stop recording
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
            }
            isRecording = false;
            return;
        }

        // Check if MediaDevices is supported
        if (navigator.mediaDevices && navigator.mediaDevices.getUserMedia) {
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                audioChunks = [];
                const mimeType = MediaRecorder.isTypeSupported("audio/webm") ? "audio/webm" : "";
                mediaRecorder = mimeType ? new MediaRecorder(stream, { mimeType }) : new MediaRecorder(stream);

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data && e.data.size > 0) audioChunks.push(e.data);
                };

                mediaRecorder.onstop = () => {
                    stream.getTracks().forEach((track) => track.stop());
                    const blob = new Blob(audioChunks, { type: mediaRecorder.mimeType || "audio/webm" });
                    sendAudioForTranscription(blob);
                };

                mediaRecorder.start();
                isRecording = true;
                micButton.classList.add("listening");
                setStatus("Recording... Speak your description, then click mic to stop.", "");
            } catch (err) {
                console.warn("MediaRecorder permission/error:", err);
                startWebSpeechFallback();
            }
        } else {
            startWebSpeechFallback();
        }
    });
}

async function refreshIndexStatus() {
    try {
        const status = await getIndexStatus();
        if (status.state === "running") {
            healthLine.textContent = `Indexing ${status.done}/${status.total} - ${status.current_camera || ""}`;
        }
    } catch (error) {
        console.warn(error);
    }
}

async function init() {
    verifiedList.innerHTML = emptyState("No face-verified match yet.");
    leadList.innerHTML = emptyState("No appearance lead yet.");
    trailList.innerHTML = emptyState("Run a search to build the trail.");

    try {
        const [health, cameraData] = await Promise.all([getHealth(), getCameras()]);
        healthLine.textContent = `${MOCK_MODE ? "Mock mode" : "Live"} - ${health.index.tracks} tracks - ${health.models.appearance}`;
        cameras = cameraData.cameras || [];
        renderCameras();
    } catch (error) {
        healthLine.textContent = "Backend unavailable";
        setStatus(error.message, "error");
    }
}

function clearUploadedPhoto() {
    if (previewImage.src && previewImage.src.startsWith("blob:")) {
        URL.revokeObjectURL(previewImage.src);
    }
    previewImage.src = "";
    selectedFile = null;
    photoInput.value = "";
    uploadPreview.classList.add("hidden");
    uploadEmpty.classList.remove("hidden");
}

const removePhotoButton = document.getElementById("removePhotoButton");
if (removePhotoButton) {
    removePhotoButton.addEventListener("click", (event) => {
        event.stopPropagation();
        event.preventDefault();
        clearUploadedPhoto();
        setStatus("Photo removed.", "");
    });
}

uploadArea.addEventListener("click", () => photoInput.click());
uploadArea.addEventListener("dragover", (event) => {
    event.preventDefault();
    uploadArea.classList.add("dragover");
});
uploadArea.addEventListener("dragleave", () => uploadArea.classList.remove("dragover"));
uploadArea.addEventListener("drop", (event) => {
    event.preventDefault();
    uploadArea.classList.remove("dragover");
    const file = event.dataTransfer.files[0];
    if (file) handleFile(file);
});
photoInput.addEventListener("change", (event) => {
    const file = event.target.files[0];
    if (file) handleFile(file);
});
findButton.addEventListener("click", submitSearch);
reindexButton.addEventListener("click", async () => {
    try {
        await rebuildIndex(true);
        setStatus("Index rebuild started.", "success");
        refreshIndexStatus();
    } catch (error) {
        setStatus(error.message, "error");
    }
});

setupSpeechInput();
init();
setInterval(refreshIndexStatus, 3000);
