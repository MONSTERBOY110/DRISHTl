const MOCK_MODE = new URLSearchParams(window.location.search).get("mock") === "1";

async function jsonFetch(url, options = {}) {
    const response = await fetch(url, options);
    const contentType = response.headers.get("content-type") || "";
    if (!contentType.includes("application/json")) {
        throw new Error("Server returned a non-JSON response");
    }
    const data = await response.json();
    if (!response.ok || data.success === false) {
        throw new Error(data.message || `Request failed (${response.status})`);
    }
    return data;
}

export async function getHealth() {
    if (MOCK_MODE) return jsonFetch("./mock/health.json");
    return jsonFetch("/api/v2/health");
}

export async function getCameras() {
    if (MOCK_MODE) return jsonFetch("./mock/cameras.json");
    return jsonFetch("/api/v2/cameras");
}

export async function getIndexStatus() {
    if (MOCK_MODE) return jsonFetch("./mock/index-status.json");
    return jsonFetch("/api/v2/index/status");
}

export async function rebuildIndex(force = false) {
    if (MOCK_MODE) return { success: true, state: "running" };
    return jsonFetch("/api/v2/index/rebuild", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force })
    });
}

export async function searchPerson(formData) {
    if (MOCK_MODE) return jsonFetch("./mock/search.json");
    return jsonFetch("/api/v2/search", {
        method: "POST",
        body: formData
    });
}

export async function transcribeAudio(formData) {
    if (MOCK_MODE) {
        return {
            success: true,
            engine: "mock",
            transcript: "child in red shirt and blue jeans",
            description_en: "child, red shirt, blue jeans",
            parsed: { upper: ["red"], lower: ["blue"] },
            message: "Description transcribed."
        };
    }
    return jsonFetch("/api/v2/intake/transcribe", {
        method: "POST",
        body: formData
    });
}

export { MOCK_MODE };
