// DOM Elements
const statusText = document.getElementById("status-text");
const statusSubtext = document.getElementById("status-subtext");
const volumeBar = document.getElementById("volume-bar");
const radioButtons = document.querySelectorAll('input[name="personality"]');
const userSubtitle = document.getElementById("user-subtitle");
const aiSubtitle = document.getElementById("ai-subtitle");
const userEmotionTag = document.getElementById("user-emotion-tag");
const gfEmotionTag = document.getElementById("gf-emotion-tag");
const moodEnergyTag = document.getElementById("mood-energy-tag");
const moodModeTag = document.getElementById("mood-mode-tag");
const chatHistory = document.getElementById("chat-history");

// WebSocket Connection
let ws;
const connectWebSocket = () => {
    ws = new WebSocket(`ws://${location.host}/ws`);

    ws.onopen = () => {
        statusSubtext.textContent = "Connected to A.I. Core";
        console.log("WebSocket connected");
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        updateUI(data);
    };

    ws.onclose = () => {
        statusText.textContent = "Offline";
        statusText.className = "status-idle";
        statusSubtext.textContent = "Connection lost. Reconnecting...";
        setTimeout(connectWebSocket, 2000);
    };
};

// UI State Updates
let currentVolume = 0.0;
let currentStatus = "idle";
let lastUserText = '';
let lastAiText = '';

const pushToHistory = (uText, aText) => {
    if (!uText && !aText) return;
    
    const turnDiv = document.createElement("div");
    turnDiv.className = "history-turn";
    
    if (uText) {
        const uP = document.createElement("p");
        uP.className = "history-user";
        uP.textContent = `"${uText}"`;
        turnDiv.appendChild(uP);
    }
    
    if (aText) {
        const aP = document.createElement("p");
        aP.className = "history-ai";
        aP.textContent = aText;
        turnDiv.appendChild(aP);
    }
    
    chatHistory.appendChild(turnDiv);
    chatHistory.scrollTop = chatHistory.scrollHeight;
};

const updateUI = (data) => {
    currentVolume = data.volume;
    
    // Subtitles Update & History Logging
    if (data.user_text !== undefined || data.response_text !== undefined) {
        // If text is cleared, push the previously completed text into history log
        if (data.user_text === "" && data.response_text === "" && (lastUserText || lastAiText)) {
            pushToHistory(lastUserText, lastAiText);
            lastUserText = "";
            lastAiText = "";
        }

        if (data.user_text !== undefined) {
            userSubtitle.textContent = data.user_text ? `"${data.user_text}"` : "";
            if (data.user_text) lastUserText = data.user_text;
        }
        if (data.response_text !== undefined) {
            aiSubtitle.textContent = data.response_text || "";
            if (data.response_text) lastAiText = data.response_text;
        }
    }
    
    // Stats Update
    if (data.emotion !== undefined && userEmotionTag) {
        userEmotionTag.textContent = data.emotion;
    }
    if (data.gf_emotion !== undefined && gfEmotionTag) {
        gfEmotionTag.textContent = data.gf_emotion;
    }
    if (data.mood_energy !== undefined && moodEnergyTag) {
        moodEnergyTag.textContent = data.mood_energy;
    }
    if (data.mood_mode !== undefined && moodModeTag) {
        moodModeTag.textContent = data.mood_mode;
    }
    
    // Smooth width update
    volumeBar.style.width = `${currentVolume * 100}%`;

    // Status update
    if (currentStatus !== data.status) {
        currentStatus = data.status;
        statusText.textContent = translateStatus(currentStatus);
        statusText.className = `status-${currentStatus}`;
        
        if (currentStatus === "listening") {
            statusSubtext.textContent = "Hearing you...";
        } else if (currentStatus === "thinking") {
            statusSubtext.textContent = "Processing logic...";
        } else if (currentStatus === "speaking") {
            statusSubtext.textContent = "Synthesizing vocal response...";
        } else {
            statusSubtext.textContent = "Awaiting input...";
        }
    }

    // Sync radio buttons if changed remotely
    const activeRadio = document.querySelector(`input[name="personality"][value="${data.personality}"]`);
    if (activeRadio && !activeRadio.checked) {
        activeRadio.checked = true;
    }
};

const translateStatus = (status) => {
    const map = {
        "idle": "Idle",
        "listening": "Listening",
        "recording": "Recording",
        "thinking": "Thinking",
        "speaking": "Speaking"
    };
    return map[status] || status;
};

// Personality Selection Handling
radioButtons.forEach(radio => {
    radio.addEventListener('change', async (e) => {
        const personality = e.target.value;
        try {
            await fetch('/api/personality', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ personality })
            });
            console.log(`Personality changed to ${personality}`);
        } catch (error) {
            console.error("Failed to change personality", error);
        }
    });
});

// ─── Three.js 3D Visualizer ───────────────────────────────────────────────

const container = document.getElementById("canvas-container");
const scene = new THREE.Scene();

// Camera
const camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
camera.position.z = 5;

// WebGL Renderer
const renderer = new THREE.WebGLRenderer({ alpha: true, antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
container.appendChild(renderer.domElement);

// Create the 3D Object (Icosahedron)
const geometry = new THREE.IcosahedronGeometry(1.5, 2);

// Material: wireframe glowing look
const material = new THREE.MeshBasicMaterial({ 
    color: 0x00ffcc, 
    wireframe: true,
    transparent: true,
    opacity: 0.6
});

const coreMesh = new THREE.Mesh(geometry, material);
scene.add(coreMesh);

// Add inner core
const innerGeo = new THREE.IcosahedronGeometry(1.2, 1);
const innerMat = new THREE.MeshBasicMaterial({ color: 0x0088ff, wireframe: false });
const innerMesh = new THREE.Mesh(innerGeo, innerMat);
scene.add(innerMesh);

// Handle Window Resize
window.addEventListener('resize', () => {
    renderer.setSize(window.innerWidth, window.innerHeight);
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
});

// Animation Loop
const animate = () => {
    requestAnimationFrame(animate);

    // Baseline rotation
    const baseSpeed = 0.002;
    
    // React to volume (volume 0.0 - 1.0)
    // Scale the mesh up when volume is high
    const targetScale = 1.0 + (currentVolume * 0.8);
    
    // Smooth lerp scale
    coreMesh.scale.setLength(THREE.MathUtils.lerp(coreMesh.scale.x, targetScale, 0.1));
    innerMesh.scale.setLength(THREE.MathUtils.lerp(innerMesh.scale.x, targetScale * 0.9, 0.1));

    // Dynamic rotation speed based on state and volume
    let rotSpeed = baseSpeed;
    if (currentStatus === "thinking") rotSpeed = 0.02; // Spin fast while thinking
    if (currentVolume > 0) rotSpeed += (currentVolume * 0.05);

    coreMesh.rotation.y += rotSpeed;
    coreMesh.rotation.x += rotSpeed * 0.8;
    
    innerMesh.rotation.y -= rotSpeed * 0.5;
    innerMesh.rotation.z += rotSpeed * 0.3;

    // Change color based on status
    if (currentStatus === "listening") {
        material.color.setHex(0x00ffcc);
        innerMat.color.setHex(0x0088ff);
    } else if (currentStatus === "thinking") {
        material.color.setHex(0xff00ff);
        innerMat.color.setHex(0x8800ff);
    } else if (currentStatus === "speaking") {
        material.color.setHex(0x0088ff);
        innerMat.color.setHex(0x00ffcc);
    } else {
        material.color.setHex(0x555555);
        innerMat.color.setHex(0x222222);
    }

    renderer.render(scene, camera);
};

// Fade out HUD on scroll
window.addEventListener('scroll', () => {
    const scrollY = window.scrollY;
    const uiLayer = document.getElementById('ui-layer');
    if (uiLayer) {
        // Fade out entirely by 400px scroll
        const opacity = Math.max(0, 1 - (scrollY / 400));
        uiLayer.style.opacity = opacity;
        
        // Disable pointer events when invisible
        if (opacity < 0.1) {
            uiLayer.style.pointerEvents = 'none';
        } else {
            // Restore pointer events depending on structure
            uiLayer.style.pointerEvents = 'none'; // The base structure keeps it none unless on controls
        }
    }
});

// Copy Link Handler
const copyBtn = document.getElementById("copy-btn");
if (copyBtn) {
    copyBtn.addEventListener("click", () => {
        const copyInput = document.getElementById("share-link-input");
        copyInput.select();
        document.execCommand("copy");
        copyBtn.innerHTML = '<i class="fa-solid fa-check"></i>';
        setTimeout(() => {
            copyBtn.innerHTML = '<i class="fa-regular fa-copy"></i>';
        }, 2000);
    });
}

// Start
connectWebSocket();
animate();
