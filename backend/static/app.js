// --- Three.js Background Implementation ---
let scene, camera, renderer, globe, particles;

function initThree() {
    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(75, window.innerWidth / window.innerHeight, 0.1, 1000);
    
    renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setSize(window.innerWidth, window.innerHeight);
    renderer.setPixelRatio(window.devicePixelRatio);
    document.getElementById('canvas-container').appendChild(renderer.domElement);

    // Add main object (Torus Knot for futuristic tech look)
    const geometry = new THREE.TorusKnotGeometry(10, 3, 100, 16);
    const material = new THREE.MeshBasicMaterial({ 
        color: 0x00f2ff, 
        wireframe: true, 
        transparent: true, 
        opacity: 0.2 
    });
    globe = new THREE.Mesh(geometry, material);
    scene.add(globe);

    // Add particle field
    const pGeometry = new THREE.BufferGeometry();
    const pCount = 2000;
    const vertices = [];
    for (let i = 0; i < pCount; i++) {
        vertices.push(
            Math.random() * 600 - 300,
            Math.random() * 600 - 300,
            Math.random() * 600 - 300
        );
    }
    pGeometry.setAttribute('position', new THREE.Float32BufferAttribute(vertices, 3));
    const pMaterial = new THREE.PointsMaterial({ color: 0xffffff, size: 0.5, transparent: true, opacity: 0.5 });
    particles = new THREE.Points(pGeometry, pMaterial);
    scene.add(particles);

    camera.position.z = 50;
}

let rotationSpeed = 0.005;

function animate() {
    requestAnimationFrame(animate);
    
    globe.rotation.y += rotationSpeed;
    globe.rotation.x += rotationSpeed * 0.5;
    
    particles.rotation.y += 0.0005;
    
    renderer.render(scene, camera);
}

window.addEventListener('resize', () => {
    camera.aspect = window.innerWidth / window.innerHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(window.innerWidth, window.innerHeight);
});

// --- UI Logic & API Integration ---

const auditBtn = document.getElementById('audit-btn');
const urlInput = document.getElementById('video-url');
const statusDisplay = document.getElementById('status-display');
const statusText = document.getElementById('status-text');
const resultsPanel = document.getElementById('results-panel');
const reportText = document.getElementById('report-text');
const issuesContainer = document.getElementById('issues-container');
const statusBadge = document.getElementById('status-badge');

async function startAudit() {
    const url = urlInput.value.trim();
    if (!url) {
        alert("Please enter a valid video URL.");
        return;
    }

    // Reset UI
    auditBtn.disabled = true;
    statusDisplay.classList.remove('status-hidden');
    resultsPanel.classList.add('results-hidden');
    statusText.innerText = "Analyzing Video (Rekognition & Transcribe)...";
    
    // speed up animation for "working" state
    rotationSpeed = 0.02;
    globe.material.opacity = 0.6;

    try {
        const response = await fetch('/api/audit', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_url: url })
        });

        const data = await response.json();

        if (response.ok && (data.success || data.status === "success")) {
            displayResults(data);
        } else {
            handleError(data);
        }
    } catch (err) {
        console.error(err);
        handleError({ errors: ["Network error or server timeout. Check console."] });
    } finally {
        auditBtn.disabled = false;
        statusDisplay.classList.add('status-hidden');
        rotationSpeed = 0.005;
        globe.material.opacity = 0.2;
    }
}

function displayResults(data) {
    resultsPanel.classList.remove('results-hidden');
    reportText.innerText = data.report || "No summary provided.";
    
    // Set badge status
    statusBadge.innerText = data.status.toUpperCase();
    statusBadge.className = 'badge ' + (data.status === 'success' ? 'success' : 'warning');

    // Clear and render issues
    issuesContainer.innerHTML = '';
    const issues = data.issues || [];
    
    if (issues.length === 0) {
        issuesContainer.innerHTML = '<div class="issue-card">No compliance issues identified.</div>';
    } else {
        issues.forEach(issue => {
            const card = document.createElement('div');
            card.className = `issue-card ${issue.severity.toLowerCase()}`;
            card.innerHTML = `
                <div class="issue-title">
                    <span>${issue.category || 'Compliance Check'}</span>
                    <span class="issue-severity">${issue.severity}</span>
                </div>
                <div class="issue-desc">${issue.description}</div>
                <div class="issue-desc" style="margin-top:0.5rem; font-style:italic; opacity:0.6;">
                    Suggestion: ${issue.suggestion || 'No specific action required.'}
                </div>
            `;
            issuesContainer.appendChild(card);
        });
    }
    
    // Scroll to results
    resultsPanel.scrollIntoView({ behavior: 'smooth' });
}

function handleError(data) {
    let msg = "Unknown error";
    if (data.errors && data.errors.length > 0) {
        msg = data.errors.join(', ');
    } else if (data.detail) {
        msg = data.detail;
    }
    alert("Audit Failed: " + msg);
}

auditBtn.addEventListener('click', startAudit);

// Initialize
initThree();
animate();
