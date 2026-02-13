/**
 * MuzikRE — Where Sound Meets Science
 * Frontend: Upload+Auto-Predict, Manual Predict, Player, Visualizer, Particles
 */

// ===== Constants =====
const FEAT_ICONS = { duration_min: '⏱️', tempo_bpm: '🥁', energy: '⚡', danceability: '💃', loudness_db: '🔊' };
const FEAT_NAMES = { duration_min: 'Duration', tempo_bpm: 'Tempo', energy: 'Energy', danceability: 'Danceability', loudness_db: 'Loudness' };

// ===== Audio State =====
let audio = new Audio();
let audioCtx = null;
let analyser = null;
let source = null;
let dataArr = null;
let isPlaying = false;
let animId = null;
let currentFile = null;   // currently uploaded file (for re-predict)

// ===== Init =====
document.addEventListener('DOMContentLoaded', () => {
    initUploadSection();
    initPlayer();
    initSliders();
    initManualForm();
    initParticles();
    initScrollReveal();
    loadModelInfo();
});

// ============================================================
//  SECTION 1: UPLOAD & AUTO-PREDICT
// ============================================================
function initUploadSection() {
    const dz = document.getElementById('drop-zone');
    const fileInput = document.getElementById('audio-file-input');
    const removeBtn = document.getElementById('file-remove');
    const reBtn = document.getElementById('btn-repredict');

    // Click drop zone → open file picker
    dz.addEventListener('click', e => {
        if (!e.target.closest('.dz-remove')) fileInput.click();
    });

    // File selected
    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) handleFileSelected(fileInput.files[0]);
    });

    // Drag & drop
    dz.addEventListener('dragover', e => { e.preventDefault(); dz.classList.add('drag-over'); });
    dz.addEventListener('dragleave', e => { e.preventDefault(); dz.classList.remove('drag-over'); });
    dz.addEventListener('drop', e => {
        e.preventDefault(); dz.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) handleFileSelected(e.dataTransfer.files[0]);
    });

    // Remove file
    removeBtn.addEventListener('click', e => {
        e.stopPropagation();
        clearUploadedFile();
    });

    // Re-predict button
    reBtn.addEventListener('click', () => {
        if (currentFile) runAutoPredict(currentFile);
    });
}

function handleFileSelected(file) {
    const allowed = ['mp3', 'wav', 'ogg', 'flac', 'm4a', 'aac', 'wma', 'aiff'];
    const ext = file.name.split('.').pop().toLowerCase();
    if (!allowed.includes(ext)) {
        alert(`Unsupported format (.${ext})\nSupported: ${allowed.join(', ')}`);
        return;
    }

    currentFile = file;

    // Show file info
    document.getElementById('dz-empty').style.display = 'none';
    document.getElementById('dz-filled').style.display = 'flex';
    document.getElementById('upload-file-name').textContent = file.name;
    document.getElementById('upload-file-size').textContent = formatBytes(file.size);

    // Show player
    document.getElementById('player-area').style.display = 'block';

    // Load into audio player
    loadTrackIntoPlayer(file);

    // Auto-predict
    runAutoPredict(file);
}

function clearUploadedFile() {
    currentFile = null;
    document.getElementById('audio-file-input').value = '';
    document.getElementById('dz-empty').style.display = 'block';
    document.getElementById('dz-filled').style.display = 'none';

    // Stop and hide player
    audio.pause();
    audio.src = '';
    isPlaying = false;
    showPlayIcon();
    cancelAnimationFrame(animId);
    document.getElementById('player-area').style.display = 'none';

    // Reset result
    document.getElementById('auto-result-content').style.display = 'none';
    document.getElementById('auto-placeholder').style.display = 'flex';

    setStatus('');
}

async function runAutoPredict(file) {
    // Show initial status
    setStatus('⏳ Validating file...', 'analyzing');

    // Create progress simulation
    let progress = 0;
    const progressInterval = setInterval(() => {
        progress += Math.random() * 5;
        if (progress > 95) progress = 95;
        setStatus(`⏳ Analyzing audio features... ${Math.round(progress)}% <br><small>(Large files may take 1-2 minutes)</small>`, 'analyzing');
    }, 800);

    const fd = new FormData();
    fd.append('audio_file', file);

    try {
        const res = await fetch('/predict-file', { method: 'POST', body: fd });

        clearInterval(progressInterval); // Stop animation

        if (!res.ok) throw new Error('Network response was not ok');

        const data = await res.json();

        if (data.status === 'success') {
            setStatus('✅ Analysis complete!', 'success');
            displayAutoResult(data.popularity, data.features, data.file_name);
            updateSlidersFromFeatures(data.features);
        } else {
            setStatus('❌ ' + (data.error || 'Analysis failed'), 'error');
        }
    } catch (err) {
        clearInterval(progressInterval);
        console.error(err);
        setStatus('❌ Server timeout or network error. <br><small>Try a shorter file or check internet connection.</small>', 'error');
    }
}

function displayAutoResult(popularity, features, fileName) {
    document.getElementById('auto-placeholder').style.display = 'none';
    const content = document.getElementById('auto-result-content');
    content.style.display = 'flex';

    document.getElementById('auto-song-name').textContent = '🎵 ' + fileName;
    animateGauge('auto-gauge-fill', 'auto-gauge-value', popularity);
    setPopLabel('auto-pop-label', popularity);
    renderFeatureDetails('auto-result-features', features);
    renderRadarChart('auto-radar-chart', features);
}

function setStatus(msg, cls) {
    const el = document.getElementById('upload-status');
    el.innerHTML = msg; // Use innerHTML to support <br>
    el.className = 'upload-status' + (cls ? ' ' + cls : '');
}

// ============================================================
//  AUDIO PLAYER
// ============================================================
function initPlayer() {
    const playBtn = document.getElementById('btn-play');
    const progTrack = document.querySelector('#progress-wrap .progress-track');

    playBtn.addEventListener('click', togglePlay);

    audio.addEventListener('timeupdate', updateProgress);
    audio.addEventListener('ended', () => {
        isPlaying = false;
        showPlayIcon();
        cancelAnimationFrame(animId);
    });

    progTrack.addEventListener('click', e => {
        if (!audio.duration) return;
        audio.currentTime = (e.offsetX / progTrack.clientWidth) * audio.duration;
    });
}

function loadTrackIntoPlayer(file) {
    const url = URL.createObjectURL(file);
    audio.src = url;

    audio.play().then(() => {
        isPlaying = true;
        showPauseIcon();
        startVisualizer();
    }).catch(() => { });
}

function togglePlay() {
    if (!audio.src) return;
    if (isPlaying) {
        audio.pause(); isPlaying = false;
        showPlayIcon();
        cancelAnimationFrame(animId);
    } else {
        audio.play(); isPlaying = true;
        showPauseIcon();
        startVisualizer();
    }
}

function showPlayIcon() { document.getElementById('icon-play').classList.remove('hidden'); document.getElementById('icon-pause').classList.add('hidden'); }
function showPauseIcon() { document.getElementById('icon-play').classList.add('hidden'); document.getElementById('icon-pause').classList.remove('hidden'); }

function updateProgress() {
    if (!audio.duration) return;
    document.getElementById('progress-fill').style.width = (audio.currentTime / audio.duration * 100) + '%';
    document.getElementById('time-current').textContent = fmtTime(audio.currentTime);
    document.getElementById('time-total').textContent = fmtTime(audio.duration);
}

function fmtTime(s) {
    return Math.floor(s / 60) + ':' + String(Math.floor(s % 60)).padStart(2, '0');
}

// ============================================================
//  VISUALIZER
// ============================================================
function startVisualizer() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
        analyser = audioCtx.createAnalyser();
        analyser.fftSize = 256;
        source = audioCtx.createMediaElementSource(audio);
        source.connect(analyser);
        analyser.connect(audioCtx.destination);
        dataArr = new Uint8Array(analyser.frequencyBinCount);
    }
    if (audioCtx.state === 'suspended') audioCtx.resume();

    const canvas = document.getElementById('visualizer-canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = canvas.parentElement.clientWidth;
    canvas.height = 120;

    function draw() {
        animId = requestAnimationFrame(draw);
        analyser.getByteFrequencyData(dataArr);
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        const bars = dataArr.length;
        const barW = (canvas.width / bars) * 2.5;
        let x = 0;
        for (let i = 0; i < bars; i++) {
            const h = dataArr[i] / 2;
            const grad = ctx.createLinearGradient(0, canvas.height, 0, 0);
            grad.addColorStop(0, '#7928ca');
            grad.addColorStop(0.5, '#ff0080');
            grad.addColorStop(1, '#00d2ff');
            ctx.fillStyle = grad;
            ctx.fillRect(x, canvas.height / 2 - h / 2, barW, h);
            x += barW + 1;
        }
    }
    draw();
}

// ============================================================
//  SECTION 2: MANUAL SLIDERS & PREDICT
// ============================================================
function initSliders() {
    document.querySelectorAll('.range').forEach(s => {
        updateSliderUI(s);
        s.addEventListener('input', () => updateSliderUI(s));
    });
}

function updateSliderUI(slider) {
    const id = slider.id;
    const val = parseFloat(slider.value);
    const unit = slider.dataset.unit || '';
    const el = document.getElementById(`${id}-val`);
    if (!el) return;

    if (id === 'energy' || id === 'danceability') el.textContent = val.toFixed(2) + unit;
    else if (id === 'duration_min' || id === 'loudness_db') el.textContent = val.toFixed(1) + unit;
    else el.textContent = Math.round(val) + unit;

    const pct = ((val - parseFloat(slider.min)) / (parseFloat(slider.max) - parseFloat(slider.min))) * 100;
    slider.style.background = `linear-gradient(90deg, #7928ca ${pct}%, #1e1e3a ${pct}%)`;
}

function updateSlidersFromFeatures(features) {
    for (const [key, val] of Object.entries(features)) {
        const s = document.getElementById(key);
        if (s) { s.value = val; updateSliderUI(s); }
    }
}

function initManualForm() {
    document.getElementById('predict-form').addEventListener('submit', async e => {
        e.preventDefault();
        const btn = document.getElementById('predict-btn');
        const btnText = btn.querySelector('.btn-text');
        const btnLoad = btn.querySelector('.btn-loader');

        btnText.style.display = 'none';
        btnLoad.style.display = 'inline-block';
        btn.disabled = true;

        const features = {
            duration_min: parseFloat(document.getElementById('duration_min').value),
            tempo_bpm: parseFloat(document.getElementById('tempo_bpm').value),
            energy: parseFloat(document.getElementById('energy').value),
            danceability: parseFloat(document.getElementById('danceability').value),
            loudness_db: parseFloat(document.getElementById('loudness_db').value)
        };

        try {
            const res = await fetch('/predict', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(features)
            });
            const data = await res.json();
            if (data.status === 'success') {
                displayManualResult(data.popularity, data.features);
            } else {
                alert('Error: ' + (data.error || 'Unknown'));
            }
        } catch (err) {
            alert('Network error: ' + err.message);
        } finally {
            btnText.style.display = 'inline';
            btnLoad.style.display = 'none';
            btn.disabled = false;
        }
    });
}

function displayManualResult(popularity, features) {
    document.getElementById('manual-placeholder').style.display = 'none';
    const content = document.getElementById('manual-result-content');
    content.style.display = 'flex';

    animateGauge('manual-gauge-fill', 'manual-gauge-value', popularity);
    setPopLabel('manual-pop-label', popularity);
    renderFeatureDetails('manual-result-features', features);
    renderRadarChart('manual-radar-chart', features);
}

// ============================================================
//  SHARED: Gauge, Labels, Details
// ============================================================
function animateGauge(fillId, valId, value) {
    const fill = document.getElementById(fillId);
    const vEl = document.getElementById(valId);
    const maxDash = 401;
    const offset = maxDash - (maxDash * (value / 100) * (267 / 360));

    // Reset for re-animation
    fill.style.transition = 'none';
    fill.style.strokeDashoffset = 401;
    fill.getBoundingClientRect(); // force reflow
    fill.style.transition = 'stroke-dashoffset 1.5s cubic-bezier(.25,.46,.45,.94)';

    setTimeout(() => { fill.style.strokeDashoffset = Math.max(offset, maxDash * 0.03); }, 50);

    const target = Math.round(value * 10) / 10;
    const duration = 1400;
    const start = performance.now();
    (function tick(now) {
        const p = Math.min((now - start) / duration, 1);
        const e = 1 - Math.pow(1 - p, 3);
        vEl.textContent = (target * e).toFixed(1);
        if (p < 1) requestAnimationFrame(tick);
    })(start);
}

function setPopLabel(id, pop) {
    const el = document.getElementById(id);
    el.className = 'pop-label';
    if (pop < 25) { el.textContent = '🔻 Low'; el.classList.add('pop-low'); }
    else if (pop < 50) { el.textContent = '📊 Moderate'; el.classList.add('pop-medium'); }
    else if (pop < 75) { el.textContent = '🔥 High'; el.classList.add('pop-high'); }
    else { el.textContent = '🚀 Viral!'; el.classList.add('pop-viral'); }
}

function renderFeatureDetails(containerId, features) {
    const grid = document.getElementById(containerId);
    grid.innerHTML = '';
    for (const [key, val] of Object.entries(features)) {
        const icon = FEAT_ICONS[key] || '';
        const name = FEAT_NAMES[key] || key;
        let fmt;
        if (key === 'energy' || key === 'danceability') fmt = val.toFixed(2);
        else if (key === 'duration_min') fmt = val.toFixed(1) + ' min';
        else if (key === 'tempo_bpm') fmt = Math.round(val) + ' BPM';
        else if (key === 'loudness_db') fmt = val.toFixed(1) + ' dB';
        else fmt = val;

        const d = document.createElement('div');
        d.className = 'detail-item';
        d.innerHTML = `<span class="label">${icon} ${name}</span><span class="value">${fmt}</span>`;
        grid.appendChild(d);
    }
}

// ============================================================
//  MODEL INFO
// ============================================================
async function loadModelInfo() {
    try {
        const res = await fetch('/model-info');
        const data = await res.json();
        renderMetrics(data.evaluation);
        renderCoefficients(data.evaluation);
        featureRanges = data.feature_ranges; // Store globally
    } catch (err) {
        console.error('Model info load failed:', err);
    }
}

function renderMetrics(ev) {
    const grid = document.getElementById('metrics-grid');
    const items = [
        { name: 'R² (Train)', val: ev.r2_train, fmt: v => v.toFixed(4) },
        { name: 'R² (Test)', val: ev.r2_test, fmt: v => v.toFixed(4) },
        { name: 'Adjusted R²', val: ev.adjusted_r2, fmt: v => v.toFixed(4) },
        { name: 'MAE', val: ev.mae, fmt: v => v.toFixed(2) },
        { name: 'MSE', val: ev.mse, fmt: v => v.toFixed(2) },
        { name: 'RMSE', val: ev.rmse, fmt: v => v.toFixed(2) },
    ];
    grid.innerHTML = items.map((m, i) => `
        <div class="metric-card reveal" style="transition-delay:${i * 80}ms">
            <div class="metric-val">${m.fmt(m.val)}</div>
            <div class="metric-name">${m.name}</div>
        </div>
    `).join('');
    requestAnimationFrame(() => grid.querySelectorAll('.reveal').forEach(el => el.classList.add('visible')));
}

function renderCoefficients(ev) {
    const grid = document.getElementById('coeff-grid');
    const coeffs = ev.coefficients;
    const maxAbs = Math.max(...Object.values(coeffs).map(Math.abs));

    grid.innerHTML = Object.entries(coeffs).map(([name, value], i) => {
        const pos = value >= 0;
        const w = (Math.abs(value) / maxAbs) * 100;
        const icon = FEAT_ICONS[name] || '';
        const label = FEAT_NAMES[name] || name;
        return `
            <div class="coeff-card reveal" style="transition-delay:${i * 80}ms">
                <div class="coeff-name">${icon} ${label}</div>
                <div class="coeff-value ${pos ? 'coeff-positive' : 'coeff-negative'}">${pos ? '+' : ''}${value.toFixed(4)}</div>
                <div class="coeff-bar ${pos ? 'coeff-bar-pos' : 'coeff-bar-neg'}" style="width:${w}%"></div>
            </div>
        `;
    }).join('');
    requestAnimationFrame(() => grid.querySelectorAll('.reveal').forEach(el => el.classList.add('visible')));
}

// ============================================================
//  BACKGROUND PARTICLES (Musical notes)
// ============================================================
function initParticles() {
    const canvas = document.getElementById('bg-particles');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    let W, H;
    const particles = [];
    const NOTES = ['♪', '♫', '♬', '♩', '𝄞'];
    const COUNT = 28;

    function resize() { W = canvas.width = window.innerWidth; H = canvas.height = window.innerHeight; }
    resize();
    window.addEventListener('resize', resize);

    for (let i = 0; i < COUNT; i++) {
        particles.push({
            x: Math.random() * W, y: Math.random() * H,
            vx: (Math.random() - .5) * .3, vy: -Math.random() * .4 - .1,
            size: Math.random() * 14 + 10,
            opacity: Math.random() * .12 + .03,
            note: NOTES[Math.floor(Math.random() * NOTES.length)],
            rot: Math.random() * 360, rotV: (Math.random() - .5) * .4
        });
    }

    function frame() {
        requestAnimationFrame(frame);
        ctx.clearRect(0, 0, W, H);
        for (const p of particles) {
            p.x += p.vx; p.y += p.vy; p.rot += p.rotV;
            if (p.y < -30) { p.y = H + 30; p.x = Math.random() * W; }
            if (p.x < -30) p.x = W + 30;
            if (p.x > W + 30) p.x = -30;
            ctx.save();
            ctx.translate(p.x, p.y);
            ctx.rotate(p.rot * Math.PI / 180);
            ctx.globalAlpha = p.opacity;
            ctx.font = `${p.size}px serif`;
            ctx.fillStyle = '#a78bfa';
            ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
            ctx.fillText(p.note, 0, 0);
            ctx.restore();
        }
    }
    frame();
}

// ============================================================
//  SCROLL REVEAL
// ============================================================
function initScrollReveal() {
    const obs = new IntersectionObserver(entries => {
        entries.forEach(e => {
            if (e.isIntersecting) {
                e.target.classList.add('reveal', 'visible');
                e.target.querySelectorAll('.reveal').forEach(el => el.classList.add('visible'));
            }
        });
    }, { threshold: 0.1 });
    document.querySelectorAll('.section').forEach(s => { s.classList.add('reveal'); obs.observe(s); });
}

// ============================================================
//  CHARTS (Radar)
// ============================================================
let chartInstances = {};
let featureRanges = null; // Populated from /model-info

function renderRadarChart(canvasId, features) {
    if (!featureRanges) return; // Not loaded yet

    const ctx = document.getElementById(canvasId).getContext('2d');

    // Destroy existing chart if any
    if (chartInstances[canvasId]) {
        chartInstances[canvasId].destroy();
    }

    // Normalize values 0-1
    const labels = Object.keys(features).map(k => FEAT_NAMES[k] || k);
    const data = Object.keys(features).map(k => {
        const val = features[k];
        const min = featureRanges[k].min;
        const max = featureRanges[k].max;
        const norm = (val - min) / (max - min);
        return Math.max(0, Math.min(1, norm)); // Clip 0-1
    });

    chartInstances[canvasId] = new Chart(ctx, {
        type: 'radar',
        data: {
            labels: labels,
            datasets: [{
                label: 'Audio Profile',
                data: data,
                backgroundColor: 'rgba(0, 210, 255, 0.2)',
                borderColor: '#00d2ff',
                pointBackgroundColor: '#fff',
                pointBorderColor: '#ff0080',
                pointHoverBackgroundColor: '#fff',
                pointHoverBorderColor: '#ff0080',
                borderWidth: 2,
                pointRadius: 4
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                r: {
                    angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
                    grid: { color: 'rgba(255, 255, 255, 0.1)' },
                    pointLabels: {
                        color: '#a0a0c0',
                        font: { family: 'Outfit', size: 12 }
                    },
                    ticks: { display: false, maxTicksLimit: 5 },
                    suggestedMin: 0,
                    suggestedMax: 1
                }
            },
            plugins: {
                legend: { display: false },
                tooltip: {
                    callbacks: {
                        label: (ctx) => {
                            const key = Object.keys(features)[ctx.dataIndex];
                            return `${FEAT_NAMES[key]}: ${features[key]}`;
                        }
                    }
                }
            }
        }
    });
}


// ============================================================
//  UTILS
// ============================================================
function formatBytes(b) {
    if (b < 1024) return b + ' B';
    if (b < 1024 * 1024) return (b / 1024).toFixed(1) + ' KB';
    return (b / (1024 * 1024)).toFixed(1) + ' MB';
}
