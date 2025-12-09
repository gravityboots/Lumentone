document.addEventListener('DOMContentLoaded', () => {
	const startBtn = document.getElementById('start-btn');
	const stopBtn = document.getElementById('stop-btn');
	const generateBtn = document.getElementById('generate-btn');
	const jumpBtn = document.getElementById('jump-btn');
	const cameraSelect = document.getElementById('camera-select');
	const hrRange = document.getElementById('heart-rate');
	const hrValue = document.getElementById('hr-value');
	const modeSelect = document.getElementById('playlist-mode');
	const moodLabel = document.getElementById('mood-label');
	const vaLabel = document.getElementById('va-label');
	const genresBox = document.getElementById('genres');
	const playlistBox = document.getElementById('playlist');
	const playlistMeta = document.getElementById('playlist-meta');
	const previewPlayer = document.getElementById('preview-player');
	const hrLive = document.getElementById('hr-live');
	const posPlot = document.getElementById('pos-plot');
	const rppgToggle = document.getElementById('rppg-toggle');
	const rppgBody = document.getElementById('rppg-body');
	const rppgArrow = document.getElementById('rppg-arrow');
	const moodIcon = document.getElementById('mood-icon');
	const posValue = document.getElementById('pos-value');
	const hrLiveText = document.getElementById('hr-live-text');
	const hrMean = document.getElementById('hr-mean');
	const hrMedian = document.getElementById('hr-median');
	const hrMode = document.getElementById('hr-mode');
	const hrMethod = document.getElementById('hr-method');

	let currentPlaylist = [];
	jumpBtn.disabled = true;
	let lastHrBpm = null;
	let posPanelOpen = false;
	let isRunning = false;

	const setDefaultMoodIcon = () => {
		if (!moodIcon) return;
		moodIcon.style.backgroundImage = 'linear-gradient(135deg, #a16aff, #4b1c92)';
		moodIcon.style.webkitMaskImage = "url('/assets/music.svg')";
		moodIcon.style.maskImage = "url('/assets/music.svg')";
		if (moodLabel) {
			moodLabel.style.color = '#adb5bd';
		}
	};

	const moodMap = {
		depressed: { icon: '/assets/depressed.svg', colors: ['#548ce0ff', '#215097ff'] },
		sad: { icon: '/assets/frown.svg', colors: ['#5fa0ff', '#3f78ff'] },
		neutral: { icon: '/assets/meh.svg', colors: ['#9da6b5', '#7f8799'] },
		idle: { icon: '/assets/meh.svg', colors: ['#9da6b5', '#7f8799'] },
		happy: { icon: '/assets/smile.svg', colors: ['#7df3ff', '#7ab8ff'] },
		laugh: { icon: '/assets/laugh.svg', colors: ['#7df3ff', '#7ab8ff'] },
		angry: { icon: '/assets/angry.svg', colors: ['#ff7f6f', '#ff3e3e'] },
		stressed: { icon: '/assets/angry.svg', colors: ['#ffb85c', '#ff7a3d'] },
		anxious: { icon: '/assets/angry.svg', colors: ['#ffb85c', '#ff7a3d'] },
	};

	const applyMoodDisplay = (label, valence) => {
		if (!label) {
			setDefaultMoodIcon();
			return;
		}
		const key = label.toLowerCase();
		let entry = moodMap.neutral;
		if (key.includes('depressed')) entry = moodMap.depressed;
		else if (key.includes('sad')) entry = moodMap.sad;
		else if (key.includes('happy')) entry = valence === 'positive' ? moodMap.laugh : moodMap.happy;
		else if (key.includes('stressed') || key.includes('anxious')) entry = moodMap.stressed;
		else if (key.includes('angry')) entry = moodMap.angry;
		else if (key.includes('idle') || key.includes('neutral')) entry = moodMap.neutral;

		if (moodIcon && entry?.icon) {
			moodIcon.style.backgroundImage = `linear-gradient(135deg, ${entry.colors[0]}, ${entry.colors[1]})`;
			moodIcon.style.webkitMaskImage = `url('${entry.icon}')`;
			moodIcon.style.maskImage = `url('${entry.icon}')`;
		}
		if (moodLabel && entry?.colors) {
			moodLabel.style.color = entry.colors[1];
		}
	};

	setDefaultMoodIcon();

	const updateHr = () => {
		hrValue.textContent = `${hrRange.value} bpm`;
	};
	updateHr();

	const refreshCameras = async () => {
		cameraSelect.innerHTML = '<option>Loading...</option>';
		try {
			const res = await fetch('/cameras');
			const data = await res.json();
			const cams = data?.cameras || [];
			cameraSelect.innerHTML = '';
			cams.forEach((cam) => {
				const opt = document.createElement('option');
				opt.value = cam.index;
				opt.textContent = cam.label || `Camera ${cam.index}`;
				cameraSelect.appendChild(opt);
			});
			if (!cams.length) {
				const opt = document.createElement('option');
				opt.value = 0;
				opt.textContent = 'Camera 0';
				cameraSelect.appendChild(opt);
			}
		} catch (err) {
			console.error(err);
			cameraSelect.innerHTML = '<option value="0">Camera 0</option>';
		}
	};

	startBtn.addEventListener('click', async () => {
		startBtn.disabled = true;
		try {
			const body = { camera_index: parseInt(cameraSelect.value || '0', 10) };
			await fetch('/start_capture', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
		} catch (err) {
			console.error(err);
		} finally {
			startBtn.disabled = false;
		}
	});

	stopBtn.addEventListener('click', async () => {
		stopBtn.disabled = true;
		try {
			await fetch('/stop_capture', { method: 'POST' });
		} catch (err) {
			console.error(err);
		} finally {
			stopBtn.disabled = false;
		}
	});

	const renderGenres = (genres) => {
		genresBox.innerHTML = '';
		(genres || []).forEach((g) => {
			const badge = document.createElement('span');
			badge.className = 'badge text-bg-secondary';
			badge.textContent = g;
			genresBox.appendChild(badge);
		});
	};

	const renderPlaylist = (tracks) => {
		playlistBox.innerHTML = '';
		if (!tracks || !tracks.length) {
			playlistMeta.textContent = 'No tracks yet';
			jumpBtn.disabled = true;
			return;
		}
		playlistMeta.textContent = `${tracks.length} tracks ready`;
		tracks.forEach((t) => {
			const item = document.createElement('div');
			item.className = 'list-group-item d-flex justify-content-between align-items-start gap-2 flex-wrap';

			const left = document.createElement('div');
			left.innerHTML = `<div class="fw-semibold">${t.title || 'Untitled'}</div><div class="small text-secondary">${t.artist || 'Unknown'} • ${t.genre || ''}</div>`;

			const right = document.createElement('div');
			right.className = 'd-flex gap-1 flex-wrap';

			if (t.preview) {
				const audioBtn = document.createElement('button');
				audioBtn.className = 'btn btn-sm btn-outline-light';
				audioBtn.type = 'button';
				audioBtn.textContent = 'Preview';
				audioBtn.addEventListener('click', () => {
					previewPlayer.src = t.preview;
					previewPlayer.play().catch(() => {});
				});
				right.appendChild(audioBtn);
			}
			if (t.spotify) {
				const sBtn = document.createElement('a');
				sBtn.className = 'btn btn-sm btn-success';
				sBtn.href = t.spotify;
				sBtn.target = '_blank';
				sBtn.textContent = 'Spotify';
				right.appendChild(sBtn);
			}
			if (t.youtube) {
				const yBtn = document.createElement('a');
				yBtn.className = 'btn btn-sm btn-danger';
				yBtn.href = t.youtube;
				yBtn.target = '_blank';
				yBtn.textContent = 'YouTube';
				right.appendChild(yBtn);
			}

			item.appendChild(left);
			item.appendChild(right);
			playlistBox.appendChild(item);
		});
		jumpBtn.disabled = false;
	};

	generateBtn.addEventListener('click', async () => {
		generateBtn.disabled = true;
		try {
			const body = {
				heart_rate: parseInt(hrRange.value, 10),
				playlist_mode: modeSelect.value,
			};
			const res = await fetch('/playlist', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(body),
			});
			const data = await res.json();
			console.log('[/playlist] response', data);
			if (!data.ok) {
				console.error('Playlist error', data.error);
				moodLabel.textContent = data.mood || '--';
				vaLabel.textContent = '-- / --';
				applyMoodDisplay(data.mood, data.valence);
				// keep genres from status
				renderPlaylist([]);
				currentPlaylist = [];
				jumpBtn.disabled = true;
				return;
			}
			moodLabel.textContent = data.mood || '--';
			vaLabel.textContent = `${data.valence || '--'} / ${data.arousal || '--'}`;
			applyMoodDisplay(data.mood, data.valence);
			renderGenres(data.genres);
			renderPlaylist(data.tracks);
			currentPlaylist = data.tracks || [];
		} catch (err) {
			console.error(err);
			jumpBtn.disabled = !(currentPlaylist.length > 0);
		} finally {
			generateBtn.disabled = false;
		}
	});

	jumpBtn.addEventListener('click', () => {
		if (!currentPlaylist.length) return;
		const first = currentPlaylist[0];
		const url = first.spotify || first.youtube || first.preview || first.source;
		if (url) {
			window.open(url, '_blank');
		}
	});

	hrRange.addEventListener('input', updateHr);

	const pollStatus = async () => {
		try {
			const res = await fetch(`/status?heart_rate=${encodeURIComponent(hrRange.value)}`);
			const data = await res.json();
			const running = data?.running === true;
			isRunning = running;
			if (running && data?.mood?.label) {
				moodLabel.textContent = data.mood.label;
				applyMoodDisplay(data.mood.label, data.valence);
			}
			if (running && data?.valence && data?.arousal) {
				vaLabel.textContent = `${data.valence} / ${data.arousal}`;
			}
			if (running && data?.genres) {
				renderGenres(data.genres);
			}
			if (running && typeof data?.hr_bpm === 'number' && !Number.isNaN(data.hr_bpm)) {
				lastHrBpm = data.hr_bpm;
				const clamp = (v, min, max) => Math.min(max, Math.max(min, v));
				const newVal = clamp(data.hr_bpm, parseInt(hrRange.min, 10), parseInt(hrRange.max, 10));
				hrRange.value = newVal;
				updateHr();
			}
			if (running && lastHrBpm !== null) {
				hrLive.textContent = `${lastHrBpm.toFixed(1)} bpm`;
				if (hrLiveText) hrLiveText.textContent = `${lastHrBpm.toFixed(1)} bpm`;
			}
			if (running && typeof data?.hr_mean === 'number') {
				hrMean.textContent = data.hr_mean.toFixed(1);
			}
			if (running && typeof data?.hr_median === 'number') {
				hrMedian.textContent = data.hr_median.toFixed(1);
			}
			if (running && typeof data?.hr_mode === 'number') {
				hrMode.textContent = data.hr_mode.toFixed(1);
			}
			if (running && typeof data?.last_pos !== 'undefined') {
				posValue.textContent = typeof data.last_pos === 'number' ? data.last_pos.toFixed(3) : '--';
			}
			if (running && typeof data?.hr_method !== 'undefined' && hrMethod) {
				hrMethod.textContent = data.hr_method || '--';
			}
		} catch (err) {
			console.error(err);
		}
	};
	setInterval(pollStatus, 1500);
	refreshCameras();

	const refreshPosPlot = () => {
		if (!posPlot || !posPanelOpen || !isRunning) return;
		posPlot.src = `/pos_plot?ts=${Date.now()}`;
	};
	setInterval(refreshPosPlot, 2000);

	if (rppgToggle && rppgBody) {
		rppgToggle.addEventListener('click', () => {
			posPanelOpen = !posPanelOpen;
			rppgBody.classList.toggle('show', posPanelOpen);
			if (rppgArrow) {
				rppgArrow.textContent = posPanelOpen ? '▲' : '▼';
			}
			if (posPanelOpen) {
				refreshPosPlot();
			}
		});
	}
});
