/* BondSight page logic. Plain JavaScript, no libraries.
   Three pages (hash routes): #inspect, #check, #review. Talks only to the /api/ addresses
   listed in web/app.py. Every error shows in the red bar under the header, never as a pop-up. */
"use strict";
const App = (() => {
  const $ = (id) => document.getElementById(id);
  const S = { status: null, unit: null, view: "top", sel: null, lastCycle: null,
              filter: "all", reviewRows: [], drawing: false, demo: false, results: new Map() };
  const FILTERS = [["all", "All", {}], ["undecided", "Not decided", { undecided: 1 }],
    ["DEFECT", "DEFECT", { verdict: "DEFECT" }], ["CHECK", "CHECK", { verdict: "CHECK" }],
    ["OK", "OK", { verdict: "OK" }], ["upload", "Photos", { source: "upload" }], ["camera", "Line", { source: "camera" }]];

  // ---------- small helpers ----------
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
  const clsName = (c) => String(c || "").replace(/_/g, " ");
  const posTag = (p) => { const m = /^wire:(\d+)$/.exec(p || ""); return m ? "W" + String(m[1]).padStart(2, "0") : (p || "?"); };
  const pct = (s) => Math.round((s || 0) * 100) + "%";
  const time = (ts) => new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  const unitName = (u) => u.source === "upload" ? (u.label || "photo") : "#" + u.unit_no;
  function store(k, v) { try { if (v === undefined) return localStorage.getItem(k); localStorage.setItem(k, v); } catch (e) { return null; } }

  function showError(msg) { const b = $("errbar"); b.textContent = msg; b.hidden = !msg; }
  async function api(path, opts = {}) {
    let r;
    try { r = await fetch(path, opts); } catch (e) { throw new Error("The inspector does not answer. Is it running?"); }
    let body = null;
    try { body = await r.json(); } catch (e) { /* empty body */ }
    if (!r.ok) {
      const d = body && body.detail;
      throw Object.assign(new Error(typeof d === "string" ? d : `request failed (${r.status})`), { status: r.status });
    }
    return body;
  }

  // ---------- operator ----------
  const operator = () => store("wb.operator") || "";
  function askOperator() {
    return new Promise((resolve) => {
      const m = $("op-modal"), inp = $("op-name");
      inp.value = operator(); m.hidden = false; inp.focus();
      const done = (name) => { m.hidden = true; $("op-form").onsubmit = null; $("op-cancel").onclick = null; resolve(name); };
      $("op-form").onsubmit = (e) => { e.preventDefault(); const n = inp.value.trim(); if (!n) return;
        store("wb.operator", n); paintOperator(); done(n); };
      $("op-cancel").onclick = () => done(null);
    });
  }
  function paintOperator() { $("who").textContent = operator() ? operator() + " · operator" : "Who is inspecting?"; }
  async function needOperator() { return operator() || await askOperator(); }

  // ---------- status / rail ----------
  async function loadStatus() {
    try {
      const s = await api("/api/status"); S.status = s;
      $("station").textContent = s.station || "station";
      $("r-model").textContent = s.model; $("r-units").textContent = s.line_units; $("r-db").textContent = s.database;
      $("lamps").innerHTML = s.cameras.map((c) => `<span><span class="led ${c.ok ? "ok" : "bad"}"></span>${esc(c.name)} cam</span>`).join("")
        + `<span><span class="led ok"></span>Database</span>`;
      const mem = $("memory"); const keep = mem.value;
      mem.innerHTML = `<option value="">none</option>` + s.cameras.map((c) => `<option value="${esc(c.name)}">${esc(c.name)}</option>`).join("");
      mem.value = keep;
      if (s.last_error) showError(s.last_error);
      else if (S.offline) showError("");          // it answers again: clear the "does not answer" bar
      S.offline = false;
    } catch (e) {
      $("lamps").innerHTML = `<span><span class="led bad"></span>Inspector offline</span>`;
      S.offline = true; showError(e.message);
    }
  }
  async function loadCounts() {
    try {
      const rows = await api("/api/units?source=camera&limit=200");
      const d = rows.filter((r) => r.verdict === "DEFECT").length, c = rows.filter((r) => r.verdict === "CHECK").length;
      $("r-dc").textContent = `${d} / ${c}`;
      const und = await api("/api/units?undecided=1&limit=500");
      $("undecided-count").textContent = und.length ? (und.length >= 500 ? "500+" : und.length) : "";
    } catch (e) { /* the rail is optional */ }
  }

  // ---------- picture with boxes ----------
  function photoEl(u, shot, opts = {}) {
    const w = shot.width || 640, h = shot.height || 480;
    const div = document.createElement("div");
    div.className = "photo"; div.style.aspectRatio = `${w} / ${h}`; div.style.setProperty("--ar", w / h);
    const src = opts.heat && shot.heat_path ? `/api/images/${u.id}/${shot.camera}_heat.jpg`
      : u.image_b64 ? `data:image/jpeg;base64,${u.image_b64}` : `/api/images/${u.id}/${shot.camera}.jpg`;
    const boxes = [];
    for (const f of u.findings.filter((f) => f.camera === shot.camera)) {
      const [x0, y0, x1, y1] = f.box;
      const c = ["box", "f", opts.pop ? "pop" : "", f.id === S.sel ? "sel" : "", f.decision ? "dec-" + f.decision : ""].join(" ");
      boxes.push(`<rect class="${c}" x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}"/>`);
    }
    for (const d of (u.decisions || []).filter((d) => d.verdict === "missed" && !d.withdrawn && d.camera === shot.camera)) {
      const [x0, y0, x1, y1] = d.box;
      boxes.push(`<rect class="box m" x="${x0}" y="${y0}" width="${x1 - x0}" height="${y1 - y0}"/>`);
    }
    div.innerHTML = `<img alt="${esc(shot.camera)} picture" src="${src}">`
      + `<svg viewBox="0 0 ${w} ${h}" preserveAspectRatio="none">${opts.heat ? "" : boxes.join("")}</svg>`;
    return div;
  }

  // ---------- Inspect page ----------
  function lamps(verdict, pulse) {
    for (const [id, v] of [["lamp-r", "DEFECT"], ["lamp-a", "CHECK"], ["lamp-g", "OK"]]) {
      const el = $(id); el.classList.toggle("lit", verdict === v); el.classList.remove("pulse");
      if (pulse && verdict === v) { void el.offsetWidth; el.classList.add("pulse"); }
    }
  }
  function shotsWithPics(u) { return u.shots.filter((s) => s.image_path || u.image_b64); }

  function renderStage(pop) {
    const u = S.unit, body = $("stage-body"), views = $("views");
    if (!u) return;
    const pics = shotsWithPics(u);
    const names = pics.map((s) => s.camera);
    if (S.view !== "3d" && S.view !== "unusual" && !names.includes(S.view)) S.view = names[0] || "3d";
    const anyHeat = pics.some((s) => s.heat_path);
    const btns = names.map((n) => [n, n[0].toUpperCase() + n.slice(1)]);
    if (pics.length > 1) btns.push(["3d", "3D"]);
    btns.push(["unusual", "Unusual"]);
    views.innerHTML = btns.map(([v, t]) => `<button class="vbtn ${S.view === v ? "on" : ""}" data-v="${v}" type="button"
      ${v === "unusual" && !anyHeat ? 'title="No unusual spots on this unit"' : ""}>${t}</button>`).join("");
    body.innerHTML = ""; body.className = "stage-body";
    if (!pics.length) { body.innerHTML = `<div class="empty">No picture: ${esc(u.shots.map((s) => s.error).filter(Boolean).join("; "))}</div>`; }
    else if (S.view === "3d") { body.classList.add("three"); pics.forEach((s) => body.appendChild(photoEl(u, s, { pop }))); }
    else if (S.view === "unusual") {
      const hs = pics.filter((s) => s.heat_path);
      if (!hs.length) body.innerHTML = `<div class="empty">No unusual spots on this unit.</div>`;
      hs.forEach((s) => body.appendChild(photoEl(u, s, { heat: true })));
    } else {
      body.classList.add("grid-bg");
      const shot = pics.find((s) => s.camera === S.view);
      body.appendChild(photoEl(u, shot, { pop }));
      body.insertAdjacentHTML("beforeend", `<div class="cross"></div>`);
    }
    $("taglines").innerHTML = u.findings.map((f) => `<button type="button" class="label ${f.decision === "false_alarm" ? "ok" : "r"} ${f.id === S.sel ? "sel" : ""}"
        data-f="${f.id}">${posTag(f.position)} · ${esc(clsName(f.cls).toUpperCase())} · ${pct(f.score)}${f.decision ? " · " + (f.decision === "real" ? "REAL" : "FALSE ALARM") : ""}</button>`).join("")
      + u.alerts.map((a) => `<span class="label am">REPEAT ${a.count}/${a.window} · ${posTag(a.position)}</span>`).join("");
  }

  function renderSide(pulse) {
    const u = S.unit, v = $("verdict");
    if (!u) return;
    v.className = "verdict " + u.verdict;
    const when = u.source === "upload" ? `Photo ${esc(u.label || "")}` : `Unit #${u.unit_no}`;
    const state = u.id == null ? "not saved" : u.review_state === "decided" ? "decided" : "decision needed";
    v.innerHTML = (u.verdict === "DEFECT" ? `<div class="hazard"></div>` : "")
      + `<div class="v">${u.verdict}</div><div class="m">${when} · ${time(u.ts)} · ${state}</div>`;
    lamps(u.verdict, pulse);
    const rows = u.findings.map((f) => `<button type="button" class="ro ${f.id === S.sel ? "sel" : ""}" data-f="${f.id}">
      <span>${posTag(f.position)} ${esc(clsName(f.cls))}${f.decision ? ` · <i>${f.decision === "real" ? "real" : "false alarm"}</i>` : ""}</span><b>${pct(f.score)}</b></button>`);
    for (const a of u.alerts) rows.push(`<div class="ro"><span>Same defect again (${posTag(a.position)})</span><b>${a.count} of last ${a.window}</b></div>`);
    const flags = u.shots.filter((s) => s.anomaly_flag != null);
    if (flags.length) rows.push(`<div class="ro"><span>Unusual-spot check</span><b>${flags.some((s) => s.anomaly_flag) ? "unusual on " + flags.filter((s) => s.anomaly_flag).map((s) => esc(s.camera)).join(", ") : "nothing unusual"}</b></div>`);
    for (const s of u.shots) for (const h of s.hints || []) rows.push(`<div class="ro"><span class="hint">${esc(h)}</span></div>`);
    for (const d of (u.decisions || []).filter((d) => d.verdict === "missed" && !d.withdrawn)) rows.push(`<div class="ro"><span>Missed: ${esc(clsName(d.cls))} (${esc(d.camera)}) · ${esc(d.operator)}</span><button type="button" class="label" data-w="${d.id}">Remove</button></div>`);
    if (u.truth) rows.push(`<div class="ro"><span>Simulator really drew</span><b>${u.truth.length ? u.truth.map((t) => "W" + String(t.wire).padStart(2, "0") + " " + esc(clsName(t.cls))).join(", ") : "nothing"}</b></div>`);
    $("readout").innerHTML = rows.join("");
    const k = [];
    const missedStands = (u.decisions || []).some((d) => d.verdict === "missed" && !d.withdrawn);
    if (u.id == null) k.push(`<div class="ro">Not saved (demo): decisions are off.</div>`);
    else if (u.findings.length) {
      const f = u.findings.find((x) => x.id === S.sel);
      k.push(`<button type="button" class="key r ${f && f.decision === "real" ? "done" : ""}" data-d="real">Real defect</button>`,
        `<button type="button" class="key g ${f && f.decision === "false_alarm" ? "done" : ""}" data-d="false_alarm">False alarm</button>`);
      if (!missedStands && u.findings.every((x) => x.decision === "false_alarm")) k.push(`<button type="button" class="key g wide" data-d="confirm_ok">Looks good</button>`);
      k.push(`<button type="button" class="key a wide" data-d="missed">Missed something</button>`);
    } else {
      if (!missedStands) k.push(`<button type="button" class="key g" data-d="confirm_ok">Looks good</button>`);
      k.push(`<button type="button" class="key a ${missedStands ? "wide" : ""}" data-d="missed">Missed something</button>`);
    }
    $("keys").innerHTML = k.join("");
  }

  function show(u, opts = {}) {
    S.unit = u;
    if (!u.findings.some((f) => f.id === S.sel)) S.sel = (u.findings.find((f) => !f.decision) || u.findings[0] || {}).id ?? null;
    renderStage(opts.pop); renderSide(opts.pulse); renderLogCur();
  }
  async function openUnit(id) {
    try { show(await api(`/api/units/${id}`), { pop: true }); go("inspect"); } catch (e) { showError(e.message); }
  }

  async function inspectNext() {
    const b = $("inspect-next"); b.disabled = true; b.textContent = "INSPECTING…";
    try {
      const u = await api("/api/inspect", { method: "POST" });
      S.lastCycle = u.timing_s && u.timing_s.total; $("r-cycle").textContent = S.lastCycle != null ? S.lastCycle.toFixed(1) + " s" : "–";
      showError(""); S.sel = null; show(u, { pop: true, pulse: true });
      loadLog(); loadCounts(); loadStatus();
    } catch (e) {
      showError(e.status === 409 ? "Busy, one inspection is still running." : e.message);
    } finally { b.disabled = false; b.textContent = "INSPECT NEXT"; }
  }

  async function decide(verdict, extra = {}) {
    const u = S.unit; if (!u || u.id == null) return;
    const op = await needOperator(); if (!op) return;
    const body = { verdict, operator: op, ...extra };
    if (verdict === "real" || verdict === "false_alarm") body.finding_id = S.sel;
    try {
      const r = await api(`/api/units/${u.id}/decisions`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
      showError("");
      const next = r.unit.findings.find((f) => !f.decision);
      if (next) S.sel = next.id;
      show(r.unit); loadLog(); loadCounts();
    } catch (e) { showError(e.message); }
  }

  // ---------- "Missed something": draw a box ----------
  async function startMissed() {
    const u = S.unit; if (!u) return;
    if (!(await needOperator())) return;
    const pics = shotsWithPics(u);
    if (S.view === "3d" || S.view === "unusual" || !pics.some((s) => s.camera === S.view)) { S.view = pics[0].camera; renderStage(); }
    const svg = $("stage-body").querySelector("svg"); if (!svg) return;
    const hint = document.createElement("div"); hint.className = "drawing-hint"; hint.textContent = "Drag a box around the defect the AI missed";
    $("stage").appendChild(hint);
    const vb = svg.viewBox.baseVal;
    const toImg = (e) => { const r = svg.getBoundingClientRect();
      return [Math.max(0, Math.min(vb.width, (e.clientX - r.left) / r.width * vb.width)), Math.max(0, Math.min(vb.height, (e.clientY - r.top) / r.height * vb.height))]; };
    let start = null, rect = null;
    const cleanup = () => { svg.onpointerdown = svg.onpointermove = svg.onpointerup = null; hint.remove(); };
    svg.onpointerdown = (e) => { e.preventDefault(); svg.setPointerCapture(e.pointerId); start = toImg(e);
      rect = document.createElementNS("http://www.w3.org/2000/svg", "rect"); rect.setAttribute("class", "box draw"); svg.appendChild(rect); };
    svg.onpointermove = (e) => { if (!start) return; const p = toImg(e);
      rect.setAttribute("x", Math.min(start[0], p[0])); rect.setAttribute("y", Math.min(start[1], p[1]));
      rect.setAttribute("width", Math.abs(p[0] - start[0])); rect.setAttribute("height", Math.abs(p[1] - start[1])); };
    svg.onpointerup = async (e) => {
      if (!start) return; const p = toImg(e); const s0 = start; start = null; cleanup();
      const box = [Math.min(s0[0], p[0]), Math.min(s0[1], p[1]), Math.max(s0[0], p[0]), Math.max(s0[1], p[1])];
      if (box[2] - box[0] < 4 || box[3] - box[1] < 4) { renderStage(); showError("The box was too small. Press Missed something and drag again."); return; }
      const cls = await pickClass(); renderStage();
      if (cls) decide("missed", { camera: S.view, cls, box: box.map((v) => Math.round(v * 10) / 10) });
    };
  }
  function pickClass() {
    return new Promise((resolve) => {
      const m = $("cls-modal"), g = $("cls-grid");
      g.innerHTML = (S.status ? S.status.classes : []).map((c) => `<button type="button" class="key" data-c="${esc(c)}">${esc(clsName(c))}</button>`).join("");
      m.hidden = false;
      const done = (v) => { m.hidden = true; g.onclick = null; $("cls-cancel").onclick = null; resolve(v); };
      g.onclick = (e) => { const b = e.target.closest("[data-c]"); if (b) done(b.dataset.c); };
      $("cls-cancel").onclick = () => done(null);
    });
  }

  // ---------- log strip ----------
  async function loadLog() {
    try {
      const rows = await api("/api/units?limit=12");
      $("log").innerHTML = rows.map((r) => `<button type="button" class="lg ${r.verdict}" data-u="${r.id}">
        <b>${esc(r.source === "upload" ? "Photo" : "#" + r.unit_no)} ${r.verdict}${r.review_state === "decided" ? " ✓" : ""}</b>
        ${esc(r.source === "upload" ? (r.label || "") : time(r.ts))}</button>`).join("") || `<div class="none">No units yet.</div>`;
      renderLogCur();
      if (!S.unit && rows.length) openUnitQuiet(rows[0].id);
    } catch (e) { /* log is optional */ }
  }
  async function openUnitQuiet(id) { try { show(await api(`/api/units/${id}`)); } catch (e) { /* ignore */ } }
  function renderLogCur() { document.querySelectorAll(".lg").forEach((b) => b.classList.toggle("cur", S.unit && +b.dataset.u === S.unit.id)); }

  // ---------- Check a photo ----------
  async function checkFiles(list) {
    const files = [...list]; if (!files.length) return;
    go("check");
    const out = $("results"), cam = $("memory").value;
    if (files.length > 20) { showError("Up to 20 photos at a time; the first 20 are checked."); files.length = 20; }
    let n = 0;
    for (const f of files) {
      $("progress").textContent = `Checking ${++n} of ${files.length}…`;
      const fd = new FormData(); fd.append("files", f, f.name); if (cam) fd.append("camera", cam);
      try { const [r] = await api("/api/check", { method: "POST", body: fd }); out.prepend(card(r)); if (S.demo && files.length === 1) showResult(r); }
      catch (e) { out.prepend(card({ label: f.name, error: e.message })); }
    }
    $("progress").textContent = `Done: ${files.length} photo${files.length > 1 ? "s" : ""} checked.`;
    loadLog(); loadCounts();
  }
  function card(r) {
    const el = document.createElement(r.error ? "div" : "button");
    el.className = "card";
    if (r.error) { el.innerHTML = `<div class="body"><b>${esc(r.label)}</b><span class="err">${esc(r.error)}</span></div>`; return el; }
    el.type = "button";
    if (r.id != null) el.dataset.u = r.id; else { const k = "r" + (S.results.size + 1); S.results.set(k, r); el.dataset.r = k; }
    const shot = r.shots[0];
    if (shot) el.appendChild(photoEl(r, shot));
    const body = document.createElement("div"); body.className = "body";
    body.innerHTML = `<div><span class="vtag ${r.verdict}">${r.verdict}</span> <b>${esc(r.label)}</b></div>`
      + (r.findings.length ? r.findings.map((f) => `<span>${esc(clsName(f.cls))} · ${pct(f.score)}</span>`).join("") : `<span>No defect found</span>`)
      + (r.hints || []).map((h) => `<span class="hint">${esc(h)}</span>`).join("");
    el.appendChild(body);
    return el;
  }
  function wireDrop(el, input) {
    input.addEventListener("change", () => { checkFiles(input.files); input.value = ""; });
    ["dragenter", "dragover"].forEach((t) => el.addEventListener(t, (e) => { e.preventDefault(); el.classList.add("over"); }));
    ["dragleave", "drop"].forEach((t) => el.addEventListener(t, (e) => { e.preventDefault(); el.classList.remove("over"); }));
    el.addEventListener("drop", (e) => checkFiles(e.dataTransfer.files));
  }

  // ---------- Review ----------
  function renderChips() {
    $("chips").innerHTML = FILTERS.map(([k, t]) => `<button type="button" class="chip ${S.filter === k ? "on" : ""}" data-k="${k}">${t}</button>`).join("");
  }
  async function loadReview(more) {
    const q = new URLSearchParams({ limit: 50, ...FILTERS.find((f) => f[0] === S.filter)[2] });
    if (more && S.reviewRows.length) q.set("before", S.reviewRows[S.reviewRows.length - 1].id);
    try {
      const rows = await api("/api/units?" + q);
      S.reviewRows = more ? S.reviewRows.concat(rows) : rows;
      $("review-list").innerHTML = S.reviewRows.map((r) => `<button type="button" class="row ${r.verdict}" data-u="${r.id}">
        <span class="bar"></span><b>${esc(r.source === "upload" ? "Photo " + (r.label || "") : "Unit #" + r.unit_no)}</b>
        <span class="sum">${r.verdict}${r.n_findings ? ` · ${r.n_findings} found` : ""}${r.n_unusual ? " · unusual" : ""}${r.n_alerts ? " · repeat" : ""}</span>
        <span class="t">${new Date(r.ts * 1000).toLocaleString()}</span>
        <span class="tick ${r.review_state === "decided" ? "" : "no"}">${r.review_state === "decided" ? "✓ decided" : "to decide"}</span></button>`).join("")
        || `<div class="none">Nothing here.</div>`;
      $("more").hidden = rows.length < 50;
    } catch (e) { showError(e.message); }
  }

  // ---------- online demo (visitor mode): only "Check a photo", nothing is saved ----------
  async function startDemo() {
    S.demo = true; document.body.classList.add("demo"); $("station").textContent = "ONLINE DEMO";
    $("demo-note").hidden = false;
    go("check");
    try {
      const list = await api("/api/samples");
      $("samples").hidden = !list.length;
      $("sample-row").innerHTML = list.map((s) => `<button type="button" class="sample" data-s="${esc(s.name)}">
        <img alt="" src="${esc(s.url)}"><span>${esc(clsName(s.name.replace(/\.[a-z]+$/i, "").replace(/^macro_/, "close-up ")))}</span></button>`).join("");
    } catch (e) { /* samples are optional */ }
  }
  async function checkSample(name) {
    $("progress").textContent = `Checking ${name}…`;
    const fd = new FormData(); fd.append("sample", name);
    try { const [r] = await api("/api/check", { method: "POST", body: fd }); $("results").prepend(card(r)); showResult(r); $("progress").textContent = ""; }
    catch (e) { showError(e.message); $("progress").textContent = ""; }
  }
  function showResult(r) { S.sel = null; S.view = r.shots[0] ? r.shots[0].camera : "top"; go("inspect"); show(r, { pop: true, pulse: true }); }

  // ---------- routing ----------
  function go(route) {
    if (location.hash !== "#" + route) { history.replaceState(null, "", "#" + route); }
    for (const r of ["inspect", "check", "review"]) {
      $("page-" + r).hidden = r !== route; $("tab-" + r).classList.toggle("on", r === route);
      $("tab-" + r).setAttribute("aria-current", r === route ? "page" : "false");
    }
    if (route === "review") { renderChips(); loadReview(); }
    $("back").hidden = !(S.demo && route === "inspect");
  }
  const route = () => { const r = location.hash.slice(1); return ["inspect", "check", "review"].includes(r) ? r : "inspect"; };

  // ---------- wiring ----------
  function init() {
    paintOperator();
    $("who").onclick = askOperator;
    $("inspect-next").onclick = inspectNext;
    $("views").onclick = (e) => { const b = e.target.closest("[data-v]"); if (b) { S.view = b.dataset.v; renderStage(); } };
    const pickFinding = (e) => {
      const w = e.target.closest("[data-w]");            // "Remove" a wrongly drawn missed box
      if (w) { decide("withdraw", { ref_id: +w.dataset.w }); return; }
      const b = e.target.closest("[data-f]"); if (b) { S.sel = +b.dataset.f; renderStage(); renderSide(); } };
    $("taglines").onclick = pickFinding; $("readout").onclick = pickFinding;
    $("keys").onclick = (e) => { const b = e.target.closest("[data-d]"); if (!b) return;
      if (b.dataset.d === "missed") startMissed(); else decide(b.dataset.d); };
    $("log").onclick = (e) => { const b = e.target.closest("[data-u]"); if (b) openUnit(+b.dataset.u); };
    $("results").onclick = (e) => {
      const d = e.target.closest("[data-r]"); if (d) { showResult(S.results.get(d.dataset.r)); return; }
      const b = e.target.closest("[data-u]"); if (b) openUnit(+b.dataset.u); };
    $("samples").onclick = (e) => { const b = e.target.closest("[data-s]"); if (b) checkSample(b.dataset.s); };
    $("back").onclick = () => go("check");
    $("review-list").onclick = (e) => { const b = e.target.closest("[data-u]"); if (b) openUnit(+b.dataset.u); };
    $("chips").onclick = (e) => { const b = e.target.closest("[data-k]"); if (b) { S.filter = b.dataset.k; renderChips(); loadReview(); } };
    $("more").onclick = () => loadReview(true);
    wireDrop($("drop"), $("files")); wireDrop($("side-slot"), $("side-files"));
    $("files2").addEventListener("change", (e) => { checkFiles(e.target.files); e.target.value = ""; });
    $("camera-in").addEventListener("change", (e) => { checkFiles(e.target.files); e.target.value = ""; });
    window.addEventListener("hashchange", () => go(route()));
    loadStatus().then(() => {
      if (S.status && S.status.demo) { startDemo(); return; }
      go(route()); loadLog(); loadCounts();
      setInterval(() => { loadStatus(); loadCounts(); }, 15000);
    });
  }
  document.addEventListener("DOMContentLoaded", init);
  return { go };
})();
