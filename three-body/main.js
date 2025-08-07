const canvas = document.getElementById('view');
const ctx = canvas.getContext('2d');
const dpr = Math.max(1, window.devicePixelRatio || 1);
let running = true;

const ui = {
  G: document.getElementById('G'),
  dt: document.getElementById('dt'),
  integrator: document.getElementById('integrator'),
  trailLength: document.getElementById('trailLength'),
  reset: document.getElementById('reset'),
  toggle: document.getElementById('toggle'),
  clearTrails: document.getElementById('clearTrails'),
  showForces: document.getElementById('showForces'),
  showVelocities: document.getElementById('showVelocities'),
  showEnergy: document.getElementById('showEnergy'),
  autoScale: document.getElementById('autoScale'),
  readout: document.getElementById('readout'),
  // ICs
  m1: document.getElementById('m1'), x1: document.getElementById('x1'), y1: document.getElementById('y1'), vx1: document.getElementById('vx1'), vy1: document.getElementById('vy1'), c1: document.getElementById('c1'),
  m2: document.getElementById('m2'), x2: document.getElementById('x2'), y2: document.getElementById('y2'), vx2: document.getElementById('vx2'), vy2: document.getElementById('vy2'), c2: document.getElementById('c2'),
  m3: document.getElementById('m3'), x3: document.getElementById('x3'), y3: document.getElementById('y3'), vx3: document.getElementById('vx3'), vy3: document.getElementById('vy3'), c3: document.getElementById('c3'),
  applyIC: document.getElementById('applyIC'),
  randomIC: document.getElementById('randomIC'),
};

function resize() {
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.floor(rect.width * dpr);
  canvas.height = Math.floor(rect.height * dpr);
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}
window.addEventListener('resize', resize);
resize();

class Body {
  constructor(mass, x, y, vx, vy, color) {
    this.mass = mass;
    this.position = { x, y };
    this.velocity = { x: vx, y: vy };
    this.color = color;
    this.trail = [];
  }
}

function cloneState(bodies) {
  return bodies.map(b => ({
    mass: b.mass,
    position: { x: b.position.x, y: b.position.y },
    velocity: { x: b.velocity.x, y: b.velocity.y },
  }));
}

function computeAccelerations(state, G) {
  const n = state.length;
  const acc = Array.from({ length: n }, () => ({ x: 0, y: 0 }));
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = state[j].position.x - state[i].position.x;
      const dy = state[j].position.y - state[i].position.y;
      const r2 = dx * dx + dy * dy + 1e-6; // softening
      const r = Math.sqrt(r2);
      const f = (G * state[i].mass * state[j].mass) / (r2);
      const fx = f * dx / r;
      const fy = f * dy / r;
      acc[i].x += fx / state[i].mass;
      acc[i].y += fy / state[i].mass;
      acc[j].x -= fx / state[j].mass;
      acc[j].y -= fy / state[j].mass;
    }
  }
  return acc;
}

function stepEuler(bodies, dt, G) {
  const state = cloneState(bodies);
  const acc = computeAccelerations(state, G);
  for (let i = 0; i < bodies.length; i++) {
    bodies[i].velocity.x += acc[i].x * dt;
    bodies[i].velocity.y += acc[i].y * dt;
    bodies[i].position.x += bodies[i].velocity.x * dt;
    bodies[i].position.y += bodies[i].velocity.y * dt;
  }
}

function stepVerlet(bodies, dt, G) {
  const state = cloneState(bodies);
  const a0 = computeAccelerations(state, G);
  const newState = cloneState(bodies);
  for (let i = 0; i < bodies.length; i++) {
    newState[i].position.x += bodies[i].velocity.x * dt + 0.5 * a0[i].x * dt * dt;
    newState[i].position.y += bodies[i].velocity.y * dt + 0.5 * a0[i].y * dt * dt;
  }
  const a1 = computeAccelerations(newState, G);
  for (let i = 0; i < bodies.length; i++) {
    bodies[i].velocity.x += 0.5 * (a0[i].x + a1[i].x) * dt;
    bodies[i].velocity.y += 0.5 * (a0[i].y + a1[i].y) * dt;
    bodies[i].position.x = newState[i].position.x;
    bodies[i].position.y = newState[i].position.y;
  }
}

function stepRK4(bodies, dt, G) {
  const n = bodies.length;
  const state0 = cloneState(bodies);

  function deriv(state) {
    const a = computeAccelerations(state, G);
    return state.map((b, i) => ({
      dx: b.velocity.x,
      dy: b.velocity.y,
      dvx: a[i].x,
      dvy: a[i].y,
    }));
  }

  const k1 = deriv(state0);
  const state1 = state0.map((b, i) => ({
    ...b,
    position: { x: b.position.x + k1[i].dx * dt / 2, y: b.position.y + k1[i].dy * dt / 2 },
    velocity: { x: b.velocity.x + k1[i].dvx * dt / 2, y: b.velocity.y + k1[i].dvy * dt / 2 },
  }));
  const k2 = deriv(state1);
  const state2 = state0.map((b, i) => ({
    ...b,
    position: { x: b.position.x + k2[i].dx * dt / 2, y: b.position.y + k2[i].dy * dt / 2 },
    velocity: { x: b.velocity.x + k2[i].dvx * dt / 2, y: b.velocity.y + k2[i].dvy * dt / 2 },
  }));
  const k3 = deriv(state2);
  const state3 = state0.map((b, i) => ({
    ...b,
    position: { x: b.position.x + k3[i].dx * dt, y: b.position.y + k3[i].dy * dt },
    velocity: { x: b.velocity.x + k3[i].dvx * dt, y: b.velocity.y + k3[i].dvy * dt },
  }));
  const k4 = deriv(state3);

  for (let i = 0; i < n; i++) {
    bodies[i].position.x += (dt / 6) * (k1[i].dx + 2 * k2[i].dx + 2 * k3[i].dx + k4[i].dx);
    bodies[i].position.y += (dt / 6) * (k1[i].dy + 2 * k2[i].dy + 2 * k3[i].dy + k4[i].dy);
    bodies[i].velocity.x += (dt / 6) * (k1[i].dvx + 2 * k2[i].dvx + 2 * k3[i].dvx + k4[i].dvx);
    bodies[i].velocity.y += (dt / 6) * (k1[i].dvy + 2 * k2[i].dvy + 2 * k3[i].dvy + k4[i].dvy);
  }
}

function computeEnergy(bodies, G) {
  let kinetic = 0;
  let potential = 0;
  for (let i = 0; i < bodies.length; i++) {
    const v2 = bodies[i].velocity.x ** 2 + bodies[i].velocity.y ** 2;
    kinetic += 0.5 * bodies[i].mass * v2;
    for (let j = i + 1; j < bodies.length; j++) {
      const dx = bodies[j].position.x - bodies[i].position.x;
      const dy = bodies[j].position.y - bodies[i].position.y;
      const r = Math.sqrt(dx * dx + dy * dy) + 1e-6;
      potential += -G * bodies[i].mass * bodies[j].mass / r;
    }
  }
  return { kinetic, potential, total: kinetic + potential };
}

function createDefaultBodies() {
  return [
    new Body(parseFloat(ui.m1.value), parseFloat(ui.x1.value), parseFloat(ui.y1.value), parseFloat(ui.vx1.value), parseFloat(ui.vy1.value), ui.c1.value),
    new Body(parseFloat(ui.m2.value), parseFloat(ui.x2.value), parseFloat(ui.y2.value), parseFloat(ui.vx2.value), parseFloat(ui.vy2.value), ui.c2.value),
    new Body(parseFloat(ui.m3.value), parseFloat(ui.x3.value), parseFloat(ui.y3.value), parseFloat(ui.vx3.value), parseFloat(ui.vy3.value), ui.c3.value),
  ];
}

let bodies = createDefaultBodies();
let t = 0;

function applyICFromUI() {
  bodies = createDefaultBodies();
  t = 0;
  for (const b of bodies) b.trail = [];
}

function randomStableIC() {
  const m = [1, 1, 1].map(v => 0.5 + Math.random() * 1.5);
  const r = 1 + Math.random() * 1.5;
  const angle = Math.random() * Math.PI * 2;
  const positions = [
    { x: r * Math.cos(angle), y: r * Math.sin(angle) },
    { x: -r * Math.cos(angle), y: -r * Math.sin(angle) },
    { x: 0, y: 0 },
  ];
  const vmag = 0.8 + Math.random() * 0.8;
  const velocities = [
    { x: -vmag * Math.sin(angle), y: vmag * Math.cos(angle) },
    { x: vmag * Math.sin(angle), y: -vmag * Math.cos(angle) },
    { x: 0, y: 0 },
  ];
  ui.m1.value = m[0].toFixed(2); ui.m2.value = m[1].toFixed(2); ui.m3.value = m[2].toFixed(2);
  ui.x1.value = positions[0].x.toFixed(2); ui.y1.value = positions[0].y.toFixed(2);
  ui.x2.value = positions[1].x.toFixed(2); ui.y2.value = positions[1].y.toFixed(2);
  ui.x3.value = positions[2].x.toFixed(2); ui.y3.value = positions[2].y.toFixed(2);
  ui.vx1.value = velocities[0].x.toFixed(2); ui.vy1.value = velocities[0].y.toFixed(2);
  ui.vx2.value = velocities[1].x.toFixed(2); ui.vy2.value = velocities[1].y.toFixed(2);
  ui.vx3.value = velocities[2].x.toFixed(2); ui.vy3.value = velocities[2].y.toFixed(2);
  applyICFromUI();
}

ui.applyIC.addEventListener('click', applyICFromUI);
ui.randomIC.addEventListener('click', randomStableIC);
ui.reset.addEventListener('click', () => { applyICFromUI(); });
ui.toggle.addEventListener('click', () => { running = !running; ui.toggle.textContent = running ? 'Pause' : 'Resume'; });
ui.clearTrails.addEventListener('click', () => { for (const b of bodies) b.trail = []; });

function worldBounds(bodies) {
  let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;
  for (const b of bodies) {
    minX = Math.min(minX, b.position.x);
    maxX = Math.max(maxX, b.position.x);
    minY = Math.min(minY, b.position.y);
    maxY = Math.max(maxY, b.position.y);
  }
  const dx = maxX - minX;
  const dy = maxY - minY;
  const cx = (maxX + minX) / 2;
  const cy = (maxY + minY) / 2;
  const size = Math.max(dx, dy, 2);
  return { cx, cy, size };
}

function draw() {
  const width = canvas.width / dpr;
  const height = canvas.height / dpr;
  ctx.clearRect(0, 0, width, height);

  // Determine scale
  const { cx, cy, size } = worldBounds(bodies);
  const margin = 0.2 * size;
  const worldSize = ui.autoScale.checked ? size + margin : 10;
  const scale = Math.min(width, height) / worldSize;
  const ox = width / 2 - cx * scale;
  const oy = height / 2 - cy * scale;

  function toScreen(p) { return { x: ox + p.x * scale, y: oy + p.y * scale }; }

  // draw grid
  ctx.save();
  ctx.globalAlpha = 0.25;
  ctx.strokeStyle = '#1f2a44';
  ctx.lineWidth = 1;
  const gridStep = Math.pow(10, Math.floor(Math.log10(worldSize / 10)));
  for (let x = Math.floor((cx - worldSize) / gridStep) * gridStep; x <= cx + worldSize; x += gridStep) {
    const s = toScreen({ x, y: 0 });
    ctx.beginPath();
    ctx.moveTo(s.x, 0);
    ctx.lineTo(s.x, height);
    ctx.stroke();
  }
  for (let y = Math.floor((cy - worldSize) / gridStep) * gridStep; y <= cy + worldSize; y += gridStep) {
    const s = toScreen({ x: 0, y });
    ctx.beginPath();
    ctx.moveTo(0, s.y);
    ctx.lineTo(width, s.y);
    ctx.stroke();
  }
  ctx.restore();

  // draw trails and bodies
  for (const b of bodies) {
    // trail
    if (b.trail.length > 1) {
      ctx.beginPath();
      ctx.lineWidth = 1.5;
      ctx.strokeStyle = b.color + 'cc';
      const first = toScreen(b.trail[0]);
      ctx.moveTo(first.x, first.y);
      for (let i = 1; i < b.trail.length; i++) {
        const p = toScreen(b.trail[i]);
        ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();
    }
    // body
    const p = toScreen(b.position);
    ctx.beginPath();
    ctx.fillStyle = b.color;
    const radius = Math.max(3, 3 + Math.log10(b.mass + 1));
    ctx.arc(p.x, p.y, radius, 0, Math.PI * 2);
    ctx.fill();
  }

  // vectors
  if (ui.showVelocities.checked || ui.showForces.checked) {
    const G = parseFloat(ui.G.value);
    const state = cloneState(bodies);
    const acc = computeAccelerations(state, G);
    for (let i = 0; i < bodies.length; i++) {
      const b = bodies[i];
      const p = toScreen(b.position);
      if (ui.showVelocities.checked) {
        ctx.strokeStyle = b.color;
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + b.velocity.x * scale * 0.1, p.y + b.velocity.y * scale * 0.1);
        ctx.stroke();
      }
      if (ui.showForces.checked) {
        ctx.strokeStyle = '#a3e635';
        ctx.lineWidth = 2;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x + acc[i].x * scale * 0.2, p.y + acc[i].y * scale * 0.2);
        ctx.stroke();
      }
    }
  }
}

function tick() {
  if (running) {
    const G = parseFloat(ui.G.value);
    const dt = parseFloat(ui.dt.value);
    const integrator = ui.integrator.value;

    // integrate multiple small steps if dt is large, for stability
    const subSteps = Math.max(1, Math.min(10, Math.ceil(dt / 0.01)));
    const h = dt / subSteps;
    for (let s = 0; s < subSteps; s++) {
      if (integrator === 'rk4') stepRK4(bodies, h, G);
      else if (integrator === 'verlet') stepVerlet(bodies, h, G);
      else stepEuler(bodies, h, G);
      t += h;
      for (const b of bodies) {
        b.trail.push({ x: b.position.x, y: b.position.y });
        if (b.trail.length > parseInt(ui.trailLength.value)) b.trail.shift();
      }
    }
  }
  draw();

  if (ui.showEnergy.checked) {
    const G = parseFloat(ui.G.value);
    const e = computeEnergy(bodies, G);
    ui.readout.textContent = `t = ${t.toFixed(2)}\nE_tot=${e.total.toFixed(4)}  (K=${e.kinetic.toFixed(4)}  U=${e.potential.toFixed(4)})`;
  } else {
    ui.readout.textContent = `t = ${t.toFixed(2)}`;
  }

  requestAnimationFrame(tick);
}

// start
applyICFromUI();
requestAnimationFrame(tick);