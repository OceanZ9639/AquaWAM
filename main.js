const moreBtn = document.getElementById("moreBtn");
const moreMenu = document.getElementById("moreMenu");
moreBtn.addEventListener("click", () => {
  const open = moreMenu.classList.toggle("show");
  moreBtn.setAttribute("aria-expanded", open ? "true" : "false");
});
document.addEventListener("click", (event) => {
  if (!event.target.closest(".more")) moreMenu.classList.remove("show");
});
const toTop = document.getElementById("toTop");
toTop.addEventListener("click", () => {
  window.scrollTo({ top: 0, behavior: "smooth" });
});
function updateToTop() {
  toTop.classList.toggle("visible", window.scrollY > 400);
}
window.addEventListener("scroll", updateToTop, { passive: true });
updateToTop();

const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

function fishSVG(id, kind) {
  const body = kind === "tang" ? "#2f6fe4" : "#ff7a1a";
  const tail = kind === "tang" ? "#facc15" : "#ff7a1a";
  const stripes = kind === "tang" ? "" :
    `<g clip-path="url(#${id})" fill="#fff" stroke="#1f2937" stroke-width="1.5">
       <rect x="36" y="0" width="9" height="60"/><rect x="58" y="0" width="8" height="60"/><rect x="80" y="0" width="5" height="60"/>
     </g>`;
  return `<svg viewBox="0 0 100 60" aria-hidden="true">
    <defs><clipPath id="${id}"><ellipse cx="56" cy="30" rx="36" ry="19"/></clipPath></defs>
    <g class="tail"><path d="M22 30 L4 14 Q9 30 4 46 Z" fill="${tail}" stroke="#1f2937" stroke-width="2" stroke-linejoin="round"/></g>
    <path d="M44 13 Q56 0 70 12 Z" fill="${body}" stroke="#1f2937" stroke-width="2" stroke-linejoin="round"/>
    <path d="M48 47 Q58 58 66 47 Z" fill="${body}" stroke="#1f2937" stroke-width="2" stroke-linejoin="round"/>
    <ellipse cx="56" cy="30" rx="36" ry="19" fill="${body}" stroke="#1f2937" stroke-width="2"/>
    ${stripes}
    <ellipse cx="56" cy="30" rx="36" ry="19" fill="none" stroke="#1f2937" stroke-width="2"/>
    <circle cx="80" cy="25" r="3.6" fill="#1f2937"/><circle cx="81.2" cy="23.8" r="1.1" fill="#fff"/>
  </svg>`;
}

const ocean = document.getElementById("ocean");
if (!reduceMotion) {
  const fish = [
    { kind: "clown", w: 110, top: 16, t: 42, d: -6, back: false },
    { kind: "clown", w: 72, top: 24, t: 50, d: -30, back: false },
    { kind: "tang", w: 96, top: 44, t: 46, d: -18, back: true },
    { kind: "clown", w: 130, top: 66, t: 58, d: -40, back: true },
    { kind: "tang", w: 70, top: 82, t: 38, d: -12, back: false },
    { kind: "clown", w: 60, top: 56, t: 34, d: -24, back: false },
  ];
  fish.forEach((f, i) => {
    const el = document.createElement("div");
    el.className = "fish" + (f.back ? " back" : "");
    el.style.top = f.top + "%";
    el.style.setProperty("--w", f.w + "px");
    el.style.setProperty("--t", f.t + "s");
    el.style.setProperty("--d", f.d + "s");
    el.innerHTML = `<div class="fish-inner">${fishSVG("fclip" + i, f.kind)}</div>`;
    ocean.appendChild(el);
  });

  // 鼠标靠近小鱼时，小鱼朝远离鼠标的方向逃开并渐隐，几秒后再游回来
  const fishEls = Array.from(ocean.querySelectorAll(".fish"));
  document.addEventListener("mousemove", (event) => {
    fishEls.forEach((el) => {
      if (el.classList.contains("flee")) return;
      const r = el.getBoundingClientRect();
      const cx = r.left + r.width / 2;
      const cy = r.top + r.height / 2;
      const dx = cx - event.clientX;
      const dy = cy - event.clientY;
      const dist = Math.hypot(dx, dy);
      if (dist > Math.max(90, r.width * 0.9)) return;
      const ux = dx / (dist || 1);
      const uy = dy / (dist || 1);
      const sx = el.classList.contains("back") ? -1 : 1;
      const inner = el.firstElementChild;
      inner.style.transform = `translate(${(ux * 220 * sx).toFixed(0)}px, ${(uy * 160).toFixed(0)}px) scale(0.7)`;
      el.classList.add("flee");
      setTimeout(() => {
        inner.style.transition = "none";
        inner.style.transform = "";
        void inner.offsetWidth;
        inner.style.transition = "";
        el.classList.remove("flee");
      }, 7000);
    });
  }, { passive: true });
  for (let i = 0; i < 18; i++) {
    const b = document.createElement("span");
    b.className = "bubble";
    b.style.left = (Math.random() * 100).toFixed(1) + "%";
    b.style.setProperty("--s", (6 + Math.random() * 12).toFixed(0) + "px");
    b.style.setProperty("--t", (10 + Math.random() * 14).toFixed(1) + "s");
    b.style.setProperty("--d", (-Math.random() * 20).toFixed(1) + "s");
    ocean.appendChild(b);
  }
}

const rov = document.getElementById("rov");
let lastY = window.scrollY;
let vel = 0;
let angle = 0;
let curTop = null;
let lastBubble = 0;

function rovTarget() {
  const max = document.documentElement.scrollHeight - window.innerHeight;
  const p = max > 0 ? Math.min(1, Math.max(0, window.scrollY / max)) : 0;
  return window.innerHeight * (0.14 + 0.6 * p);
}

function spawnTrail(strong) {
  const r = rov.getBoundingClientRect();
  const b = document.createElement("span");
  b.className = "trail";
  b.style.left = (r.left + r.width * (0.08 + Math.random() * 0.25)).toFixed(0) + "px";
  b.style.top = (r.top + r.height * (0.2 + Math.random() * 0.3)).toFixed(0) + "px";
  b.style.setProperty("--s", ((strong ? 7 : 5) + Math.random() * 8).toFixed(0) + "px");
  b.style.setProperty("--dx", ((Math.random() - 0.5) * 40).toFixed(0) + "px");
  b.addEventListener("animationend", () => b.remove());
  document.body.appendChild(b);
}

function frame(t) {
  const y = window.scrollY;
  const dy = y - lastY;
  lastY = y;
  vel = vel * 0.86 + dy * 0.14;
  const targetAngle = Math.max(-26, Math.min(26, vel * 1.4));
  angle += (targetAngle - angle) * 0.1;
  const target = rovTarget() + Math.sin(t / 900) * 6;
  curTop = curTop === null ? target : curTop + (target - curTop) * 0.1;
  rov.style.transform = `translate3d(0, ${curTop.toFixed(1)}px, 0) rotate(${angle.toFixed(2)}deg)`;
  if (rov.getClientRects().length && !reduceMotion) {
    const moving = Math.abs(vel) > 0.5;
    if ((moving && t - lastBubble > 60) || t - lastBubble > 900) {
      spawnTrail(moving);
      lastBubble = t;
    }
  }
  requestAnimationFrame(frame);
}
requestAnimationFrame(frame);

const stage = document.getElementById("stage");
const screen = document.querySelector(".screen");
const title = document.getElementById("stage-title");
const note = document.getElementById("stage-note");

document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.remove("on"));
    tab.classList.add("on");
    title.textContent = tab.dataset.title;
    note.textContent = tab.dataset.note;
    screen.style.maxWidth = tab.dataset.max + "px";
    stage.style.aspectRatio = tab.dataset.ratio;
    stage.src = tab.dataset.src;
    stage.play().catch(() => {});
  });
});

document.getElementById("copy").addEventListener("click", async () => {
  const text = document.getElementById("bib").textContent;
  try {
    await navigator.clipboard.writeText(text);
  } catch (err) {
    const area = document.createElement("textarea");
    area.value = text;
    document.body.appendChild(area);
    area.select();
    document.execCommand("copy");
    area.remove();
  }
  const btn = document.getElementById("copy");
  btn.textContent = "Copied";
  setTimeout(() => { btn.textContent = "Copy BibTeX"; }, 1400);
});
