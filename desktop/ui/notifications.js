export function createNotifications({ document, mount, setTimer, clearTimer }) {
  const active = new Map();
  const timers = new Set();
  let disposed = false;
  function later(fn, delay) {
    const id = setTimer(() => { timers.delete(id); if (!disposed) fn(); }, delay);
    timers.add(id);
    return id;
  }
  function notify(message, { kind = "warning", key = `${kind}:${message}`, persistent = false } = {}) {
    if (disposed || !message || active.has(key)) return;
    while (active.size >= 4) active.values().next().value.dismiss(true);
    const node = document.createElement("div");
    node.className = `toast ${kind} entering`;
    node.setAttribute("role", kind === "error" ? "alert" : "status");
    node.dataset.kind = kind;
    const text = document.createElement("span");
    text.className = "toast-message";
    text.textContent = String(message);
    const close = document.createElement("button");
    close.className = "toast-close icon-button";
    close.setAttribute("type", "button");
    close.setAttribute("aria-label", "Dismiss notification");
    close.textContent = "×";
    const cancelTimers = new Set();
    let leaving = false;
    const dismiss = (immediate = false) => {
      if (leaving && !immediate) return;
      leaving = true;
      for (const id of cancelTimers) { clearTimer(id); timers.delete(id); }
      active.delete(key);
      if (immediate) { node.remove(); return; }
      node.classList.remove("entering");
      node.classList.add("leaving");
      later(() => node.remove(), 180);
    };
    close.addEventListener("click", () => dismiss());
    node.append(text, close);
    mount.append(node);
    active.set(key, { dismiss });
    cancelTimers.add(later(() => node.classList.remove("entering"), 180));
    if (!persistent) cancelTimers.add(later(() => dismiss(), kind === "error" ? 9000 : 6500));
  }
  return {
    notify,
    dispose() {
      disposed = true;
      for (const id of timers) clearTimer(id);
      timers.clear();
      for (const entry of active.values()) entry.dismiss(true);
      active.clear();
      mount.replaceChildren();
    },
  };
}
