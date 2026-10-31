// Debug-only native WebKit exercise; enabled only with an isolated test directory.
(async () => {
  const invoke = globalThis.__TAURI__.core.invoke;
  const checks = [];
  const check = (condition, label) => {
    if (!condition) throw new Error(label);
    checks.push(label);
  };
  const tick = () => new Promise((resolve) => setTimeout(resolve, 150));
  const $ = (id) => document.getElementById(id);
  const title = "Native workspace smoke test";
  const draft = "Unsaved-looking draft, saved locally.\nSecond line.";
  try {
    const deadline = Date.now() + 15000;
    while (!globalThis.__LMA_DESKTOP__ && Date.now() < deadline) await tick();
    const app = globalThis.__LMA_DESKTOP__;
    check(Boolean(app), "desktop controller started");
    await app.ready;
    check(app.store.persistence.hydrated, "workspace hydrated through native IPC");
    check(!app.store.persistence.readError, "no workspace read error");
    check(!app.state.error, "no startup error");
    check(document.querySelectorAll("dialog[open]").length === 0, "startup has no blocking dialog");
    const restored = app.store.current?.title === title;
    if (restored) {
      check(app.store.current.draft === draft, "draft restored across native app restart");
      check(app.store.current.messages.length === 2, "message history restored across native app restart");
      check(app.store.current.settings.temperature === 0.5, "conversation settings restored across native app restart");
    } else {
      const id = app.store.current.id;
      app.store.update(id, (conversation) => {
        conversation.title = title;
        conversation.messages = [
          {role: "user", content: "Show a small Rust example."},
          {role: "assistant", content: "```rust\nfn main() { println!(\"hello\"); }\n```\n\n<img src=x onerror=\"globalThis.__unsafeRan=true\">"},
        ];
      });
      $("draft").value = draft;
      $("draft").dispatchEvent(new Event("input", {bubbles: true}));
      check(app.store.current.draft === draft, "composer input updates its conversation");
    }
    app.setView("chat");
    await tick();
    check($("messages").querySelectorAll("pre code").length === 1, "native Markdown renders fenced code");
    check($("messages").querySelectorAll("script, img, iframe").length === 0 && !globalThis.__unsafeRan, "model HTML stays inert");
    const viewport = () => {
      check(document.documentElement.scrollWidth <= innerWidth + 1, "no horizontal document overflow");
      const composer = $("composer-form").getBoundingClientRect();
      check(composer.width >= 250 && composer.left >= -1 && composer.right <= innerWidth + 1 && composer.bottom <= innerHeight + 1, "composer fits the native window");
    };
    viewport();
    $("nav-models").click();
    check(!$("models-view").hidden && $("chat-view").hidden, "model library is a full-page view");
    const count = $("installed-models").querySelectorAll("[data-model-id]").length;
    check(count === app.state.models.length || count > 0, "installed model rows rendered");
    $("model-search").value = "no-model-can-match-this-smoke-test";
    $("model-search").dispatchEvent(new Event("input", {bubbles: true}));
    check($("installed-models").querySelectorAll("[data-model-id]").length === 0, "model search filters the library");
    $("model-search").value = "";
    $("model-search").dispatchEvent(new Event("input", {bubbles: true}));
    $("nav-chat").click();
    $("conversation-options").click();
    check(!$("inspector").hidden, "conversation options open inline");
    $("setting-temperature").value = "0.5";
    $("setting-temperature").dispatchEvent(new Event("input", {bubbles: true}));
    check(app.store.current.settings.temperature === 0.5, "inspector edits conversation settings");
    $("close-inspector").click();
    check($("inspector").hidden, "inspector closes without blocking chat");
    await invoke("desktop_test_resize", {width: 720, height: 640});
    await tick();
    viewport();
    app.openInspector(true);
    const panel = $("inspector").getBoundingClientRect();
    check(panel.left >= -1 && panel.right <= innerWidth + 1 && panel.bottom <= innerHeight + 1, "inspector fits a narrow native window");
    app.openInspector(false);
    await invoke("desktop_test_resize", {width: 1180, height: 820});
    await tick();
    const original = app.store.current.id;
    $("new-chat").click();
    check(app.store.current.id !== original, "new chat does not replace saved history");
    app.selectConversation(original);
    check(app.store.current.draft === draft, "switching chats preserves the draft");
    check(await app.store.flush(), "workspace saved through native IPC");
    const disk = await invoke("workspace_load");
    const saved = disk.conversations.find((conversation) => conversation.id === original);
    check(saved?.draft === draft && saved.messages.length === 2 && saved.settings.temperature === 0.5, "native disk snapshot matches the workspace");
    await invoke("desktop_test_report", {report: {ok: true, phase: restored ? "restored" : "written", checks, models: count}});
  } catch (error) {
    await invoke("desktop_test_report", {report: {ok: false, error: String(error.message || error), checks}});
  }
})();
