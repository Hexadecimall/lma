import {
  element,
  button,
  decorateIcon,
  renderMarkdown,
  copyText,
  download,
  buildRequest,
  validateSettings,
  validateModelOptions,
  normalizeImage,
  checkImageBudget,
  apiFailure,
  LIMITS,
  DICTATION,
  createPcmRecorder,
} from "./core.js";
import { createWorkspaceStore, uniqueId, markdownExport } from "./workspace.js";
import {
  DEFAULT_APP_SETTINGS,
  validateAppSettings,
  folderLines,
  sourceLabel,
} from "./preferences.js";
import { createNotifications } from "./notifications.js";

export function modelReference(model) {
  return model?.id === `lma:${model?.name}` ? model.id : model?.model || "";
}
export function modelIdentity(model) {
  return model
    ? `${model.ollama ? "ollama" : "lma"}:${modelReference(model)}:${model.ollamaDir || ""}:${model.projector || ""}`
    : "";
}
export function formatSize(bytes) {
  if (!Number.isFinite(bytes) || bytes <= 0) return "Size unknown";
  if (bytes >= 1024 ** 3) return `${(bytes / 1024 ** 3).toFixed(1)} GiB`;
  if (bytes >= 1024 ** 2) return `${Math.round(bytes / 1024 ** 2)} MiB`;
  if (bytes >= 1024) return `${Math.round(bytes / 1024)} KiB`;
  return `${bytes} B`;
}
const errorText = (error) => String(error?.message || error || "Unknown error");

// Injectable IPC and timers make the real DOM/controller lifecycle testable without
// a browser backend, network calls, or a second implementation of the app.
export function createDesktopApp({
  document: doc = globalThis.document,
  window: win = globalThis.window,
  invoke = (command, args) => win.__TAURI__.core.invoke(command, args),
  setTimer = setTimeout,
  clearTimer = clearTimeout,
} = {}) {
  const $ = (id) => doc.getElementById(id);
  const state = {
    view: "chat",
    inspector: false,
    booting: true,
    status: null,
    statusKnown: false,
    models: [],
    selected: null,
    loadedChoice: null,
    remembered: null,
    scanning: false,
    modelBusy: false,
    sourcePending: false,
    update: { state: "Idle", log: "", error: null },
    active: null,
    preparing: false,
    preparingConversationId: null,
    error: "",
    warnings: [],
    appSettings: validateAppSettings(DEFAULT_APP_SETTINGS),
    appSettingsKnown: false,
    appSettingsDirty: false,
    appSettingsSaving: false,
    modelsTab: "installed",
    storeModels: [],
    storeSearching: false,
    storeDownloading: null,
    storeStatus: "",
    storeRevision: 0,
    modelOperation: null,
    importing: false,
    restoreConsidered: false,
    disposed: false,
  };
  const dictation = {
    phase: "idle",
    modelPath: null,
    setup: false,
    saving: false,
    installing: false,
    inspecting: false,
    plan: null,
    models: [],
    scanning: false,
    modelsError: "",
    modelWarnings: [],
    selectedModel: "",
    error: "",
    note: "",
    elapsed: 0,
    activated: false,
  };
  let recorder = null,
    dictationTarget = null,
    dictationTimer = null;
  let renderedDictationPlan = null,
    renderedDictationModels = null;
  let draftRevision = 0;
  const errors = new Map();
  let statusTimer = null,
    updateTimer = null,
    discoveryTimer = null,
    modelIntent = 0,
    discoveryEpoch = 0;
  let messageSignature = "",
    messageConversationId = null,
    messageCount = 0,
    messageEpoch = 0,
    attachmentSignature = "",
    settingsId = null;
  let renderConversationId = null,
    renderContentRevision = -1,
    conversationSnapshot = null;
  function renderSnapshot() {
    if (
      renderConversationId !== store.currentId ||
      renderContentRevision !== store.contentRevision
    ) {
      renderConversationId = store.currentId;
      renderContentRevision = store.contentRevision;
      conversationSnapshot = store.current;
    }
    return conversationSnapshot;
  }
  let startPromise = null;
  let renderedStoreRevision = -1;
  let storeSearchEpoch = 0;
  const storeFiles = new Map();
  const shownWarnings = new Set();
  const deletedModelIdentities = new Set();
  const notifications = createNotifications({
    document: doc,
    mount: $("notifications"),
    setTimer,
    clearTimer,
  });
  const listeners = [];
  const on = (node, event, handler) => {
    node.addEventListener(event, handler);
    listeners.push(() => node.removeEventListener(event, handler));
  };
  const store = createWorkspaceStore({
    invoke,
    setTimer,
    clearTimer,
    onChange: (reason) => {
      if (state.disposed) return;
      if (reason === "draft") {
        draftRevision++;
        renderComposer();
        renderNotices();
      } else if (reason === "persistence") renderNotices();
      else render();
    },
  });
  const busy = () =>
    state.booting ||
    state.modelBusy ||
    state.importing ||
    state.status?.loading ||
    state.status?.updating ||
    state.sourcePending ||
    state.update.state === "Running";
  const locked = () =>
    busy() ||
    !!state.active ||
    state.preparing ||
    dictation.phase === "transcribing";
  const guard =
    (fn) =>
    (...args) => {
      try {
        const result = fn(...args);
        if (result?.catch) result.catch(showError);
      } catch (error) {
        showError(error);
      }
    };
  function showError(error) {
    state.error = errorText(error);
    notifications.notify(state.error, { kind: "error" });
    renderNotices();
  }
  function notifyWarnings(warnings) {
    for (const value of warnings.filter(Boolean)) {
      const message = String(value);
      if (shownWarnings.has(message)) continue;
      if (shownWarnings.size >= 256)
        shownWarnings.delete(shownWarnings.values().next().value);
      shownWarnings.add(message);
      notifications.notify(message);
    }
  }
  function renderNotices() {
    const p = store.persistence;
    const persistenceError = p.readError || p.draftError || p.saveError;
    $("persistence-error").hidden = !persistenceError;
    $("persistence-error-text").textContent = persistenceError;
    $("persistence-retry").disabled = p.loading || p.saving;
    $("app-error").hidden = true;
    $("app-error-text").textContent = state.error;
    $("save-state").textContent = persistenceError
      ? "Not saved"
      : p.saving
        ? "Saving…"
        : p.dirty
          ? "Unsaved"
          : "";
  }
  function setView(view) {
    if (!["chat", "models", "settings"].includes(view)) return;
    void store.flush();
    state.view = view;
    if (view !== "chat") state.inspector = false;
    if (view === "settings" && !state.appSettingsDirty) fillAppSettings();
    render();
  }
  function openInspector(open = !state.inspector) {
    state.inspector = open;
    if (open) fillSettings();
    render();
    if (!open) $("conversation-options").focus();
  }
  function selectConversation(id) {
    store.select(id);
    state.view = "chat";
    fillSettings();
    render();
  }
  function newChat() {
    if (state.booting) return;
    void store.flush();
    store.add();
    state.view = "chat";
    fillSettings();
    render();
    $("draft").focus();
  }
  function removeConversation(id) {
    if (
      state.booting ||
      state.active?.conversationId === id ||
      state.preparingConversationId === id
    )
      return;
    if (
      !win.confirm(
        "Delete this conversation and its attachments from this device?",
      )
    )
      return;
    store.remove(id);
    if (!store.current) store.add();
    fillSettings();
    render();
  }
  function renameConversation(id) {
    selectConversation(id);
    openInspector(true);
    $("rename-title").focus();
    $("rename-title").select();
  }
  function renderSidebar() {
    const query = $("conversation-search").value.toLowerCase().trim();
    const list = $("conversation-list");
    const w = { currentId: store.currentId };
    const conversations = store.listConversations(query);
    list.replaceChildren();
    for (const c of conversations) {
      const row = element(
        "div",
        `conversation-row${c.id === w.currentId ? " selected" : ""}`,
      );
      row.dataset.conversationId = c.id;
      const select = button(
        c.title,
        () => selectConversation(c.id),
        "conversation-select",
      );
      select.title = c.title;
      if (c.id === w.currentId) select.setAttribute("aria-current", "true");
      const actions = element("div", "row-actions");
      const rename = button("Rename conversation", () =>
        renameConversation(c.id),
      );
      decorateIcon(rename, "Edit", `Rename ${c.title}`);
      const remove = button(
        "Delete conversation",
        guard(() => removeConversation(c.id)),
      );
      decorateIcon(remove, "trash", `Delete ${c.title}`);
      remove.disabled =
        state.active?.conversationId === c.id ||
        state.preparingConversationId === c.id;
      actions.append(rename, remove);
      row.append(select, actions);
      list.append(row);
    }
    if (!conversations.length)
      list.append(
        element(
          "p",
          "empty-list",
          query
            ? "No matches"
            : state.booting
              ? "Opening workspace…"
              : "No conversations yet",
        ),
      );
  }
  function imageFigure(image, remove) {
    const figure = element("figure", "image-figure");
    const img = element("img");
    img.src = `data:${image.mime};base64,${image.base64}`;
    img.alt = image.name;
    figure.append(img);
    if (remove) {
      const control = button("Remove image", guard(remove));
      decorateIcon(control, "close", `Remove ${image.name}`);
      figure.append(control);
    }
    return figure;
  }
  function renderAttachments(c) {
    const signature = `${c?.id}:${store.contentRevision}:${state.preparing}`;
    if (signature === attachmentSignature) return;
    attachmentSignature = signature;
    const node = $("attachments");
    node.replaceChildren();
    node.hidden = !c?.attachments.length;
    c?.attachments.forEach((image, index) =>
      node.append(
        imageFigure(image, () => {
          if (state.preparing) return;
          store.update(c.id, (conversation) =>
            conversation.attachments.splice(index, 1),
          );
        }),
      ),
    );
  }
  function renderMessages(c) {
    const pending = state.active?.conversationId === c?.id;
    const signature = `${c?.id}:${store.contentRevision}:${pending}:${locked()}:${state.booting}`;
    if (signature === messageSignature) return;
    const scroller = $("message-scroll");
    const nearBottom =
      scroller.scrollHeight - scroller.scrollTop - scroller.clientHeight < 100;
    const changedConversation = messageConversationId !== c?.id;
    if (changedConversation) messageEpoch++;
    const epoch = messageEpoch;
    const conversationId = c?.id;
    const entering = [];
    const previousCount = changedConversation ? 0 : messageCount;
    messageConversationId = c?.id;
    messageCount = c?.messages.length || 0;
    messageSignature = signature;
    const container = $("messages");
    container.replaceChildren();
    $("welcome-view").hidden = !!c?.messages.length || pending;
    scroller.classList.toggle("is-empty", !c?.messages.length && !pending);
    c?.messages.forEach((message, index) => {
      const article = element("article", `message ${message.role}`);
      if (!changedConversation && !state.booting && index >= previousCount)
        entering.push(index);
      article.dataset.messageIndex = String(index);
      const heading = element("div", "message-heading");
      heading.append(
        element(
          "span",
          "message-role",
          message.role === "user" ? "You" : "Assistant",
        ),
      );
      if (message.mode === "raw")
        heading.append(element("span", "badge", "Raw"));
      const body = element("div", "message-body");
      if (message.role === "assistant")
        body.append(renderMarkdown(message.content));
      else body.textContent = message.content;
      if (message.images?.length) {
        const images = element("div", "message-images");
        message.images.forEach((image) => images.append(imageFigure(image)));
        body.append(images);
      }
      const footer = element("div", "message-footer");
      const actions = element("div", "message-actions");
      actions.append(
        button(
          "Copy",
          (event) => void copyText(message.content, event.currentTarget),
        ),
        button("Download", () =>
          download(
            message.content,
            `${message.role}-${index + 1}.md`,
            "text/markdown;charset=utf-8",
          ),
        ),
      );
      const branch = button(
        message.role === "user" ? "Edit" : "Regenerate",
        guard(() => branchAt(index, message.role === "assistant")),
      );
      branch.setAttribute(
        "aria-label",
        message.role === "user"
          ? "Edit as a new branch"
          : "Regenerate as a new branch",
      );
      branch.title = branch.getAttribute("aria-label");
      branch.disabled = message.role === "assistant" ? locked() : state.booting;
      actions.append(branch);
      const stats = [];
      if (message.model) stats.push(message.model.split(/[\\/]/).pop());
      if (message.elapsed !== undefined)
        stats.push(`${(message.elapsed / 1000).toFixed(1)}s`);
      if (message.usage?.completion_tokens !== undefined)
        stats.push(`${message.usage.completion_tokens} tokens`);
      footer.append(
        actions,
        element("span", "message-stats", stats.join(" · ")),
      );
      article.append(heading, body, footer);
      container.append(article);
    });
    if (pending) {
      const node = element("div", "generation-pending");
      node.setAttribute("role", "status");
      node.append(
        element("span", "pending-dot"),
        element("span", "", "Generating…"),
      );
      container.append(node);
    }
    if (entering.length) {
      // Request-state updates can rebuild this history synchronously. Mark the
      // final DOM once, before paint, rather than lose or restart entry motion.
      queueMicrotask(() => {
        if (
          state.disposed ||
          epoch !== messageEpoch ||
          conversationId !== messageConversationId
        )
          return;
        for (const index of entering)
          container.children[index]?.classList.add("message-enter");
      });
    }
    if (nearBottom || changedConversation || pending)
      scroller.scrollTop = scroller.scrollHeight;
  }
  function isLoaded(model) {
    if (!state.status?.loaded) return false;
    if (state.loadedChoice)
      return modelIdentity(model) === modelIdentity(state.loadedChoice);
    const current = state.status.model;
    const reference = modelReference(model);
    return [reference, model.name, reference.split(/[\\/]/).pop()].includes(
      current,
    );
  }
  function modelChoices() {
    const models = [...state.models];
    for (const m of [state.remembered, state.loadedChoice]) {
      if (m && !models.some((v) => modelIdentity(v) === modelIdentity(m)))
        models.push({
          ...m,
          id: modelIdentity(m),
          name: m.model.split(/[\\/]/).pop(),
          source: m.ollama ? "ollama" : "lma",
        });
    }
    return models;
  }
  function renderQuickSwitch() {
    const select = $("model-quick-switch");
    const choices = modelChoices();
    select.replaceChildren();
    select.value = "";
    const placeholder = element(
      "option",
      "",
      state.status?.loaded
        ? state.status.model || "Loaded model"
        : "Select model",
    );
    placeholder.value = "";
    select.append(placeholder);
    for (const model of choices) {
      const option = element(
        "option",
        "",
        `${model.name || model.model} · ${sourceLabel(model.source || (model.ollama ? "ollama" : "lma"))}`,
      );
      option.value = modelIdentity(model);
      select.append(option);
      if (isLoaded(model)) select.value = option.value;
    }
    const current = choices.find((model) => isLoaded(model));
    const label = current
      ? current.name || current.model
      : placeholder.textContent;
    $("model-quick-switch-label").textContent = label;
    select.title = current
      ? `${label} · ${sourceLabel(current.source || (current.ollama ? "ollama" : "lma"))}`
      : label;
    select.disabled = locked();
  }
  function selectedInstalledModel() {
    return state.models.find(
      (model) =>
        typeof model.id === "string" &&
        model.id &&
        model.id === state.selected?.id &&
        modelIdentity(model) === modelIdentity(state.selected),
    );
  }
  const deletionLocked = () =>
    state.disposed ||
    locked() ||
    state.scanning ||
    !!state.storeDownloading ||
    !state.statusKnown ||
    !!state.status?.loaded;
  function renderModels() {
    const query = $("model-search").value.toLowerCase().trim();
    const list = $("installed-models");
    list.replaceChildren();
    const choices = modelChoices().filter((m) =>
      `${m.name} ${m.model} ${m.source}`.toLowerCase().includes(query),
    );
    for (const model of choices) {
      const selected = modelIdentity(model) === modelIdentity(state.selected);
      const row = button(
        "",
        () => {
          modelIntent++;
          state.selected = model;
          renderModels();
        },
        `model-row${selected ? " selected" : ""}`,
      );
      row.dataset.modelId = model.id || modelIdentity(model);
      row.setAttribute("aria-pressed", String(selected));
      const info = element("span", "model-info");
      info.append(element("span", "model-name", model.name || model.model));
      const meta = element("span", "model-meta");
      meta.append(
        element(
          "span",
          "badge",
          sourceLabel(model.source || (model.ollama ? "ollama" : "lma")),
        ),
        element("span", "", formatSize(model.sizeBytes)),
      );
      if (isLoaded(model))
        meta.append(element("span", "badge loaded-badge", "Loaded"));
      if (model.projector) meta.append(element("span", "badge", "Vision"));
      info.append(meta);
      row.append(info);
      list.append(row);
    }
    if (!choices.length) {
      const empty = element("div", "no-models");
      empty.append(
        element(
          "h3",
          "",
          state.scanning
            ? "Scanning local models…"
            : query
              ? "No matching models"
              : "No installed models found",
        ),
        element(
          "p",
          "",
          query
            ? "Try another name or source."
            : "Import a model or open the Store.",
        ),
      );
      list.append(empty);
    }
    $("model-count").textContent = String(state.models.length || "");
    $("selected-model-name").textContent =
      state.selected?.name || state.selected?.model || "No model selected";
    $("selected-model-path").textContent = state.selected?.model || "";
    const selectedLoaded = state.selected
      ? isLoaded(state.selected)
      : !!state.status?.loaded;
    $("load-panel").hidden = !state.selected && !state.status?.loaded;
    $("load-model").disabled =
      locked() || (!state.selected && !state.status?.loaded);
    $("load-model").textContent = state.modelBusy
      ? state.modelOperation === "delete"
        ? "Load model"
        : state.modelOperation === "unload"
          ? "Unloading…"
          : "Loading…"
      : selectedLoaded
        ? "Unload model"
        : "Load model";
    $("load-model").dataset.action = selectedLoaded ? "unload" : "load";
    $("model-advanced").hidden = state.appSettings.loadPolicy !== "manual";
    $("load-policy-preview").textContent =
      state.appSettings.loadPolicy === "automatic"
        ? "Automatic"
        : state.appSettings.loadPolicy === "defaults"
          ? "Defaults file"
          : "Manual load options";
    $("delete-model").hidden = !selectedInstalledModel();
    $("delete-model").disabled = deletionLocked() || !selectedInstalledModel();
    $("delete-model").textContent =
      state.modelBusy && state.modelOperation === "delete"
        ? "Deleting…"
        : "Delete model";
    $("delete-model-note").textContent = state.status?.loaded
      ? "Unload the loaded model first. Models can only be deleted when no model is loaded."
      : "Deletion is permanent and is only available when no model is loaded and the app is idle.";
    $("unload-model").disabled = !state.status?.loaded || locked();
    $("rescan-models").disabled = state.scanning || busy();
    $("rescan-models").textContent = state.scanning ? "Scanning…" : "Rescan";
    $("manual-load").disabled = locked();
    $("load-gpu").disabled = locked();
    $("load-context").disabled = locked();
    $("discovery-warning").hidden = true;
    $("discovery-warning").textContent = "";
    notifyWarnings([...state.warnings, state.status?.preferenceWarning]);
  }
  function setModelsTab(tab) {
    if (!["installed", "store"].includes(tab)) return;
    state.modelsTab = tab;
    for (const name of ["installed", "store"]) {
      $(
        name === "installed" ? "installed-models-panel" : "store-models-panel",
      ).hidden = name !== tab;
      const button = $(`models-${name}-tab`);
      button.setAttribute("aria-selected", String(name === tab));
      button.setAttribute("tabindex", name === tab ? "0" : "-1");
    }
  }
  function fillAppSettings() {
    const settings = state.appSettings;
    $("app-load-policy").value = settings.loadPolicy;
    $("app-defaults-path").value = settings.defaultsPath || "";
    $("app-gpu").value = settings.gpu;
    $("app-context").value = String(settings.context);
    $("app-auto-restore").checked = settings.autoRestore;
    $("app-model-dirs").value = settings.modelDirectories.join("\n");
    $("app-ollama-dir").value = settings.ollamaDirectory || "";
    $("app-lm-studio-dirs").value = settings.lmStudioDirectories.join("\n");
    renderAppSettings();
  }
  function renderAppSettings() {
    const policy = $("app-load-policy").value;
    $("app-defaults-row").hidden = policy !== "defaults";
    $("app-manual-load-fields").hidden = policy !== "manual";
    $("load-policy-note").textContent =
      policy === "defaults"
        ? "JSON fields: gpu (or gpu_layers), context. Default: ~/.lma/config.json."
        : policy === "manual"
          ? "Applied the next time a model loads."
          : "Uses the compiled GPU and context defaults.";
    $("save-app-settings").disabled =
      state.booting || state.appSettingsSaving || !state.appSettingsKnown;
    $("save-app-settings").textContent = state.appSettingsSaving
      ? "Saving…"
      : "Save settings";
    $("model-backend").textContent = state.status?.loaded
      ? state.status.backend || ""
      : "";
  }
  async function loadAppSettings() {
    try {
      const settings = validateAppSettings(await invoke("app_settings_load"));
      if (state.disposed) return;
      state.appSettings = settings;
      state.appSettingsKnown = true;
      if (!state.appSettingsDirty) fillAppSettings();
      $("load-gpu").value = settings.gpu;
      $("load-context").value = String(settings.context);
    } catch (error) {
      if (state.disposed) return;
      $("app-settings-status").textContent = errorText(error);
      showError(`Could not read application settings. ${errorText(error)}`);
    }
  }
  async function saveAppSettings() {
    if (state.appSettingsSaving || !state.appSettingsKnown) return;
    let settings;
    try {
      settings = validateAppSettings({
        loadPolicy: $("app-load-policy").value,
        defaultsPath: $("app-defaults-path").value,
        autoRestore: $("app-auto-restore").checked,
        gpu:
          $("app-load-policy").value === "manual"
            ? $("app-gpu").value
            : state.appSettings.gpu,
        context:
          $("app-load-policy").value === "manual"
            ? $("app-context").value
            : state.appSettings.context,
        modelDirectories: folderLines($("app-model-dirs").value),
        ollamaDirectory: $("app-ollama-dir").value,
        lmStudioDirectories: folderLines($("app-lm-studio-dirs").value),
      });
    } catch (error) {
      $("app-settings-status").textContent = errorText(error);
      return;
    }
    state.appSettingsSaving = true;
    $("app-settings-status").textContent = "";
    renderAppSettings();
    try {
      await invoke("app_settings_save", { settings });
      if (state.disposed) return;
      state.appSettings = settings;
      state.appSettingsDirty = false;
      $("load-gpu").value = settings.gpu;
      $("load-context").value = String(settings.context);
      $("app-settings-status").textContent = "Saved";
      notifications.notify("Settings saved", { kind: "success" });
      await discover();
    } catch (error) {
      if (!state.disposed)
        $("app-settings-status").textContent = errorText(error);
    } finally {
      state.appSettingsSaving = false;
      if (!state.disposed) render();
    }
  }
  function storeKey(model) {
    return `${model.source}:${model.id}`;
  }
  function storeModelInstalled(model, file) {
    if (model.source !== "ollama") {
      const selectedFile =
        file || (model.files?.length === 1 ? model.files[0].name : null);
      return (
        !!selectedFile &&
        !!model.files?.find((item) => item.name === selectedFile)?.installed
      );
    }
    return !!model.installed;
  }
  function renderStore() {
    $("store-search").disabled = state.storeSearching;
    $("store-search").textContent = state.storeSearching
      ? "Searching…"
      : "Search";
    $("store-status").textContent = state.storeStatus;
    if (renderedStoreRevision === state.storeRevision) return;
    renderedStoreRevision = state.storeRevision;
    const container = $("store-results");
    container.replaceChildren();
    for (const model of state.storeModels) {
      const key = storeKey(model);
      const card = element("article", "store-model-card");
      card.dataset.storeId = key;
      const title = element("h3", "", model.name || model.id);
      const meta = element(
        "p",
        "",
        `${model.source === "ollama" ? "Ollama" : "Hugging Face"}${model.totalBytes ? ` · ${formatSize(model.totalBytes)}` : ""}`,
      );
      const actions = element("div", "store-model-actions");
      let file = null;
      if (model.source !== "ollama" && model.files?.length > 1) {
        file = element("select");
        file.setAttribute("aria-label", `File for ${model.name || model.id}`);
        const placeholder = element("option", "", "Select file");
        placeholder.value = "";
        file.append(placeholder);
        for (const item of model.files) {
          const option = element(
            "option",
            "",
            `${item.name}${item.bytes ? ` · ${formatSize(item.bytes)}` : ""}${item.installed ? " · Installed" : ""}`,
          );
          option.value = item.name;
          file.append(option);
        }
        file.value = storeFiles.get(key) || "";
        file.disabled = !!state.storeDownloading;
        file.addEventListener("change", () => {
          storeFiles.set(key, file.value);
          const installed = storeModelInstalled(model, file.value);
          downloadButton.textContent = installed ? "Installed" : "Download";
          downloadButton.disabled =
            !!state.storeDownloading || !file.value || installed;
        });
        actions.append(file);
      }
      const downloadButton = button(
        !model.exact
          ? "View files"
          : storeModelInstalled(model, file?.value)
            ? "Installed"
            : !model.files?.length
              ? "Unavailable"
              : state.storeDownloading === key
                ? "Downloading…"
                : "Download",
        guard(() => {
          if (state.disposed) return;
          if (!model.exact) return searchStore(model.id, model.source);
          const selectedFile =
            file?.value ||
            (model.source !== "ollama" && model.files?.length === 1
              ? model.files[0].name
              : null);
          return downloadStoreModel(model, selectedFile);
        }),
        storeModelInstalled(model, file?.value)
          ? "secondary-button"
          : "primary-button",
      );
      downloadButton.disabled =
        !!state.storeDownloading ||
        (model.exact &&
          (storeModelInstalled(model, file?.value) ||
            !model.files?.length ||
            (!!file && !file.value)));
      actions.append(downloadButton);
      card.append(title, meta, actions);
      if (state.storeDownloading === key) {
        const progress = element("progress");
        progress.setAttribute(
          "aria-label",
          `Downloading ${model.name || model.id}`,
        );
        card.append(progress);
      }
      container.append(card);
    }
  }
  async function searchStore(
    query = $("store-query").value,
    source = $("store-source").value,
  ) {
    query = query.trim();
    if (!query || query.length > 512) {
      state.storeStatus = "Enter a model name or owner/repository.";
      renderStore();
      return;
    }
    const epoch = ++storeSearchEpoch;
    state.storeSearching = true;
    if (!state.storeDownloading) state.storeStatus = "";
    $("store-query").value = query;
    $("store-source").value = source;
    renderStore();
    try {
      const result = await invoke("store_search", { query, source });
      if (state.disposed || epoch !== storeSearchEpoch) return;
      state.storeModels = Array.isArray(result.models) ? result.models : [];
      state.storeRevision++;
      if (!state.storeDownloading)
        state.storeStatus = state.storeModels.length
          ? ""
          : "No matching models.";
      notifyWarnings(Array.isArray(result.warnings) ? result.warnings : []);
    } catch (error) {
      if (state.disposed || epoch !== storeSearchEpoch) return;
      state.storeStatus = errorText(error);
      showError(error);
    } finally {
      if (epoch === storeSearchEpoch) state.storeSearching = false;
      if (!state.disposed) renderStore();
    }
  }
  async function downloadStoreModel(model, file) {
    if (state.storeDownloading || storeModelInstalled(model, file) || busy())
      return;
    const key = storeKey(model);
    state.storeDownloading = key;
    state.storeStatus = `Downloading ${model.name || model.id}…`;
    state.storeRevision++;
    renderStore();
    try {
      await invoke("store_download", {
        id: model.id,
        source: model.source,
        file,
      });
      if (state.disposed) return;
      model.installed = true;
      if (file) {
        const installedFile = model.files?.find((item) => item.name === file);
        if (installedFile) installedFile.installed = true;
      }
      state.storeStatus = "Downloaded";
      notifications.notify(`${model.name || model.id} downloaded`, {
        kind: "success",
      });
      await discover();
    } catch (error) {
      if (!state.disposed) {
        state.storeStatus = errorText(error);
        showError(error);
      }
    } finally {
      state.storeDownloading = null;
      state.storeRevision++;
      if (!state.disposed) renderStore();
    }
  }
  function scheduleDiscovery() {
    if (state.disposed || discoveryTimer !== null) return;
    discoveryTimer = setTimer(async () => {
      discoveryTimer = null;
      if (state.disposed) return;
      if (!doc.hidden && !busy()) await discover();
      scheduleDiscovery();
    }, 30000);
  }
  function fillSettings() {
    const c = store.composer;
    settingsId = c?.id;
    if (!c) return;
    $("rename-title").value = c.title;
    for (const [id, key] of [
      ["setting-mode", "mode"],
      ["setting-system", "system"],
      ["setting-max-tokens", "max_tokens"],
      ["setting-temperature", "temperature"],
      ["setting-top-p", "top_p"],
      ["setting-seed", "seed"],
    ])
      $(id).value = String(c.settings[key]);
    $("setting-coding").checked = c.settings.coding;
    $("settings-error").hidden = true;
    $("raw-note").hidden = c.settings.mode !== "raw";
  }
  function saveSettings() {
    const c = store.composer;
    if (!c) return;
    try {
      const settings = validateSettings({
        mode: $("setting-mode").value,
        system: $("setting-system").value,
        coding: $("setting-coding").checked,
        max_tokens: $("setting-max-tokens").value,
        temperature: $("setting-temperature").value,
        top_p: $("setting-top-p").value,
        seed: $("setting-seed").value,
      });
      store.update(c.id, (conversation) => {
        conversation.settings = settings;
      });
      $("settings-error").hidden = true;
      $("raw-note").hidden = settings.mode !== "raw";
    } catch (error) {
      $("settings-error").textContent = errorText(error);
      $("settings-error").hidden = false;
    }
  }
  function renderUpdate() {
    $("update-state").textContent = state.update.state;
    $("update-error").hidden = !state.update.error;
    $("update-error").textContent = state.update.error || "";
    const text = [state.update.log, state.update.error]
      .filter(Boolean)
      .join("\n");
    $("update-log").hidden = !text;
    if ($("update-log").textContent !== text) {
      $("update-log").textContent = text;
      $("update-log").scrollTop = $("update-log").scrollHeight;
    }
    $("source-update").disabled = locked();
    $("source-update").textContent =
      state.sourcePending || state.update.state === "Running"
        ? "Updating…"
        : "Update application…";
  }
  function clearDictationTimer() {
    if (dictationTimer !== null) clearTimer(dictationTimer);
    dictationTimer = null;
  }
  function cancelDictationCapture() {
    if (dictationTarget) dictationTarget.valid = false;
    if (dictation.phase === "transcribing") return;
    recorder?.cancel();
    recorder = null;
    dictationTarget = null;
    clearDictationTimer();
    dictation.phase = "idle";
    dictation.elapsed = 0;
    dictation.note = "";
  }
  function renderDictation() {
    if (
      dictationTarget &&
      (dictationTarget.id !== store.currentId ||
        state.view !== "chat" ||
        busy())
    )
      cancelDictationCapture();
    const capturing = ["requesting", "recording"].includes(dictation.phase);
    const label = capturing
      ? "Stop dictation"
      : dictation.phase === "transcribing"
        ? "Transcribing"
        : "Start dictation";
    const control = $("dictation-toggle");
    control.setAttribute("aria-label", label);
    control.title = `${label} (Ctrl/Cmd+Shift+D)`;
    control.setAttribute("aria-pressed", String(capturing));
    control.setAttribute(
      "aria-busy",
      String(dictation.phase === "transcribing"),
    );
    control.classList.toggle("recording", capturing);
    control.classList.toggle(
      "transcribing",
      dictation.phase === "transcribing",
    );
    control.disabled =
      state.booting ||
      !store.currentId ||
      dictation.saving ||
      dictation.phase === "transcribing" ||
      (!capturing && locked());
    $("dictation-configure").disabled =
      state.booting || dictation.phase !== "idle" || dictation.saving;
    $("dictation-configure").setAttribute(
      "aria-expanded",
      String(dictation.setup),
    );
    $("dictation-setup").hidden = !dictation.setup;
    $("dictation-setup").setAttribute("aria-busy", String(dictation.saving));
    const plan =
      dictation.plan?.repository === $("dictation-repository").value.trim()
        ? dictation.plan
        : null;
    $("dictation-install-model").disabled = dictation.saving;
    $("dictation-install-model").setAttribute(
      "aria-busy",
      String(dictation.saving),
    );
    $("dictation-install-model").textContent = dictation.inspecting
      ? "Inspecting…"
      : dictation.installing
        ? "Installing…"
        : plan
          ? `Install (${(plan.total_bytes / 1048576).toFixed(1)} MiB)`
          : "Inspect download";
    $("dictation-repository").disabled = dictation.saving;
    $("dictation-plan").hidden = !plan;
    if (renderedDictationPlan !== plan) {
      renderedDictationPlan = plan;
      $("dictation-plan-summary").textContent = plan
        ? `${plan.repository} · revision ${plan.revision} · ${plan.files.length} files · ${plan.total_bytes} bytes (${(plan.total_bytes / 1048576).toFixed(1)} MiB)`
        : "";
      $("dictation-plan-directory").textContent = plan
        ? `Destination: ${plan.directory}`
        : "";
      $("dictation-plan-files").replaceChildren();
      for (const file of plan?.files || [])
        $("dictation-plan-files").append(
          element("li", "", `${file.name} · ${file.bytes} bytes`),
        );
    }
    if (renderedDictationModels !== dictation.models) {
      renderedDictationModels = dictation.models;
      const select = $("dictation-installed-models");
      select.replaceChildren(
        element("option", "", "Choose an installed speech model"),
      );
      select.firstChild.value = "";
      for (const model of dictation.models) {
        const option = element(
          "option",
          "",
          `${model.name} — ${model.modelPath}`,
        );
        option.value = model.modelPath;
        select.append(option);
      }
    }
    $("dictation-installed-models").value = dictation.selectedModel;
    $("dictation-installed-models").disabled =
      dictation.saving || dictation.scanning || !dictation.models.length;
    $("dictation-use-installed").disabled =
      dictation.saving || dictation.scanning || !dictation.selectedModel;
    $("dictation-rescan-models").disabled =
      dictation.saving || dictation.scanning;
    $("dictation-models-status").textContent = dictation.scanning
      ? "Scanning installed speech models…"
      : dictation.modelsError ||
        [
          dictation.models.length
            ? `${dictation.models.length} compatible installed speech models`
            : "No compatible installed speech models found.",
          ...dictation.modelWarnings,
        ].join(" · ");
    $("dictation-save-model").disabled = dictation.saving;
    $("dictation-model-path").disabled = dictation.saving;
    $("dictation-close-setup").disabled = dictation.saving;
    const status = dictation.inspecting
      ? "Inspecting repository metadata…"
      : dictation.installing
        ? "Downloading listed model files…"
        : dictation.phase === "recording"
          ? "Recording"
          : dictation.phase === "transcribing"
            ? "Transcribing"
            : dictation.phase === "requesting"
              ? "Microphone permission…"
              : dictation.note;
    $("dictation-status").textContent = status;
    const seconds = Math.floor(dictation.elapsed);
    $("dictation-elapsed").textContent = ["recording", "transcribing"].includes(
      dictation.phase,
    )
      ? `${Math.floor(seconds / 60)}:${String(seconds % 60).padStart(2, "0")}`
      : "";
    $("dictation-error").textContent = dictation.error;
    $("dictation-feedback").hidden = !status && !dictation.error;
  }
  function tickDictation(started) {
    clearDictationTimer();
    dictationTimer = setTimer(() => {
      dictationTimer = null;
      if (state.disposed || dictation.phase === "idle") return;
      dictation.elapsed = (Date.now() - started) / 1000;
      if (
        dictation.phase === "recording" &&
        dictation.elapsed >= DICTATION.seconds
      )
        void finishDictation();
      else {
        renderDictation();
        tickDictation(started);
      }
    }, 500);
  }
  function openDictationSetup() {
    if (dictation.phase !== "idle" || dictation.saving) return;
    dictation.setup = !dictation.setup;
    dictation.error = "";
    dictation.note = "";
    dictation.plan = null;
    if (dictation.setup)
      $("dictation-model-path").value = dictation.modelPath || "";
    renderDictation();
    if (dictation.setup) {
      void discoverDictationModels();
      $("dictation-installed-models").focus();
    }
  }
  async function discoverDictationModels() {
    if (state.disposed || dictation.scanning || dictation.saving) return;
    dictation.scanning = true;
    dictation.modelsError = "";
    renderDictation();
    try {
      const catalog = await invoke("dictation_models");
      if (state.disposed) return;
      dictation.models = catalog.models;
      dictation.modelWarnings = catalog.warnings;
      dictation.selectedModel = catalog.models.some(
        (m) => m.modelPath === dictation.modelPath,
      )
        ? dictation.modelPath
        : "";
    } catch (error) {
      if (!state.disposed)
        dictation.modelsError = `Could not scan installed speech models: ${errorText(error)}`;
    } finally {
      dictation.scanning = false;
      if (!state.disposed) renderDictation();
    }
  }
  async function saveDictationModel(install = false, selectedPath = null) {
    if (state.disposed || dictation.saving || dictation.phase !== "idle")
      return;
    const path = (selectedPath ?? $("dictation-model-path").value).trim();
    const repository = $("dictation-repository").value.trim();
    const plan =
      dictation.plan?.repository === repository ? dictation.plan : null;
    if (install && !plan) dictation.plan = null;
    if (install && !repository) {
      dictation.error = "Enter a Hugging Face owner/repository.";
      renderDictation();
      return;
    }
    if (!install && !path) {
      dictation.error = "Enter a local Whisper model directory.";
      renderDictation();
      return;
    }
    dictation.saving = true;
    dictation.inspecting = install && !plan;
    dictation.installing = install && !!plan;
    dictation.error = "";
    dictation.note = "";
    renderDictation();
    try {
      if (install && !plan) {
        const inspected = await invoke("dictation_model_plan", { repository });
        if (state.disposed) return;
        if ($("dictation-repository").value.trim() !== repository)
          throw new Error(
            "Repository changed. Inspect it again before installing.",
          );
        if (
          inspected.repository !== repository ||
          !inspected.revision ||
          !inspected.directory ||
          !Number.isSafeInteger(inspected.total_bytes) ||
          inspected.total_bytes <= 0 ||
          !Array.isArray(inspected.files) ||
          !inspected.files.length ||
          inspected.files.some(
            (file) =>
              typeof file.name !== "string" ||
              !file.name ||
              !Number.isSafeInteger(file.bytes) ||
              file.bytes <= 0,
          ) ||
          inspected.files.reduce((total, file) => total + file.bytes, 0) !==
            inspected.total_bytes
        )
          throw new Error(
            "Invalid model installation plan. Inspect the repository again.",
          );
        dictation.plan = inspected;
        dictation.note =
          "Review the files, size and destination, then click Install.";
        return;
      }
      const settings = install
        ? await invoke("dictation_install_model", { plan })
        : await invoke("dictation_set_model", { modelPath: path });
      if (state.disposed) return;
      dictation.modelPath = settings.modelPath;
      dictation.selectedModel = settings.modelPath;
      dictation.plan = null;
      dictation.setup = false;
      if (install)
        dictation.note = "Model installed. Click the microphone to dictate.";
      $("dictation-toggle").focus();
    } catch (error) {
      if (!state.disposed) {
        dictation.error = errorText(error);
        if (install) dictation.plan = null;
      }
    } finally {
      dictation.saving = false;
      dictation.installing = false;
      dictation.inspecting = false;
      if (!state.disposed) renderDictation();
    }
  }
  async function toggleDictation(fromShortcut = false) {
    if (dictation.phase === "recording") return finishDictation();
    if (dictation.phase === "requesting") {
      cancelDictationCapture();
      render();
      return;
    }
    if (
      state.disposed ||
      locked() ||
      dictation.phase !== "idle" ||
      dictation.saving ||
      state.view !== "chat"
    )
      return;
    dictation.error = "";
    dictation.note = "";
    if (!dictation.modelPath) {
      if (!dictation.setup) openDictationSetup();
      return;
    }
    // A shortcut never initiates the first permission prompt. Require a visible
    // microphone button click before allowing shortcut-driven capture this session.
    if (fromShortcut && !dictation.activated) {
      dictation.note = "Click the microphone to enable dictation.";
      renderDictation();
      $("dictation-toggle").focus();
      return;
    }
    const c = store.composer;
    if (!c) return;
    const target = {
      id: c.id,
      revision: draftRevision,
      draft: c.draft,
      valid: true,
    };
    dictationTarget = target;
    dictation.setup = false;
    dictation.phase = "requesting";
    recorder = createPcmRecorder({
      window: win,
      navigator: win.navigator || globalThis.navigator,
      onLimit: () => void finishDictation(),
      onError: (error) => {
        cancelDictationCapture();
        dictation.error = errorText(error);
        if (!state.disposed) render();
      },
    });
    const capture = recorder;
    render();
    try {
      const started = await capture.start();
      if (
        !started ||
        dictationTarget !== target ||
        !target.valid ||
        state.disposed
      ) {
        capture.cancel();
        return;
      }
      dictation.activated = true;
      dictation.phase = "recording";
      dictation.elapsed = 0;
      tickDictation(Date.now());
    } catch (error) {
      if (dictationTarget !== target || state.disposed) return;
      cancelDictationCapture();
      dictation.error = errorText(error);
    }
    if (!state.disposed) render();
  }
  async function finishDictation() {
    if (dictation.phase !== "recording") return;
    const target = dictationTarget,
      capture = recorder;
    recorder = null;
    clearDictationTimer();
    dictation.phase = "transcribing";
    dictation.elapsed = 0;
    dictation.error = "";
    render();
    try {
      const pcm16 = capture.finish();
      if (!pcm16) throw new Error("No microphone audio captured. Try again.");
      tickDictation(Date.now());
      const transcript = await invoke("dictation_transcribe", { pcm16 });
      if (state.disposed || !target.valid || store.currentId !== target.id)
        return;
      const text = String(transcript.text || "").trim();
      if (!text)
        throw new Error(
          "No speech detected. Check the microphone and try again.",
        );
      const current = store.composer;
      // Never restore a captured draft over newer typing or consume attachments.
      const draft =
        target.revision === draftRevision && current.draft === target.draft
          ? target.draft
          : current.draft;
      store.updateDraft(
        target.id,
        draft + (draft && !/\s$/.test(draft) ? " " : "") + text,
      );
    } catch (error) {
      if (!state.disposed && target.valid && store.currentId === target.id)
        dictation.error = errorText(error);
    } finally {
      clearDictationTimer();
      dictationTarget = null;
      dictation.phase = "idle";
      dictation.elapsed = 0;
      if (!state.disposed) render();
    }
  }
  function resizeDraft() {
    $("draft").style.height = "auto";
    $("draft").style.height =
      `${Math.min(180, Math.max(72, $("draft").scrollHeight))}px`;
  }
  function render() {
    if (state.disposed) return;
    const c = store.composer;
    $("workspace").classList.toggle("inspector-open", state.inspector);
    $("chat-view").hidden = state.view !== "chat";
    $("models-view").hidden = state.view !== "models";
    $("settings-view").hidden = state.view !== "settings";
    $("inspector").hidden = !state.inspector || state.view !== "chat";
    $("conversation-options").hidden = state.view !== "chat";
    $("conversation-options").setAttribute(
      "aria-expanded",
      String(state.inspector),
    );
    $("conversation-title").textContent =
      state.view === "models"
        ? "Models"
        : state.view === "settings"
          ? "Settings"
          : c?.title || "New conversation";
    $("view-label").textContent = "";
    for (const view of ["chat", "models", "settings"]) {
      if (state.view === view)
        $(`nav-${view}`).setAttribute("aria-current", "page");
      else $(`nav-${view}`).removeAttribute("aria-current");
    }
    $("new-chat").disabled = state.booting;
    renderDictation();
    renderComposer();
    const error = errors.get(c?.id);
    $("request-error").hidden = !error;
    $("request-error-text").textContent = error?.message || "";
    $("retry-request").disabled = locked() || !state.status?.loaded;
    $("retry-request").hidden = !error || error.userIndex === undefined;
    $("delete-conversation").disabled =
      !c ||
      state.active?.conversationId === c.id ||
      state.preparingConversationId === c.id;
    if (settingsId !== c?.id) fillSettings();
    renderSidebar();
    const snapshot = renderSnapshot();
    renderMessages(snapshot);
    renderAttachments(snapshot);
    renderQuickSwitch();
    renderModels();
    setModelsTab(state.modelsTab);
    renderStore();
    renderAppSettings();
    renderUpdate();
    renderNotices();
  }
  function renderComposer() {
    const c = store.composer;
    if ($("draft").value !== (c?.draft || "")) {
      $("draft").value = c?.draft || "";
      resizeDraft();
    }
    $("draft").disabled = state.booting || !c;
    $("send").disabled =
      dictation.phase !== "idle" ||
      locked() ||
      !state.statusKnown ||
      !state.status?.loaded ||
      !c ||
      !!c.draftError ||
      (!c.draft.trim() && !c.attachmentCount);
    $("send").hidden = !!state.active;
    $("stop").hidden = !state.active;
    $("attach-image").disabled =
      state.booting || state.preparing || c?.settings.mode === "raw";
    $("composer-mode").textContent =
      c?.settings.mode === "raw"
        ? "Raw completion"
        : c?.settings.coding
          ? "Chat · Coding"
          : "Chat";
    $("composer-hint").textContent = "";
    $("model-state").textContent = "";
  }
  function applyStatus(status) {
    if (!status || typeof status.loaded !== "boolean")
      throw new Error("The host returned an invalid model status.");
    state.status = status;
    state.statusKnown = true;
    if (!status.loaded) state.loadedChoice = null;
    render();
    if (status.loading || status.updating) scheduleStatus();
    if (status.updating) scheduleUpdate();
  }
  async function refreshStatus() {
    try {
      applyStatus(await invoke("desktop_status"));
      return true;
    } catch (error) {
      state.statusKnown = false;
      showError(`Could not read model status. ${errorText(error)}`);
      render();
      return false;
    }
  }
  function scheduleStatus() {
    if (statusTimer !== null || state.disposed) return;
    statusTimer = setTimer(async () => {
      statusTimer = null;
      await refreshStatus();
    }, 800);
  }
  let discoveryQueued = false;
  async function discover({ restore = false } = {}) {
    if (state.disposed) return;
    if (state.scanning) {
      discoveryQueued = true;
      return;
    }
    const epoch = ++discoveryEpoch;
    const intent = modelIntent;
    state.scanning = true;
    render();
    try {
      const result = await invoke("discover_models");
      if (epoch !== discoveryEpoch || state.disposed) return;
      state.models = Array.isArray(result.models) ? result.models : [];
      state.warnings = Array.isArray(result.warnings)
        ? result.warnings.map(String)
        : [];
      state.remembered = result.lastModel || null;
      if (state.remembered) {
        const identity = modelIdentity(state.remembered);
        if (state.models.some((model) => modelIdentity(model) === identity))
          deletedModelIdentities.delete(identity);
        else if (deletedModelIdentities.has(identity)) state.remembered = null;
      }
      state.storeRevision++;
      if (restore && state.remembered && intent === modelIntent) {
        $("load-gpu").value = state.remembered.gpu;
        $("load-context").value = String(state.remembered.context);
      }
      if (!state.selected)
        state.selected =
          state.models.find(isLoaded) ||
          (state.remembered &&
            modelChoices().find(
              (m) => modelIdentity(m) === modelIdentity(state.remembered),
            )) ||
          state.models[0] ||
          null;
      if (restore && !state.restoreConsidered) {
        state.restoreConsidered = true;
        if (
          state.remembered &&
          state.appSettingsKnown &&
          state.appSettings.autoRestore &&
          intent === modelIntent &&
          !state.active &&
          !state.modelBusy &&
          !state.sourcePending &&
          state.statusKnown &&
          !state.status.loaded &&
          !state.status.loading &&
          !state.status.updating
        ) {
          // Discovery can be slow. Recheck the host immediately before restoring.
          const known = await refreshStatus();
          if (
            known &&
            intent === modelIntent &&
            !state.status.loaded &&
            !state.status.loading &&
            !state.status.updating &&
            !state.active &&
            !state.sourcePending
          ) {
            const saved = state.remembered;
            $("load-gpu").value = saved.gpu;
            $("load-context").value = String(saved.context);
            await loadModel(saved, true);
          }
        }
      }
    } catch (error) {
      showError(`Model scan failed. ${errorText(error)}`);
    } finally {
      state.scanning = false;
      render();
      if (discoveryQueued && !state.disposed) {
        discoveryQueued = false;
        await discover();
      }
    }
  }
  function selectedOptions(model = state.selected) {
    return validateModelOptions({
      ...model,
      model: modelReference(model),
      ollamaDir: model?.ollamaDir || "",
      projector: model?.projector || "",
      gpu:
        state.appSettings.loadPolicy === "manual"
          ? $("load-gpu").value
          : "auto",
      context:
        state.appSettings.loadPolicy === "manual"
          ? $("load-context").value
          : DEFAULT_APP_SETTINGS.context,
    });
  }
  async function loadModel(input, restoring = false) {
    if (
      locked() &&
      !(
        restoring &&
        state.booting &&
        !state.modelBusy &&
        !state.status?.loading &&
        !state.status?.updating
      )
    )
      return false;
    const options = validateModelOptions({
      ...input,
      model: modelReference(input),
      ollamaDir: input.ollamaDir || "",
      projector: input.projector || "",
    });
    modelIntent++;
    state.modelBusy = true;
    state.modelOperation = "load";
    state.error = "";
    render();
    try {
      const status = await invoke("load_model", { options });
      state.loadedChoice = options;
      state.remembered = options;
      state.selected =
        modelChoices().find(
          (m) => modelIdentity(m) === modelIdentity(options),
        ) || options;
      applyStatus(status);
      return true;
    } catch (error) {
      showError(
        `${restoring ? "Could not restore the last model" : "Could not load model"}. ${errorText(error)}`,
      );
      await refreshStatus();
      return false;
    } finally {
      state.modelBusy = false;
      messageSignature = "";
      render();
    }
  }
  async function deleteModel() {
    const model = selectedInstalledModel();
    if (!model || deletionLocked()) return;
    const name = model.name || model.model;
    if (
      !win.confirm(
        `Permanently delete "${name}" from ${sourceLabel(model.source)}? This cannot be undone. Shared aliases may only have their entry removed; shared or external weights may be kept. No model will be automatically unloaded or loaded.`,
      )
    )
      return;
    if (deletionLocked()) return;
    modelIntent++;
    discoveryEpoch++;
    state.modelBusy = true;
    state.modelOperation = "delete";
    state.error = "";
    render();
    try {
      const result = await invoke("model_delete", { id: model.id });
      if (state.disposed) return;
      const identity = modelIdentity(model);
      deletedModelIdentities.add(identity);
      state.models = state.models.filter((item) => item.id !== model.id);
      if (modelIdentity(state.selected) === identity) state.selected = null;
      if (modelIdentity(state.remembered) === identity) state.remembered = null;
      if (modelIdentity(state.loadedChoice) === identity)
        state.loadedChoice = null;
      const retained = result?.retainedFiles ?? result?.retained_files;
      notifications.notify(
        Array.isArray(retained) && retained.length
          ? `${name} entry removed. ${retained.length} shared or external file(s) kept. To remove external weights, select their original source entry; shared weights may still be needed by other models.`
          : `${name} deleted.`,
        { kind: "success" },
      );
      if (Array.isArray(result?.warnings)) notifyWarnings(result.warnings);
      await discover();
      if (state.disposed) return;
      state.selected =
        state.models.find((item) => item.id === state.selected?.id) ||
        state.models[0] ||
        null;
    } catch (error) {
      if (!state.disposed) {
        showError(`Could not delete ${name}. ${errorText(error)}`);
        await refreshStatus();
      }
    } finally {
      state.modelBusy = false;
      state.modelOperation = null;
      if (!state.disposed) render();
    }
  }
  async function unload() {
    if (locked()) return;
    modelIntent++;
    state.modelBusy = true;
    state.modelOperation = "unload";
    render();
    try {
      applyStatus(await invoke("unload_model"));
    } catch (error) {
      showError(error);
      await refreshStatus();
    } finally {
      state.modelBusy = false;
      render();
    }
  }
  async function requestFor(c, userIndex) {
    if (locked() || !state.statusKnown || !state.status?.loaded) return;
    const user = c.messages[userIndex];
    const settings = user.settings || c.settings;
    const request = buildRequest(
      c.messages.slice(0, userIndex),
      user.content,
      user.images || [],
      settings,
      state.status.model || "",
    );
    const token = {
      id: uniqueId(),
      conversationId: c.id,
      userIndex,
      canceled: false,
      started: Date.now(),
    };
    state.active = token;
    errors.delete(c.id);
    messageSignature = "";
    render();
    try {
      const response = await invoke("api_request", {
        id: token.id,
        method: "POST",
        path: request.endpoint,
        body: JSON.parse(request.json),
      });
      if (token.canceled || state.disposed) return;
      let data = response.body;
      if (typeof data === "string") {
        try {
          data = JSON.parse(data);
        } catch {
          throw new Error("The host returned an unreadable response.");
        }
      }
      if (response.status < 200 || response.status >= 300) {
        const failure = apiFailure(
          response.status,
          data?.error?.message || data?.error || "",
        );
        throw new Error(`${failure.title}\n${failure.detail}`);
      }
      const choice = data?.choices?.[0];
      const content =
        settings.mode === "raw" ? choice?.text : choice?.message?.content;
      if (typeof content !== "string")
        throw new Error(
          "The model returned no text response. Retry or choose another model.",
        );
      store.update(c.id, (conversation) => {
        conversation.messages.push({
          role: "assistant",
          content,
          mode: settings.mode,
          elapsed: Date.now() - token.started,
          ...(data.usage ? { usage: data.usage } : {}),
          ...(typeof choice.finish_reason === "string"
            ? { finish: choice.finish_reason }
            : {}),
          model:
            typeof data.model === "string"
              ? data.model
              : state.status.model || "",
        });
      });
    } catch (error) {
      if (!token.canceled && !state.disposed)
        errors.set(c.id, { userIndex, message: errorText(error) });
    } finally {
      if (state.active === token) state.active = null;
      messageSignature = "";
      if (!state.disposed) render();
    }
  }
  async function send() {
    if (
      dictation.phase !== "idle" ||
      locked() ||
      !state.statusKnown ||
      !state.status?.loaded
    )
      return;
    if (store.composer?.draftError) throw new Error(store.composer.draftError);
    const c = store.current;
    if (!c) return;
    // Validate before consuming a draft. Atomic store validation also preserves it
    // if message or workspace storage limits would be exceeded.
    buildRequest(
      c.messages,
      c.draft,
      c.attachments,
      c.settings,
      state.status.model || "",
    );
    const index = c.messages.length;
    store.update(c.id, (conversation) => {
      conversation.messages.push({
        role: "user",
        content: c.draft,
        images: c.attachments,
        mode: c.settings.mode,
        settings: c.settings,
      });
      if (
        !conversation.messages.slice(0, index).length &&
        conversation.title === "New conversation"
      )
        conversation.title = Array.from(
          c.draft.trim().split("\n")[0] || "Image conversation",
        )
          .slice(0, 55)
          .join("");
      conversation.draft = "";
      conversation.attachments = [];
    });
    await requestFor(store.current, index);
  }
  function stop() {
    const token = state.active;
    if (!token) return;
    token.canceled = true;
    state.active = null;
    errors.set(token.conversationId, {
      userIndex: token.userIndex,
      message: "Generation stopped.",
    });
    void invoke("cancel_request", { id: token.id }).catch((error) =>
      showError(`Could not cancel the native request. ${errorText(error)}`),
    );
    messageSignature = "";
    render();
  }
  async function retry() {
    const c = store.current,
      error = errors.get(c?.id);
    if (error?.userIndex !== undefined) await requestFor(c, error.userIndex);
  }
  async function branchAt(index, regenerate = false) {
    if (state.booting || (regenerate && locked())) return;
    const c = store.current;
    if (!c) return;
    let userIndex = index;
    while (userIndex >= 0 && c.messages[userIndex].role !== "user") userIndex--;
    if (userIndex < 0) return;
    void store.flush();
    store.branch(c.id, userIndex);
    fillSettings();
    render();
    if (regenerate) await send();
    else $("draft").focus();
  }
  async function attach(files) {
    const c = store.current;
    if (!c || state.preparing || c.settings.mode === "raw") return;
    state.preparing = true;
    state.preparingConversationId = c.id;
    render();
    try {
      const picked = Array.from(files);
      if (
        picked.some(
          (f) =>
            !["image/png", "image/jpeg"].includes(f.type) ||
            f.size > LIMITS.imageBytes,
        )
      )
        throw new Error("Choose PNG or JPEG files up to 8 MiB each.");
      if (picked.length + c.attachments.length > LIMITS.images)
        throw new Error(
          "Choose at most 8 images including existing attachments.",
        );
      const prepared = [];
      for (const file of picked) {
        const data = await new Promise((resolve, reject) => {
          const reader = new win.FileReader();
          reader.onload = () => resolve(reader.result);
          reader.onerror = () =>
            reject(new Error(`Could not read ${file.name}.`));
          reader.readAsDataURL(file);
        });
        prepared.push(normalizeImage(data, file.type, file.name));
      }
      const conversation = store.workspace.conversations.find(
        (v) => v.id === c.id,
      );
      if (!conversation)
        throw new Error("The destination conversation no longer exists.");
      checkImageBudget([
        ...conversation.messages.flatMap((m) => m.images || []),
        ...conversation.attachments,
        ...prepared,
      ]);
      store.update(c.id, (v) => v.attachments.push(...prepared));
    } catch (error) {
      showError(error);
    } finally {
      state.preparing = false;
      state.preparingConversationId = null;
      $("image-input").value = "";
      render();
    }
  }
  async function importModel() {
    if (locked()) return;
    const options = selectedOptions({
      model: $("manual-model").value,
      ollama: $("manual-source").value === "ollama",
      ollamaDir: $("manual-ollama-dir").value,
      projector: $("manual-projector").value,
    });
    const intent = ++modelIntent;
    state.importing = true;
    render();
    try {
      const result = await invoke("model_import", {
        options,
        name: $("manual-name").value.trim() || null,
      });
      if (state.disposed) return;
      await discover();
      if (intent === modelIntent)
        state.selected =
          modelChoices().find(
            (model) =>
              model.name === result.model || model.model === result.path,
          ) || state.selected;
      $("manual-import").hidden = true;
      $("toggle-manual-model").setAttribute("aria-expanded", "false");
      notifications.notify(`${result.model} imported`, { kind: "success" });
      render();
    } catch (error) {
      showError(error);
    } finally {
      state.importing = false;
      if (!state.disposed) render();
    }
  }
  function exportConversation(format) {
    const c = store.current;
    if (!c) return;
    const name =
      c.title.replace(/[^a-z0-9_-]/gi, "-").slice(0, 70) || "conversation";
    if (format === "json")
      download(JSON.stringify(c, null, 2), `${name}.json`, "application/json");
    else
      download(markdownExport(c), `${name}.md`, "text/markdown;charset=utf-8");
  }
  function scheduleUpdate() {
    if (updateTimer !== null || state.disposed) return;
    updateTimer = setTimer(async () => {
      updateTimer = null;
      try {
        state.update = await invoke("update_status");
        render();
        if (state.update.state === "Running" || state.sourcePending)
          scheduleUpdate();
        else await refreshStatus();
      } catch (error) {
        showError(`Could not read update progress. ${errorText(error)}`);
        if (state.sourcePending || state.status?.updating) scheduleUpdate();
      }
    }, 700);
  }
  async function sourceUpdate() {
    if (locked()) return;
    if (
      !win.confirm(
        "Update lma from source? This unloads the model and builds the application. Keep this window open until the update finishes.",
      )
    )
      return;
    modelIntent++;
    state.sourcePending = true;
    state.update = {
      state: "Running",
      log: "Starting source update…",
      error: null,
    };
    render();
    scheduleUpdate();
    try {
      const result = await invoke("source_update");
      state.sourcePending = false;
      state.update = await invoke("update_status");
      if (!state.update.log) state.update.log = result.message || "";
    } catch (error) {
      state.sourcePending = false;
      state.update = {
        state: "Failed",
        log: state.update.log,
        error: errorText(error),
      };
    } finally {
      state.sourcePending = false;
      await refreshStatus();
      if (state.update.state === "Running") scheduleUpdate();
      render();
    }
  }
  async function start() {
    if (startPromise) return startPromise;
    startPromise = (async () => {
      render();
      // Status is always the first IPC command. Never blindly restore a model.
      await refreshStatus();
      await loadAppSettings();
      if (state.disposed) return;
      const discovering = discover({ restore: true });
      const dictationSettings = invoke("dictation_settings")
        .then((settings) => {
          if (!state.disposed) dictation.modelPath = settings.modelPath;
        })
        .catch((error) => {
          dictation.error = errorText(error);
        });
      await store.hydrate();
      await dictationSettings;
      if (state.disposed) return;
      if (!store.current && store.workspace.conversations.length)
        store.select(store.workspace.conversations[0].id);
      if (!store.current) store.add();
      state.booting = false;
      fillSettings();
      render();
      await discovering;
      scheduleDiscovery();
    })();
    return startPromise;
  }
  function dispose() {
    void store.flush();
    state.disposed = true;
    cancelDictationCapture();
    clearDictationTimer();
    if (statusTimer !== null) clearTimer(statusTimer);
    if (updateTimer !== null) clearTimer(updateTimer);
    if (discoveryTimer !== null) clearTimer(discoveryTimer);
    notifications.dispose();
    store.dispose();
    for (const remove of listeners) remove();
  }

  if (!$("store-source").value) $("store-source").value = "all";
  for (const [id, icon, label] of [
    ["new-chat", "plus", "New conversation"],
    ["conversation-options", "settings", "Conversation options"],
    ["close-inspector", "close", "Close conversation options"],
    ["attach-image", "attach", "Attach PNG or JPEG"],
    ["dictation-toggle", "mic", "Start dictation"],
    ["dictation-configure", "settings", "Dictation model"],
    ["send", "send", "Send message"],
    ["stop", "stop", "Stop generation"],
    ["dismiss-error", "close", "Dismiss error"],
  ])
    decorateIcon($(id), icon, label);
  for (const [id, icon] of [
    ["chat-icon", "chat"],
    ["models-icon", "models"],
    ["history-search-icon", "search"],
    ["model-search-icon", "search"],
  ]) {
    const holder = element("span");
    decorateIcon(holder, icon, "");
    $(id).append(holder.firstChild);
  }
  on($("new-chat"), "click", guard(newChat));
  on($("nav-chat"), "click", () => setView("chat"));
  on($("nav-models"), "click", () => setView("models"));
  on($("nav-settings"), "click", () => setView("settings"));
  on($("open-settings"), "click", () => setView("settings"));
  on($("history-search-toggle"), "click", () => {
    const open = $("history-search-field").hidden;
    $("history-search-field").hidden = !open;
    $("history-search-toggle").setAttribute("aria-expanded", String(open));
    if (open) $("conversation-search").focus();
    else {
      $("conversation-search").value = "";
      renderSidebar();
    }
  });
  on($("conversation-search"), "keydown", (event) => {
    if (event.key !== "Escape") return;
    event.preventDefault();
    $("conversation-search").value = "";
    $("history-search-field").hidden = true;
    $("history-search-toggle").setAttribute("aria-expanded", "false");
    renderSidebar();
    $("history-search-toggle").focus();
  });
  for (const tab of ["installed", "store"]) {
    on($(`models-${tab}-tab`), "click", () => setModelsTab(tab));
    on($(`models-${tab}-tab`), "keydown", (event) => {
      if (!["ArrowLeft", "ArrowRight", "Home", "End"].includes(event.key))
        return;
      event.preventDefault();
      const next =
        event.key === "Home"
          ? "installed"
          : event.key === "End"
            ? "store"
            : tab === "installed"
              ? "store"
              : "installed";
      setModelsTab(next);
      $(`models-${next}-tab`).focus();
    });
  }
  on(
    $("store-search-form"),
    "submit",
    guard((event) => {
      event.preventDefault();
      return searchStore();
    }),
  );
  on(
    $("store-source"),
    "change",
    guard(() => {
      if ($("store-query").value.trim()) return searchStore();
    }),
  );
  on($("app-settings-form"), "input", () => {
    state.appSettingsDirty = true;
    $("app-settings-status").textContent = "";
    renderAppSettings();
  });
  on($("app-settings-form"), "change", () => {
    state.appSettingsDirty = true;
    renderAppSettings();
  });
  on(
    $("app-settings-form"),
    "submit",
    guard((event) => {
      event.preventDefault();
      return saveAppSettings();
    }),
  );
  on($("open-library"), "click", () => setView("models"));
  on($("conversation-options"), "click", () => openInspector());
  on($("close-inspector"), "click", () => openInspector(false));
  on($("conversation-search"), "input", renderSidebar);
  on($("model-search"), "input", renderModels);
  on(
    $("draft"),
    "input",
    guard(() => {
      const id = store.currentId;
      if (id) store.updateDraft(id, $("draft").value);
      resizeDraft();
    }),
  );
  on(
    $("draft"),
    "keydown",
    guard((event) => {
      if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
        event.preventDefault();
        return send();
      }
    }),
  );
  on(
    $("composer-form"),
    "submit",
    guard((event) => {
      event.preventDefault();
      return send();
    }),
  );
  on($("dictation-toggle"), "click", () => void toggleDictation());
  on($("dictation-configure"), "click", openDictationSetup);
  on($("dictation-save-model"), "click", () => void saveDictationModel());
  on(
    $("dictation-install-model"),
    "click",
    () => void saveDictationModel(true),
  );
  on($("dictation-repository"), "input", () => {
    if (dictation.saving) return;
    dictation.plan = null;
    dictation.note = "";
    dictation.error = "";
    renderDictation();
  });
  on($("dictation-installed-models"), "change", () => {
    dictation.selectedModel = $("dictation-installed-models").value;
    renderDictation();
  });
  on(
    $("dictation-use-installed"),
    "click",
    () => void saveDictationModel(false, dictation.selectedModel),
  );
  on(
    $("dictation-rescan-models"),
    "click",
    () => void discoverDictationModels(),
  );
  on($("dictation-close-setup"), "click", () => {
    dictation.setup = false;
    dictation.plan = null;
    renderDictation();
    $("dictation-toggle").focus();
  });
  on($("dictation-model-path"), "keydown", (event) => {
    if (event.key === "Enter" && !event.isComposing) {
      event.preventDefault();
      void saveDictationModel();
    }
  });
  on($("stop"), "click", stop);
  on($("retry-request"), "click", guard(retry));
  on($("attach-image"), "click", () => $("image-input").click());
  on($("image-input"), "change", () => void attach($("image-input").files));
  on($("draft"), "paste", (event) => {
    const files = Array.from(event.clipboardData?.files || []);
    if (files.length) {
      event.preventDefault();
      void attach(files);
    }
  });
  on(
    $("model-quick-switch"),
    "change",
    guard(async () => {
      if (locked()) return;
      const selected = modelChoices().find(
        (m) => modelIdentity(m) === $("model-quick-switch").value,
      );
      if (selected) {
        state.selected = selected;
        await loadModel(selectedOptions(selected));
      } else setView("models");
    }),
  );
  on($("rescan-models"), "click", () => {
    if (!busy()) void discover();
  });
  on(
    $("load-model"),
    "click",
    guard(() =>
      (state.selected ? isLoaded(state.selected) : state.status?.loaded)
        ? unload()
        : loadModel(selectedOptions()),
    ),
  );
  on($("unload-model"), "click", guard(unload));
  on($("delete-model"), "click", guard(deleteModel));
  function toggleManualImport() {
    const open = $("manual-import").hidden;
    $("manual-import").hidden = !open;
    for (const id of ["manual-toggle", "toggle-manual-model"])
      $(id).setAttribute("aria-expanded", String(open));
    if (open) $("manual-model").focus();
  }
  on($("manual-toggle"), "click", toggleManualImport);
  on($("toggle-manual-model"), "click", toggleManualImport);
  on(
    $("manual-import"),
    "submit",
    guard((event) => {
      event.preventDefault();
      return importModel();
    }),
  );
  on(
    $("rename-conversation"),
    "click",
    guard(() => {
      const c = store.current;
      if (c) store.rename(c.id, $("rename-title").value);
    }),
  );
  on(
    $("rename-title"),
    "keydown",
    guard((event) => {
      if (event.key === "Enter") {
        event.preventDefault();
        const c = store.current;
        if (c) store.rename(c.id, $("rename-title").value);
      }
    }),
  );
  on(
    $("delete-conversation"),
    "click",
    guard(() => {
      if (store.current) removeConversation(store.current.id);
    }),
  );
  on($("settings-form"), "input", saveSettings);
  on($("settings-form"), "change", saveSettings);
  on($("settings-form"), "submit", (event) => {
    event.preventDefault();
    saveSettings();
  });
  on($("export-json"), "click", () => exportConversation("json"));
  on($("export-markdown"), "click", () => exportConversation("markdown"));
  on($("source-update"), "click", guard(sourceUpdate));
  on(
    $("persistence-retry"),
    "click",
    guard(async () => {
      if (!store.persistence.hydrated) await store.hydrate();
      else await store.flush();
    }),
  );
  on($("dismiss-error"), "click", () => {
    state.error = "";
    renderNotices();
  });
  on(win, "beforeunload", () => {
    cancelDictationCapture();
    void store.flush();
  });
  on(win, "pagehide", () => {
    cancelDictationCapture();
  });
  on(doc, "visibilitychange", () => {
    if (doc.hidden) {
      cancelDictationCapture();
      render();
    }
  });
  on(
    doc,
    "keydown",
    guard((event) => {
      if (event.key === "Escape") {
        if (["recording", "requesting"].includes(dictation.phase)) {
          cancelDictationCapture();
          render();
          return;
        }
        if (dictation.setup && !dictation.saving) {
          dictation.setup = false;
          dictation.plan = null;
          renderDictation();
          $("dictation-toggle").focus();
          return;
        }
        if (state.inspector) openInspector(false);
      }
      if (
        !event.defaultPrevented &&
        !event.isComposing &&
        !event.repeat &&
        !event.altKey &&
        (event.metaKey || event.ctrlKey) &&
        event.shiftKey &&
        event.key.toLowerCase() === "d" &&
        state.view === "chat" &&
        !state.inspector &&
        !doc.querySelector('dialog[open], [aria-modal="true"]')
      ) {
        event.preventDefault();
        return toggleDictation(true);
      }
      if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "n") {
        event.preventDefault();
        newChat();
      }
    }),
  );
  return {
    start,
    dispose,
    store,
    state,
    send,
    stop,
    retry,
    attach,
    branchAt,
    discover,
    loadModel,
    unload,
    sourceUpdate,
    selectConversation,
    newChat,
    setView,
    openInspector,
    render,
    refreshStatus,
    exportConversation,
    saveAppSettings,
    setModelsTab,
    searchStore,
    downloadStoreModel,
  };
}

if (typeof document !== "undefined" && !globalThis.__LMA_NO_AUTOSTART__) {
  const app = createDesktopApp();
  // Native launch harness: await .ready, then inspect .store/.state or use stable IDs.
  globalThis.__LMA_DESKTOP__ = app;
  app.ready = app.start().catch((error) => {
    app.state.error = `Could not open the workspace. ${errorText(error)}`;
    app.state.booting = false;
    app.render();
  });
}
