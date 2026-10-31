import { DEFAULT_SETTINGS, validateSettings, normalizeImage } from "./core.js";

export const CAPS = Object.freeze({
  conversations: 100,
  messages: 1000,
  total: 10000,
  text: 1048576,
  bytes: 64 * 1048576,
});
const encoder = new TextEncoder();
const clone = (value) => JSON.parse(JSON.stringify(value));
const fail = (message) => {
  throw new Error(message);
};
function text(value, limit = CAPS.text, label = "Text") {
  if (typeof value !== "string" || encoder.encode(value).length > limit)
    fail(`${label} exceeds its limit or is invalid.`);
  return value;
}
function settings(value) {
  if (!value || typeof value !== "object")
    fail("Invalid conversation settings.");
  const s = validateSettings(value);
  return {
    mode: s.mode,
    system: text(s.system),
    coding: s.coding,
    max_tokens: s.max_tokens,
    temperature: s.temperature,
    top_p: s.top_p,
    seed: s.seed,
  };
}
function images(value) {
  if (!Array.isArray(value) || value.length > 8)
    fail("Invalid image list (maximum 8 per message).");
  return value.map((image) => {
    if (!image || typeof image.name !== "string") fail("Invalid image.");
    const result = normalizeImage(image.base64, image.mime, image.name);
    if (result.size !== image.size)
      fail("Image size does not match its bytes.");
    return result;
  });
}
function message(value) {
  if (!value || !["user", "assistant"].includes(value.role))
    fail("Invalid message role.");
  const result = { role: value.role, content: text(value.content) };
  if (value.images !== undefined) {
    if (value.role !== "user") fail("Only user messages can have images.");
    result.images = images(value.images);
  }
  if (value.mode !== undefined) {
    if (!["chat", "raw"].includes(value.mode)) fail("Invalid message mode.");
    result.mode = value.mode;
  }
  if (value.settings !== undefined) result.settings = settings(value.settings);
  if (value.elapsed !== undefined) {
    if (!Number.isFinite(value.elapsed) || value.elapsed < 0)
      fail("Invalid elapsed time.");
    result.elapsed = value.elapsed;
  }
  if (value.usage !== undefined) {
    result.usage = {};
    for (const key of ["prompt_tokens", "completion_tokens", "total_tokens"]) {
      if (value.usage[key] === undefined) continue;
      if (!Number.isSafeInteger(value.usage[key]) || value.usage[key] < 0)
        fail("Invalid token usage.");
      result.usage[key] = value.usage[key];
    }
  }
  for (const key of ["finish", "model"])
    if (value[key] !== undefined) result[key] = text(value[key]);
  return result;
}

// Explicit allowlists at every level keep transient state and future fields off disk.
// Validation is atomic: oversized or damaged data is never silently truncated.
export function normalizeWorkspace(value) {
  if (!value || value.version !== 1 || !Array.isArray(value.conversations))
    fail("Unsupported or invalid workspace.");
  if (value.conversations.length > CAPS.conversations)
    fail("Workspace has more than 100 conversations.");
  const ids = new Set();
  let total = 0;
  const conversations = value.conversations.map((c) => {
    if (!c || !c.id || ids.has(c.id))
      fail("Missing or duplicate conversation ID.");
    const id = text(c.id, 128, "Conversation ID");
    ids.add(id);
    if (!Array.isArray(c.messages) || c.messages.length > CAPS.messages)
      fail("Conversation has more than 1000 messages.");
    total += c.messages.length;
    if (total > CAPS.total) fail("Workspace has more than 10000 messages.");
    return {
      id,
      title: text(c.title, 256, "Title"),
      createdAt: text(c.createdAt, 64, "Timestamp"),
      updatedAt: text(c.updatedAt, 64, "Timestamp"),
      settings: settings(c.settings),
      draft: text(c.draft),
      attachments: images(c.attachments),
      messages: c.messages.map(message),
    };
  });
  if (value.currentId !== null && !ids.has(value.currentId))
    fail("Current conversation does not exist.");
  const result = { version: 1, currentId: value.currentId, conversations };
  if (encoder.encode(JSON.stringify(result)).length > CAPS.bytes)
    fail(
      "Workspace exceeds the 64 MiB storage limit. Export and remove a conversation first.",
    );
  return result;
}

export const emptyWorkspace = () => ({
  version: 1,
  currentId: null,
  conversations: [],
});
export function uniqueId() {
  if (globalThis.crypto?.randomUUID) return globalThis.crypto.randomUUID();
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2)}-${Math.random().toString(36).slice(2)}`;
}
export function newConversation(initial = {}) {
  const now = new Date().toISOString();
  return {
    id: uniqueId(),
    title: "New conversation",
    createdAt: now,
    updatedAt: now,
    settings: { ...DEFAULT_SETTINGS },
    draft: "",
    attachments: [],
    messages: [],
    ...initial,
  };
}
function branchTitle(title) {
  let prefix = "";
  for (const character of title) {
    if (encoder.encode(prefix + character).length > 240) break;
    prefix += character;
  }
  return `${prefix} · branch`;
}
export function branchConversation(conversation, userIndex) {
  const user = conversation.messages[userIndex];
  if (user?.role !== "user") fail("Choose a user message to branch from.");
  return newConversation({
    title: branchTitle(conversation.title),
    settings: clone(user.settings || conversation.settings),
    messages: clone(conversation.messages.slice(0, userIndex)),
    draft: user.content,
    attachments: clone(user.images || []),
  });
}
export function markdownExport(c) {
  const lines = [
    `# ${c.title}`,
    "",
    ...c.messages.flatMap((m) => [
      `## ${m.role === "user" ? "You" : "Assistant"}${m.mode === "raw" ? " · raw" : ""}`,
      "",
      m.content,
      ...(m.images || []).map(
        (i) =>
          `![${i.name.replace(/[\[\]\r\n]/g, "")}](data:${i.mime};base64,${i.base64})`,
      ),
      "",
    ]),
  ];
  if (c.draft || c.attachments.length)
    lines.push(
      "## Draft",
      "",
      c.draft,
      ...c.attachments.map(
        (i) => `![Image](data:${i.mime};base64,${i.base64})`,
      ),
    );
  return lines.join("\n");
}

export function createWorkspaceStore({
  invoke,
  onChange = () => {},
  delay = 400,
  setTimer = setTimeout,
  clearTimer = clearTimeout,
}) {
  let workspace = emptyWorkspace();
  let hydrated = false,
    loading = false,
    revision = 0,
    contentRevision = 0,
    saved = 0,
    timer = null,
    flight = null;
  let readError = "",
    saveError = "";
  // Invalid drafts stay local, outside the valid schema, until corrected. Keeping
  // them separate lets unrelated chats remain editable without discarding text.
  const oversizedDrafts = new Map();
  const current = () =>
    workspace.conversations.find((c) => c.id === workspace.currentId) || null;
  function draftError(id = oversizedDrafts.keys().next().value) {
    if (!oversizedDrafts.has(id)) return "";
    const title =
      workspace.conversations.find((c) => c.id === id)?.title ||
      "Untitled conversation";
    return `Draft in “${title}” exceeds the 1 MiB text limit. Shorten it or export it before saving or sending. The full draft is kept in this window.`;
  }
  function snapshot(conversation) {
    if (!conversation) return null;
    const result = clone(conversation);
    if (oversizedDrafts.has(conversation.id))
      result.draft = oversizedDrafts.get(conversation.id);
    return result;
  }
  const notify = (reason) => onChange(reason);
  const cancelTimer = () => {
    if (timer !== null) clearTimer(timer);
    timer = null;
  };
  function schedule() {
    cancelTimer();
    if (hydrated)
      timer = setTimer(() => {
        timer = null;
        void flush();
      }, delay);
  }
  function mutate(fn) {
    const candidate = clone(workspace);
    fn(candidate);
    const next = normalizeWorkspace(candidate);
    for (const id of oversizedDrafts.keys()) {
      const c = next.conversations.find((v) => v.id === id);
      const previous = workspace.conversations.find((v) => v.id === id);
      if (!c || c.draft !== previous?.draft) oversizedDrafts.delete(id);
    }
    workspace = next;
    revision++;
    contentRevision++;
    schedule();
    notify("data");
  }
  function updateDraft(id, draft) {
    if (typeof draft !== "string") fail("Draft must be text.");
    const c = workspace.conversations.find((v) => v.id === id);
    if (!c) fail("Conversation not found.");
    if (draft === (oversizedDrafts.get(id) ?? c.draft)) return;
    if (encoder.encode(draft).length > CAPS.text) {
      oversizedDrafts.set(id, draft);
    } else {
      oversizedDrafts.delete(id);
      const updated = { ...c, draft, updatedAt: new Date().toISOString() };
      workspace = {
        ...workspace,
        conversations: workspace.conversations.map((v) =>
          v.id === id ? updated : v,
        ),
      };
    }
    // No history clone, serialization or image normalization on the typing path.
    // The save boundary below still validates the complete, atomic snapshot.
    revision++;
    schedule();
    notify("draft");
  }
  async function hydrate() {
    if (loading || hydrated) return false;
    loading = true;
    notify("persistence");
    try {
      const loaded = normalizeWorkspace(await invoke("workspace_load"));
      // A failed first read may leave local working data. Merge by UUID on retry;
      // never replace it with the disk snapshot or save before a successful read.
      const local = workspace.conversations;
      const ids = new Set(local.map((c) => c.id));
      workspace = normalizeWorkspace({
        ...loaded,
        currentId: workspace.currentId || loaded.currentId,
        conversations: [
          ...local,
          ...loaded.conversations.filter((c) => !ids.has(c.id)),
        ],
      });
      hydrated = true;
      contentRevision++;
      readError = "";
      if (local.length) {
        revision++;
        schedule();
      }
      notify("hydrate");
      return true;
    } catch (error) {
      readError = `Could not read saved conversations. Local changes will not overwrite the saved workspace. ${String(error.message || error)}`;
      return false;
    } finally {
      loading = false;
      notify("persistence");
    }
  }
  function flush() {
    cancelTimer();
    if (!hydrated || saved === revision) return Promise.resolve(!readError);
    if (flight) return flight;
    // One writer, with a fresh snapshot after each acknowledgement. A stale save
    // can never overtake a newer edit, including edits made while IPC is pending.
    flight = (async () => {
      while (saved !== revision) {
        const target = revision;
        try {
          const invalidDraft = draftError();
          if (invalidDraft) fail(invalidDraft);
          const snapshot = normalizeWorkspace(workspace);
          await invoke("workspace_save", { workspace: snapshot });
          saved = target;
          saveError = "";
        } catch (error) {
          saveError = `Changes are kept in this window but could not be saved. ${String(error.message || error)}`;
          notify("persistence");
          return false;
        }
      }
      return true;
    })().finally(() => {
      flight = null;
      notify("persistence");
    });
    notify("persistence");
    return flight;
  }
  function add(conversation = newConversation()) {
    mutate((w) => {
      w.conversations.unshift(conversation);
      w.currentId = conversation.id;
    });
    return conversation.id;
  }
  return {
    get workspace() {
      return {
        version: 1,
        currentId: workspace.currentId,
        conversations: workspace.conversations.map(snapshot),
      };
    },
    get current() {
      return snapshot(current());
    },
    get currentId() {
      return workspace.currentId;
    },
    get contentRevision() {
      return contentRevision;
    },
    get composer() {
      const c = current();
      return c
        ? {
            id: c.id,
            title: c.title,
            draft: oversizedDrafts.get(c.id) ?? c.draft,
            settings: { ...c.settings },
            attachmentCount: c.attachments.length,
            draftError: draftError(c.id),
          }
        : null;
    },
    listConversations(query = "") {
      const search = query.toLowerCase().trim();
      return workspace.conversations
        .filter(
          (c) =>
            !search ||
            c.title.toLowerCase().includes(search) ||
            c.messages.some((m) => m.content.toLowerCase().includes(search)),
        )
        .map((c) => ({ id: c.id, title: c.title }));
    },
    get persistence() {
      return {
        hydrated,
        loading,
        dirty: revision !== saved,
        saving: !!flight,
        readError,
        saveError,
        draftError: draftError(),
      };
    },
    hydrate,
    flush,
    mutate,
    updateDraft,
    add,
    select(id) {
      mutate((w) => {
        if (!w.conversations.some((c) => c.id === id))
          fail("Conversation not found.");
        w.currentId = id;
      });
      void flush();
    },
    update(id, fn) {
      mutate((w) => {
        const c = w.conversations.find((c) => c.id === id);
        if (!c) fail("Conversation not found.");
        fn(c);
        c.updatedAt = new Date().toISOString();
      });
    },
    rename(id, title) {
      this.update(id, (c) => {
        c.title = title.trim() || "Untitled conversation";
      });
    },
    remove(id) {
      mutate((w) => {
        w.conversations = w.conversations.filter((c) => c.id !== id);
        if (w.currentId === id) w.currentId = w.conversations[0]?.id || null;
      });
      void flush();
    },
    branch(id, index) {
      const c = workspace.conversations.find((c) => c.id === id);
      return add(branchConversation(c, index));
    },
    dispose() {
      cancelTimer();
    },
  };
}
