import test from "node:test";
import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import { setImmediate as nextTurn } from "node:timers/promises";
import * as core from "../desktop/ui/core.js";
import {
  CAPS,
  normalizeWorkspace,
  emptyWorkspace,
  newConversation,
  branchConversation,
  createWorkspaceStore,
  markdownExport,
} from "../desktop/ui/workspace.js";
import {
  createDesktopApp,
  modelIdentity,
  modelReference,
} from "../desktop/ui/app.js";
import {
  DEFAULT_APP_SETTINGS,
  validateAppSettings,
} from "../desktop/ui/preferences.js";

const shell = await readFile(
  new URL("../desktop/ui/index.html", import.meta.url),
  "utf8",
);
const clone = (v) => JSON.parse(JSON.stringify(v));
const drain = async () => {
  await nextTurn();
  await nextTurn();
};
const deferred = () => {
  let resolve, reject;
  const promise = new Promise((a, b) => {
    resolve = a;
    reject = b;
  });
  return { promise, resolve, reject };
};
const workspaceOf = (...conversations) => ({
  version: 1,
  currentId: conversations[0]?.id || null,
  conversations,
});
const png = core.normalizeImage("iVBORw0KGgo=", "image/png", "test.png");
const jpeg = core.normalizeImage("/9j/", "image/jpeg", "test.jpg");
const status = (extra = {}) => ({
  loaded: true,
  model: "tiny.gguf",
  backend: "llama.cpp",
  platform: "macos",
  loading: false,
  updating: false,
  ...extra,
});
const installed = {
  id: "tiny",
  name: "Tiny model",
  source: "lma",
  model: "/models/tiny.gguf",
  ollama: false,
  ollamaDir: null,
  projector: null,
  sizeBytes: 2 * 1024 ** 3,
};
const ollama = {
  id: "ollama:coder",
  name: "coder:latest",
  source: "ollama",
  model: "coder:latest",
  ollama: true,
  ollamaDir: "/custom/ollama",
  projector: null,
  sizeBytes: 3 * 1024 ** 3,
};
const savedModel = {
  model: installed.model,
  ollama: false,
  ollamaDir: null,
  projector: null,
  gpu: "7",
  context: 512,
};
const response = (content = "Whole response") => ({
  status: 200,
  body: {
    model: "tiny.gguf",
    choices: [{ message: { content }, finish_reason: "stop" }],
    usage: { prompt_tokens: 2, completion_tokens: 3, total_tokens: 5 },
  },
});
function clock() {
  const tasks = new Map();
  let sequence = 0;
  return {
    tasks,
    setTimer(fn, delay) {
      const id = ++sequence;
      tasks.set(id, { fn, delay });
      return id;
    },
    clearTimer(id) {
      tasks.delete(id);
    },
    async fire(delay) {
      for (const [id, task] of [...tasks])
        if (delay === undefined || task.delay === delay) {
          tasks.delete(id);
          await task.fn();
        }
      await drain();
    },
  };
}

test("workspace allowlists the exact host schema and round-trips drafts, images and metadata", () => {
  const c = newConversation({
    attachments: [png],
    draft: "draft",
    error: "never persist",
    settings: {
      ...core.DEFAULT_SETTINGS,
      system: "local",
      coding: true,
      extra: true,
    },
    messages: [
      {
        role: "user",
        content: "image",
        images: [jpeg],
        mode: "chat",
        settings: core.DEFAULT_SETTINGS,
        active: true,
      },
      {
        role: "assistant",
        content: "answer",
        elapsed: 13,
        usage: { prompt_tokens: 1, total_tokens: 3, private: "no" },
        finish: "stop",
        model: "tiny",
        tools: "never",
      },
    ],
  });
  const result = normalizeWorkspace({
    ...workspaceOf(c),
    key: "no",
    active: true,
  });
  assert.deepEqual(Object.keys(result), [
    "version",
    "currentId",
    "conversations",
  ]);
  assert.deepEqual(Object.keys(result.conversations[0]), [
    "id",
    "title",
    "createdAt",
    "updatedAt",
    "settings",
    "draft",
    "attachments",
    "messages",
  ]);
  assert.deepEqual(
    Object.keys(result.conversations[0].settings),
    Object.keys(core.DEFAULT_SETTINGS),
  );
  assert.deepEqual(Object.keys(result.conversations[0].attachments[0]).sort(), [
    "base64",
    "mime",
    "name",
    "size",
  ]);
  assert.equal(result.conversations[0].messages[0].active, undefined);
  assert.deepEqual(result.conversations[0].messages[1].usage, {
    prompt_tokens: 1,
    total_tokens: 3,
  });
  assert.equal(result.conversations[0].messages[1].tools, undefined);
  assert.deepEqual(normalizeWorkspace(clone(result)), result);
  assert.equal(core.DEFAULT_SETTINGS.max_tokens, 128);
});

test("workspace limits reject atomically, unknown current IDs and unsafe images cannot reach disk", () => {
  const c = newConversation();
  for (const value of [
    null,
    {},
    { ...emptyWorkspace(), version: 2 },
    { ...workspaceOf(c), currentId: "missing" },
    workspaceOf(c, c),
    workspaceOf({ ...c, id: "a".repeat(129) }),
    workspaceOf({ ...c, title: "a".repeat(257) }),
    workspaceOf({ ...c, createdAt: "a".repeat(65) }),
    workspaceOf({ ...c, draft: "🫧".repeat(CAPS.text / 4 + 1) }),
    workspaceOf({ ...c, attachments: [{ ...png, size: 1 }] }),
    workspaceOf({ ...c, attachments: [{ ...png, mime: "image/svg+xml" }] }),
    workspaceOf({ ...c, messages: [{ role: "system", content: "bad" }] }),
    workspaceOf({
      ...c,
      messages: [{ role: "assistant", content: "", images: [png] }],
    }),
    workspaceOf({
      ...c,
      messages: Array.from({ length: 1001 }, () => ({
        role: "user",
        content: "",
      })),
    }),
    {
      ...emptyWorkspace(),
      conversations: Array.from({ length: 101 }, () => newConversation()),
    },
  ])
    assert.throws(() => normalizeWorkspace(value));
  const many = Array.from({ length: 11 }, () =>
    newConversation({
      messages: Array.from({ length: 1000 }, () => ({
        role: "user",
        content: "",
      })),
    }),
  );
  assert.throws(() => normalizeWorkspace(workspaceOf(...many)), /10000/);
});

test("branching retains originals and snapshots the selected turn settings and images", () => {
  const original = newConversation({
    title: "Original",
    messages: [
      {
        role: "user",
        content: "first",
        images: [png],
        settings: { ...core.DEFAULT_SETTINGS, temperature: 0.2 },
      },
      { role: "assistant", content: "answer" },
      { role: "user", content: "second" },
    ],
  });
  const before = clone(original);
  const branch = branchConversation(original, 0);
  assert.notEqual(branch.id, original.id);
  assert.deepEqual(branch.messages, []);
  assert.equal(branch.draft, "first");
  assert.equal(branch.settings.temperature, 0.2);
  branch.attachments[0].name = "changed";
  assert.deepEqual(original, before);
  assert.deepEqual(
    branchConversation(original, 2).messages,
    before.messages.slice(0, 2),
  );
  assert.throws(() => branchConversation(original, 1));
  assert.equal(
    new Set(Array.from({ length: 1000 }, () => newConversation().id)).size,
    1000,
  );
});

test("autosave does not write before hydration and serializes newer edits behind pending saves", async () => {
  const time = clock(),
    read = deferred(),
    writes = [];
  const c = newConversation({ draft: "on disk" });
  const store = createWorkspaceStore({
    ...time,
    invoke(command, args) {
      if (command === "workspace_load") return read.promise;
      const pending = deferred();
      writes.push({ ...pending, snapshot: clone(args.workspace) });
      return pending.promise;
    },
  });
  const hydration = store.hydrate();
  await time.fire();
  assert.equal(writes.length, 0);
  assert.equal(await store.flush(), true);
  read.resolve(workspaceOf(c));
  assert.equal(await hydration, true);
  assert.equal(store.current.draft, "on disk");
  assert.equal(writes.length, 0);
  store.update(c.id, (v) => {
    v.draft = "first edit";
  });
  const saving = store.flush();
  assert.equal(writes.length, 1);
  store.update(c.id, (v) => {
    v.draft = "newer edit";
    v.settings.system = "local system";
  });
  assert.equal(store.flush(), saving);
  assert.equal(writes.length, 1);
  writes[0].resolve(null);
  await drain();
  assert.equal(writes.length, 2);
  assert.equal(writes[0].snapshot.conversations[0].draft, "first edit");
  assert.equal(writes[1].snapshot.conversations[0].draft, "newer edit");
  writes[1].resolve(null);
  assert.equal(await saving, true);
  assert.equal(store.persistence.dirty, false);
  store.dispose();
});

test("failed reads block all writes; retry merges working chats rather than overwriting either copy", async () => {
  const time = clock(),
    c = newConversation({ title: "Disk conversation" });
  let broken = true;
  const writes = [];
  const store = createWorkspaceStore({
    ...time,
    invoke: async (command, args) => {
      if (command === "workspace_load") {
        if (broken) throw new Error("permission denied");
        return workspaceOf(c);
      }
      writes.push(args.workspace);
    },
  });
  assert.equal(await store.hydrate(), false);
  const id = store.add(newConversation({ draft: "unsaved local work" }));
  await time.fire();
  await store.flush();
  assert.equal(writes.length, 0);
  assert.match(store.persistence.readError, /permission denied/);
  broken = false;
  assert.equal(await store.hydrate(), true);
  assert.equal(store.current.id, id);
  assert.equal(store.workspace.conversations.length, 2);
  await time.fire(400);
  assert.equal(writes.at(-1).conversations.length, 2);
  assert.equal(store.persistence.readError, "");
  store.dispose();
});

test("save failures preserve dirty state and new chats; retries and navigation flush the latest data", async () => {
  const time = clock(),
    c = newConversation({ title: "Keep me" });
  let broken = true;
  const writes = [];
  const store = createWorkspaceStore({
    ...time,
    invoke: async (command, args) => {
      if (command === "workspace_load") return workspaceOf(c);
      if (broken) throw new Error("disk full");
      writes.push(args.workspace);
    },
  });
  await store.hydrate();
  const id = store.add();
  assert.equal(await store.flush(), false);
  assert.match(store.persistence.saveError, /disk full/);
  assert.equal(store.persistence.dirty, true);
  assert.equal(store.workspace.conversations.length, 2);
  assert.throws(() => store.rename(id, "a".repeat(257)));
  assert.equal(store.current.title, "New conversation");
  broken = false;
  store.rename(id, "Latest title");
  store.select(c.id);
  await drain();
  assert.equal(writes.at(-1).currentId, c.id);
  assert.equal(
    writes.at(-1).conversations.find((v) => v.id === id).title,
    "Latest title",
  );
  assert.equal(store.persistence.saveError, "");
  store.dispose();
});

// Tiny strict DOM, based on the browser UI test helper. No HTML parser is ever
// applied to generated content; this parser reads only our checked-in shell.
class Node {
  constructor(tagName, doc) {
    Object.assign(this, {
      tagName,
      ownerDocument: doc,
      children: [],
      attributes: {},
      events: new Map(),
      className: "",
      dataset: {},
      style: {},
      value: "",
      hidden: false,
      disabled: false,
      checked: false,
      scrollTop: 0,
      scrollHeight: 1200,
      clientHeight: 600,
      _text: "",
    });
    this.classList = {
      contains: (name) => this.className.split(/\s+/).includes(name),
      add: (name) => {
        if (!this.classList.contains(name)) this.className += ` ${name}`;
      },
      remove: (name) => {
        this.className = this.className
          .split(/\s+/)
          .filter((v) => v !== name)
          .join(" ");
      },
      toggle: (name, force) => {
        const add = force ?? !this.classList.contains(name);
        this.classList[add ? "add" : "remove"](name);
        return add;
      },
    };
  }
  get firstChild() {
    return this.children[0] || null;
  }
  get textContent() {
    return this._text + this.children.map((v) => v.textContent).join("");
  }
  set textContent(value) {
    this.replaceChildren();
    this._text = String(value);
  }
  set innerHTML(_) {
    this.ownerDocument.unsafe("innerHTML");
  }
  set outerHTML(_) {
    this.ownerDocument.unsafe("outerHTML");
  }
  insertAdjacentHTML() {
    this.ownerDocument.unsafe("insertAdjacentHTML");
  }
  append(...nodes) {
    for (let node of nodes) {
      if (typeof node === "string")
        node = this.ownerDocument.createTextNode(node);
      if (node.tagName === "#fragment") {
        this.append(...node.children.slice());
        continue;
      }
      node.remove();
      node.parent = this;
      this.children.push(node);
    }
  }
  replaceChildren(...nodes) {
    this._text = "";
    for (const child of this.children) child.parent = null;
    this.children = [];
    this.append(...nodes);
  }
  remove() {
    if (this.parent)
      this.parent.children = this.parent.children.filter((n) => n !== this);
    this.parent = null;
  }
  setAttribute(name, value) {
    if (/^on/i.test(name) || name === "style") this.ownerDocument.unsafe(name);
    this.attributes[name] = String(value);
    if (name === "class") this.className = String(value);
    if (name === "value") this.value = String(value);
    if (["hidden", "disabled", "checked"].includes(name)) this[name] = true;
  }
  getAttribute(name) {
    return this.attributes[name] ?? null;
  }
  removeAttribute(name) {
    delete this.attributes[name];
  }
  addEventListener(type, listener) {
    if (!this.events.has(type)) this.events.set(type, []);
    this.events.get(type).push(listener);
  }
  removeEventListener(type, listener) {
    this.events.set(
      type,
      (this.events.get(type) || []).filter((v) => v !== listener),
    );
  }
  emit(type, extra = {}) {
    const event = {
      target: this,
      currentTarget: this,
      defaultPrevented: false,
      preventDefault() {
        this.defaultPrevented = true;
      },
      ...extra,
    };
    for (const listener of this.events.get(type) || []) listener(event);
    return event;
  }
  click() {
    if (this.disabled) return;
    if (this.tagName === "a" && this.download)
      this.ownerDocument.downloads.push({
        name: this.download,
        url: this.href,
      });
    this.emit("click");
  }
  focus() {
    this.ownerDocument.activeElement = this;
  }
  select() {}
  querySelectorAll(selector) {
    const result = [],
      matches = (n) =>
        selector.split(",").some((part) => {
          part = part.trim();
          if (part.startsWith(".")) return n.classList.contains(part.slice(1));
          const attr = /^(\w+)?\[([\w-]+)(?:="([^"]*)")?\]$/.exec(part);
          if (attr)
            return (
              (!attr[1] || n.tagName === attr[1]) &&
              n.getAttribute(attr[2]) !== null &&
              (attr[3] === undefined || n.getAttribute(attr[2]) === attr[3])
            );
          return n.tagName === part;
        });
    const visit = (parent) => {
      for (const node of parent.children) {
        if (matches(node)) result.push(node);
        visit(node);
      }
    };
    visit(this);
    return result;
  }
  querySelector(selector) {
    return this.querySelectorAll(selector)[0] || null;
  }
}
function makeDocument() {
  const doc = new Node("#document");
  doc.ownerDocument = doc;
  doc.ids = new Map();
  doc.downloads = [];
  doc.violations = [];
  doc.unsafe = (message) => {
    doc.violations.push(message);
    throw new Error(`Unsafe DOM operation: ${message}`);
  };
  doc.createElement = (tag) => new Node(tag, doc);
  doc.createElementNS = (ns, tag) => {
    assert.equal(ns, "http://www.w3.org/2000/svg");
    assert.ok(["svg", "path"].includes(tag));
    return new Node(tag, doc);
  };
  doc.createTextNode = (text) => {
    const n = new Node("#text", doc);
    n.textContent = text;
    return n;
  };
  doc.createDocumentFragment = () => new Node("#fragment", doc);
  doc.getElementById = (id) => {
    assert.ok(doc.ids.has(id), `Missing shell ID: ${id}`);
    return doc.ids.get(id);
  };
  const stack = [doc],
    voidTags = new Set(["meta", "link", "input", "br", "img", "hr"]);
  for (const match of shell.matchAll(/<\/?([a-z][\w-]*)\b[^>]*>/gi)) {
    const tag = match[1].toLowerCase();
    if (match[0].startsWith("</")) {
      assert.equal(stack.pop().tagName, tag);
      continue;
    }
    const n = new Node(tag, doc);
    for (const attr of match[0]
      .slice(match[0].indexOf(tag) + tag.length, -1)
      .matchAll(/([\w:-]+)(?:\s*=\s*"([^"]*)")?/g)) {
      n.setAttribute(attr[1], attr[2] ?? "");
      if (attr[1] === "id") {
        assert.ok(!doc.ids.has(attr[2]));
        doc.ids.set(attr[2], n);
      }
    }
    stack.at(-1).append(n);
    if (tag === "body") doc.body = n;
    if (!voidTags.has(tag) && !match[0].endsWith("/>")) stack.push(n);
  }
  assert.deepEqual(stack, [doc]);
  return doc;
}
const speechPlan = (
  repository = "fixture-owner/speech-assets",
  overrides = {},
) => {
  const revision = "a".repeat(40);
  const files = [
    { name: "config.json", bytes: 4096, sha256: null },
    { name: "tokenizer.json", bytes: 2097152, sha256: null },
    { name: "model.safetensors", bytes: 67108864, sha256: "b".repeat(64) },
  ];
  return {
    repository,
    revision,
    files,
    total_bytes: files.reduce((sum, file) => sum + file.bytes, 0),
    directory: `/models/huggingface/${repository}/${revision}`,
    ...overrides,
  };
};

async function harness(t, options = {}) {
  const document = makeDocument(),
    time = clock(),
    calls = [],
    blobs = new Map(),
    clipboard = [],
    readers = [],
    confirmations = [];
  const globals = new Map();
  const install = (key, value) => {
    globals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, {
      value,
      configurable: true,
      writable: true,
    });
  };
  const window = new Node("#window", document);
  window.confirm = (message) => {
    confirmations.push(message);
    return options.confirm !== false;
  };
  window.FileReader = class {
    readAsDataURL(file) {
      readers.push(this);
      this.result = file.data || `data:${file.type};base64,${png.base64}`;
      if (!options.deferImages) queueMicrotask(() => this.onload());
    }
  };
  install("document", document);
  install("window", window);
  install("navigator", {
    clipboard: {
      writeText: async (text) => {
        clipboard.push(text);
      },
    },
  });
  install("setTimeout", time.setTimer);
  install("clearTimeout", time.clearTimer);
  install(
    "localStorage",
    new Proxy(
      {},
      {
        get() {
          throw new Error("localStorage forbidden");
        },
      },
    ),
  );
  install("fetch", () => {
    throw new Error("Browser network forbidden");
  });
  const NativeURL = URL;
  install(
    "URL",
    class extends NativeURL {
      static createObjectURL(blob) {
        const id = `blob:test-${blobs.size}`;
        blobs.set(id, blob);
        return id;
      }
      static revokeObjectURL() {}
    },
  );
  let hostStatus = options.status || status();
  let disk = clone(options.workspace || emptyWorkspace());
  let appSettings = clone({ ...DEFAULT_APP_SETTINGS, ...options.appSettings });
  const invoke = async (command, args) => {
    calls.push({ command, args: args ? clone(args) : undefined });
    if (options.handlers?.[command]) return options.handlers[command](args);
    switch (command) {
      case "app_settings_load":
        return clone(appSettings);
      case "app_settings_save":
        appSettings = clone(args.settings);
        return null;
      case "store_search":
        return clone(options.storeResults || { models: [], warnings: [] });
      case "store_download":
        return { path: "/models/download.gguf", model: args.id };
      case "model_import":
        return { model: args.options.model, path: args.options.model };
      case "dictation_settings":
        return { modelPath: options.dictationModel || null };
      case "dictation_set_model":
        return { modelPath: args.modelPath };
      case "dictation_models":
        return {
          models: clone(options.dictationModels || []),
          warnings: clone(options.dictationWarnings || []),
        };
      case "dictation_model_plan":
        return speechPlan(args.repository);
      case "dictation_install_model":
        return { modelPath: args.plan.directory };
      case "dictation_transcribe":
        return { text: "Dictated text", language: "en" };
      case "desktop_status":
        return clone(hostStatus);
      case "workspace_load":
        return clone(disk);
      case "workspace_save":
        disk = clone(args.workspace);
        return null;
      case "discover_models":
        return clone(
          options.discovery || {
            models: [installed, ollama],
            warnings: [],
            lastModel: null,
          },
        );
      case "load_model":
        hostStatus = status({ model: args.options.model });
        return clone(hostStatus);
      case "unload_model":
        hostStatus = status({ loaded: false, model: null });
        return clone(hostStatus);
      case "cancel_request":
        return null;
      case "api_request":
        return response();
      case "update_status":
        return { state: "Completed", log: "Build published", error: null };
      case "source_update":
        return { message: "Build published" };
      default:
        throw new Error(`Undocumented IPC command: ${command}`);
    }
  };
  const app = createDesktopApp({ document, window, invoke, ...time });
  t.after(async () => {
    app.dispose();
    await drain();
    assert.deepEqual(document.violations, []);
    for (const [key, descriptor] of globals) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
  });
  const get = (id) => document.getElementById(id);
  const h = {
    app,
    get,
    document,
    window,
    time,
    calls,
    blobs,
    clipboard,
    readers,
    confirmations,
    setStatus(v) {
      hostStatus = v;
    },
    posts: () => calls.filter((v) => v.command === "api_request"),
    count: (command) => calls.filter((v) => v.command === command).length,
    draft(text) {
      get("draft").value = text;
      get("draft").emit("input");
    },
    async click(id) {
      get(id).click();
      await drain();
    },
    async export(format) {
      app.exportConversation(format);
      const entry = document.downloads.at(-1);
      return { name: entry.name, text: await blobs.get(entry.url).text() };
    },
  };
  if (options.start !== false) await app.start();
  return h;
}

function mockMicrophone(h, options = {}) {
  const tracks = [],
    contexts = [],
    requests = [];
  h.window.navigator = {
    mediaDevices: {
      getUserMedia: async (constraints) => {
        requests.push(constraints);
        if (options.error) throw options.error;
        if (options.permission) return options.permission.promise;
        return stream;
      },
    },
  };
  const track = {
    stopped: 0,
    onended: null,
    stop() {
      this.stopped++;
    },
  };
  tracks.push(track);
  const stream = { getTracks: () => tracks };
  const node = () => ({
    connections: [],
    disconnected: 0,
    connect(other) {
      this.connections.push(other);
    },
    disconnect() {
      this.disconnected++;
    },
  });
  h.window.AudioContext = class {
    constructor() {
      this.sampleRate = options.sampleRate || 48000;
      this.state = "suspended";
      this.destination = node();
      this.closed = 0;
      contexts.push(this);
    }
    async resume() {
      this.state = "running";
    }
    async close() {
      this.closed++;
      this.state = "closed";
    }
    createMediaStreamSource() {
      this.source = node();
      return this.source;
    }
    createScriptProcessor() {
      this.processor = node();
      return this.processor;
    }
    createGain() {
      this.gain = { ...node(), gain: { value: 1 } };
      return this.gain;
    }
  };
  return {
    track,
    tracks,
    contexts,
    requests,
    stream,
    feed(channels = [new Float32Array(4800).fill(0.5)]) {
      const processor = contexts.at(-1).processor;
      assert.ok(
        processor.onaudioprocess,
        "Microphone callback must still be connected",
      );
      processor.onaudioprocess({
        inputBuffer: {
          numberOfChannels: channels.length,
          getChannelData: (i) => channels[i],
        },
      });
    },
  };
}
const setSpeechRepository = (h, value) => {
  h.get("dictation-repository").value = value;
  h.get("dictation-repository").emit("input");
};
const dictationShortcut = (h, extra = {}) =>
  h.document.emit("keydown", {
    key: "D",
    ctrlKey: true,
    shiftKey: true,
    ...extra,
  });

test("dictation setup remembers only a local model path and never asks for a microphone on startup or setup", async (t) => {
  const h = await harness(t);
  const mic = mockMicrophone(h);
  assert.equal(h.count("dictation_settings"), 1);
  assert.equal(mic.requests.length, 0);
  await h.click("dictation-toggle");
  assert.equal(h.get("dictation-setup").hidden, false);
  assert.equal(mic.requests.length, 0);
  h.get("dictation-model-path").value = "/local/whisper-tiny";
  await h.click("dictation-save-model");
  assert.deepEqual(
    h.calls.find((c) => c.command === "dictation_set_model").args,
    { modelPath: "/local/whisper-tiny" },
  );
  assert.equal(h.get("dictation-setup").hidden, true);
  assert.equal(mic.requests.length, 0);
  dictationShortcut(h);
  await drain();
  assert.equal(mic.requests.length, 0);
  assert.match(h.get("dictation-status").textContent, /Click the microphone/);
  await h.click("dictation-toggle");
  assert.equal(mic.requests.length, 1);
  assert.equal(mic.requests[0].video, false);
  assert.equal(h.get("dictation-toggle").getAttribute("aria-pressed"), "true");
  h.document.emit("keydown", { key: "Escape" });
  assert.equal(mic.track.stopped, 1);
  assert.equal(h.count("dictation_transcribe"), 0);
  assert.equal(h.get("dictation-toggle").getAttribute("aria-pressed"), "false");
});

test("speech installation requires repository inspection and a second explicit click on the displayed immutable plan", async (t) => {
  const inspecting = deferred(),
    installing = deferred();
  const plan = speechPlan();
  const c = newConversation({ draft: "Keep draft", attachments: [png] });
  const h = await harness(t, {
    workspace: workspaceOf(c),
    handlers: {
      dictation_model_plan: () => inspecting.promise,
      dictation_install_model: () => installing.promise,
    },
  });
  const mic = mockMicrophone(h);
  assert.equal(h.count("dictation_models"), 0);
  assert.equal(h.count("dictation_model_plan"), 0);
  assert.equal(h.count("dictation_install_model"), 0);
  await h.click("dictation-toggle");
  assert.equal(h.count("dictation_models"), 1);
  assert.equal(h.get("dictation-repository").value, "");
  assert.equal(
    h.get("dictation-install-model").textContent,
    "Inspect download",
  );
  await h.click("dictation-install-model");
  assert.match(h.get("dictation-error").textContent, /owner\/repository/);
  assert.equal(h.count("dictation_model_plan"), 0);
  setSpeechRepository(h, plan.repository);
  await h.click("dictation-install-model");
  assert.deepEqual(
    h.calls.find((c) => c.command === "dictation_model_plan").args,
    { repository: plan.repository },
  );
  assert.match(
    h.get("dictation-status").textContent,
    /Inspecting repository metadata/,
  );
  assert.equal(h.get("dictation-repository").disabled, true);
  assert.equal(
    h.get("dictation-install-model").getAttribute("aria-busy"),
    "true",
  );
  assert.equal(h.count("dictation_install_model"), 0);
  inspecting.resolve(plan);
  await drain();
  assert.equal(h.get("dictation-plan").hidden, false);
  assert.equal(
    h.get("dictation-plan-summary").textContent,
    `${plan.repository} · revision ${plan.revision} · 3 files · ${plan.total_bytes} bytes (${(plan.total_bytes / 1048576).toFixed(1)} MiB)`,
  );
  assert.deepEqual(
    h.get("dictation-plan-files").children.map((n) => n.textContent),
    plan.files.map((f) => `${f.name} · ${f.bytes} bytes`),
  );
  assert.equal(
    h.get("dictation-plan-directory").textContent,
    `Destination: ${plan.directory}`,
  );
  assert.equal(
    h.get("dictation-install-model").textContent,
    `Install (${(plan.total_bytes / 1048576).toFixed(1)} MiB)`,
  );
  assert.equal(h.count("dictation_install_model"), 0);
  assert.equal(mic.requests.length, 0);
  await h.click("dictation-install-model");
  assert.deepEqual(
    h.calls.find((c) => c.command === "dictation_install_model").args,
    { plan },
  );
  assert.match(
    h.get("dictation-status").textContent,
    /Downloading listed model files/,
  );
  for (const id of [
    "dictation-toggle",
    "dictation-configure",
    "dictation-install-model",
    "dictation-repository",
    "dictation-save-model",
    "dictation-model-path",
    "dictation-close-setup",
    "dictation-installed-models",
    "dictation-use-installed",
    "dictation-rescan-models",
  ])
    assert.equal(h.get(id).disabled, true, id);
  h.draft("Edited during download");
  await h.click("dictation-install-model");
  dictationShortcut(h);
  h.document.emit("keydown", { key: "Escape" });
  await drain();
  assert.equal(h.count("dictation_install_model"), 1);
  assert.equal(h.get("dictation-setup").hidden, false);
  installing.resolve({ modelPath: plan.directory });
  await drain();
  assert.equal(h.get("dictation-setup").hidden, true);
  assert.equal(
    h.get("dictation-install-model").getAttribute("aria-busy"),
    "false",
  );
  assert.equal(h.get("dictation-toggle").disabled, false);
  assert.match(h.get("dictation-status").textContent, /Model installed/);
  assert.equal(mic.requests.length, 0);
  assert.equal(h.posts().length, 0);
  assert.equal(h.app.store.current.draft, "Edited during download");
  assert.deepEqual(h.app.store.current.attachments, [png]);
  await h.click("dictation-configure");
  assert.equal(h.get("dictation-model-path").value, plan.directory);
  assert.equal(h.get("dictation-plan").hidden, true);
  await h.click("dictation-close-setup");
  dictationShortcut(h);
  await drain();
  assert.equal(mic.requests.length, 0);
  await h.click("dictation-toggle");
  assert.equal(mic.requests.length, 1);
});

test("failed speech installation requires fresh inspection and preserves the previous model and draft", async (t) => {
  let fail = true;
  const h = await harness(t, {
    dictationModel: "/local/existing-speech-model",
    handlers: {
      dictation_install_model: async ({ plan }) => {
        if (fail) throw new Error("Installation plan changed; inspect again.");
        return { modelPath: plan.directory };
      },
    },
  });
  const mic = mockMicrophone(h);
  h.draft("Keep this");
  await h.click("dictation-configure");
  setSpeechRepository(h, speechPlan().repository);
  await h.click("dictation-install-model");
  await h.click("dictation-install-model");
  assert.equal(h.get("dictation-setup").hidden, false);
  assert.match(h.get("dictation-error").textContent, /inspect again/);
  assert.equal(h.get("dictation-status").textContent, "");
  assert.equal(h.get("dictation-plan").hidden, true);
  assert.equal(
    h.get("dictation-install-model").textContent,
    "Inspect download",
  );
  assert.equal(
    h.get("dictation-model-path").value,
    "/local/existing-speech-model",
  );
  assert.equal(h.app.store.current.draft, "Keep this");
  fail = false;
  await h.click("dictation-install-model");
  assert.equal(h.count("dictation_model_plan"), 2);
  assert.equal(h.count("dictation_install_model"), 1);
  await h.click("dictation-install-model");
  assert.equal(h.count("dictation_install_model"), 2);
  assert.equal(h.get("dictation-error").textContent, "");
  assert.equal(h.get("dictation-setup").hidden, true);
  assert.equal(mic.requests.length, 0);
  assert.equal(h.posts().length, 0);
});

test("late speech inspection, installation and model scans never update a disposed app", async (t) => {
  for (const command of [
    "dictation_model_plan",
    "dictation_install_model",
    "dictation_models",
  ]) {
    await t.test(command, async (t) => {
      const pending = deferred();
      const h = await harness(t, {
        handlers: { [command]: () => pending.promise },
      });
      const mic = mockMicrophone(h);
      await h.click("dictation-toggle");
      setSpeechRepository(h, speechPlan().repository);
      if (command !== "dictation_models")
        await h.click("dictation-install-model");
      if (command === "dictation_install_model")
        await h.click("dictation-install-model");
      const summary = h.get("dictation-plan-summary").textContent;
      h.app.dispose();
      pending.resolve(
        command === "dictation_models"
          ? { models: [], warnings: [] }
          : command === "dictation_model_plan"
            ? speechPlan()
            : { modelPath: speechPlan().directory },
      );
      await drain();
      assert.equal(h.get("dictation-setup").hidden, false);
      assert.equal(h.get("dictation-plan-summary").textContent, summary);
      assert.equal(mic.requests.length, 0);
      assert.equal(h.posts().length, 0);
    });
  }
});

test("editing or reopening the repository clears prior consent and requires a fresh size inspection", async (t) => {
  const h = await harness(t);
  const mic = mockMicrophone(h);
  await h.click("dictation-toggle");
  setSpeechRepository(h, "fixture-owner/first-model");
  await h.click("dictation-install-model");
  assert.equal(h.get("dictation-plan").hidden, false);
  setSpeechRepository(h, "another-owner/second-model");
  assert.equal(h.get("dictation-plan").hidden, true);
  assert.equal(h.get("dictation-plan-summary").textContent, "");
  assert.equal(
    h.get("dictation-install-model").textContent,
    "Inspect download",
  );
  await h.click("dictation-install-model");
  assert.equal(h.count("dictation_model_plan"), 2);
  assert.equal(h.count("dictation_install_model"), 0);
  assert.match(
    h.get("dictation-plan-summary").textContent,
    /another-owner\/second-model/,
  );
  await h.click("dictation-close-setup");
  await h.click("dictation-configure");
  assert.equal(h.get("dictation-plan").hidden, true);
  await h.click("dictation-install-model");
  assert.equal(h.count("dictation_model_plan"), 3);
  assert.equal(h.count("dictation_install_model"), 0);
  assert.equal(mic.requests.length, 0);
});

test("repository changes during inspection discard the response rather than presenting stale consent", async (t) => {
  const pending = deferred();
  const h = await harness(t, {
    handlers: { dictation_model_plan: () => pending.promise },
  });
  await h.click("dictation-toggle");
  setSpeechRepository(h, speechPlan().repository);
  await h.click("dictation-install-model");
  assert.equal(h.get("dictation-repository").disabled, true);
  h.get("dictation-repository").value = "fixture-owner/different-model";
  pending.resolve(speechPlan());
  await drain();
  assert.match(h.get("dictation-error").textContent, /Repository changed/);
  assert.equal(h.get("dictation-plan").hidden, true);
  assert.equal(h.count("dictation_install_model"), 0);
});

test("inspection failures and malformed plans never enable downloads or consume the working draft", async (t) => {
  for (const result of [
    null,
    speechPlan(undefined, { total_bytes: 1 }),
    speechPlan(undefined, { files: [] }),
    speechPlan("different/repository"),
  ]) {
    await t.test(result?.repository || "network failure", async (t) => {
      const h = await harness(t, {
        handlers: {
          dictation_model_plan: async () => {
            if (!result)
              throw new Error(
                "Cannot inspect repository; check network access.",
              );
            return result;
          },
        },
      });
      const mic = mockMicrophone(h);
      h.draft("Retain this");
      await h.click("dictation-toggle");
      setSpeechRepository(h, speechPlan().repository);
      await h.click("dictation-install-model");
      assert.match(
        h.get("dictation-error").textContent,
        /Cannot inspect|Invalid model installation plan/,
      );
      assert.equal(h.get("dictation-plan").hidden, true);
      assert.equal(h.get("dictation-repository").disabled, false);
      assert.equal(
        h.get("dictation-install-model").textContent,
        "Inspect download",
      );
      assert.equal(h.count("dictation_install_model"), 0);
      assert.equal(h.app.store.current.draft, "Retain this");
      assert.equal(mic.requests.length, 0);
    });
  }
});

test("installed speech models are real scan results and require explicit selection before settings change", async (t) => {
  const models = [
    {
      name: "fixture-owner/local-speech",
      modelPath: "/models/huggingface/local-speech/revision",
    },
    {
      name: "Config model name <script>literal</script>",
      modelPath: "/local/other-speech-model",
    },
  ];
  const h = await harness(t, {
    dictationModels: models,
    dictationWarnings: ["Scan depth limit reached"],
  });
  const mic = mockMicrophone(h);
  h.draft("Keep text");
  assert.equal(h.count("dictation_models"), 0);
  await h.click("dictation-toggle");
  assert.equal(h.count("dictation_models"), 1);
  assert.equal(h.get("dictation-installed-models").children.length, 3);
  assert.equal(
    h.get("dictation-installed-models").children[2].textContent,
    `${models[1].name} — ${models[1].modelPath}`,
  );
  assert.match(
    h.get("dictation-models-status").textContent,
    /2 compatible.*depth limit/,
  );
  assert.equal(h.get("dictation-use-installed").disabled, true);
  assert.equal(h.count("dictation_set_model"), 0);
  h.get("dictation-installed-models").value = models[1].modelPath;
  h.get("dictation-installed-models").emit("change");
  assert.equal(h.get("dictation-use-installed").disabled, false);
  assert.equal(h.count("dictation_set_model"), 0);
  await h.click("dictation-use-installed");
  assert.deepEqual(
    h.calls.find((c) => c.command === "dictation_set_model").args,
    { modelPath: models[1].modelPath },
  );
  assert.equal(h.get("dictation-setup").hidden, true);
  assert.equal(h.app.store.current.draft, "Keep text");
  assert.equal(h.count("dictation_model_plan"), 0);
  assert.equal(h.count("dictation_install_model"), 0);
  assert.equal(mic.requests.length, 0);
  await h.click("dictation-configure");
  assert.equal(h.get("dictation-installed-models").value, models[1].modelPath);
  assert.equal(h.get("dictation-model-path").value, models[1].modelPath);
  await h.click("dictation-rescan-models");
  assert.equal(h.count("dictation_models"), 3);
  assert.equal(h.count("dictation_set_model"), 1);
});

test("empty or failed installed-model scans leave manual selection and inspection available without capture", async (t) => {
  for (const failed of [false, true]) {
    await t.test(failed ? "failed" : "empty", async (t) => {
      const h = await harness(t, {
        handlers: failed
          ? {
              dictation_models: async () => {
                throw new Error("Folder unavailable");
              },
            }
          : {},
      });
      const mic = mockMicrophone(h);
      await h.click("dictation-toggle");
      assert.match(
        h.get("dictation-models-status").textContent,
        failed ? /Folder unavailable/ : /No compatible installed/,
      );
      assert.equal(h.get("dictation-use-installed").disabled, true);
      assert.equal(h.get("dictation-rescan-models").disabled, false);
      assert.equal(h.get("dictation-save-model").disabled, false);
      assert.equal(h.get("dictation-install-model").disabled, false);
      h.get("dictation-model-path").value = "/external/compatible-speech-model";
      await h.click("dictation-save-model");
      assert.equal(h.count("dictation_set_model"), 1);
      assert.equal(h.count("dictation_install_model"), 0);
      assert.equal(mic.requests.length, 0);
    });
  }
});

test("plan metadata is rendered as text and no fixed repository, size or storage preset is offered", async (t) => {
  const plan = speechPlan();
  plan.files.push({
    name: "<img src=x onerror=alert(1)>.json",
    bytes: 37,
    sha256: null,
  });
  plan.total_bytes += 37;
  const h = await harness(t, {
    handlers: { dictation_model_plan: async () => plan },
  });
  await h.click("dictation-toggle");
  setSpeechRepository(h, plan.repository);
  await h.click("dictation-install-model");
  assert.match(
    h.get("dictation-plan-files").textContent,
    /<img src=x onerror=alert\(1\)>\.json/,
  );
  assert.equal(h.count("dictation_install_model"), 0);
  assert.match(shell, /model_type:\s+whisper\s+only/);
  const production = (
    await Promise.all(
      ["app.js", "core.js", "index.html", "style.css", "workspace.js"].map(
        (file) =>
          readFile(new URL(`../desktop/ui/${file}`, import.meta.url), "utf8"),
      ),
    )
  ).join("\n");
  assert.doesNotMatch(
    production,
    /whisper-tiny|151\s*(?:MB|MiB)|app-data\/speech|app_data_dir\(\).*speech/s,
  );
});

test("dictation appends to the latest draft, preserves images, never sends, and keeps processing honest", async (t) => {
  const pending = deferred();
  const c = newConversation({ draft: "Existing", attachments: [png] });
  const h = await harness(t, {
    workspace: workspaceOf(c),
    dictationModel: "/local/whisper-tiny",
    handlers: { dictation_transcribe: () => pending.promise },
  });
  const mic = mockMicrophone(h);
  await h.click("dictation-toggle");
  mic.feed([new Float32Array(4800).fill(1), new Float32Array(4800).fill(-1)]);
  await h.click("dictation-toggle");
  assert.equal(mic.track.stopped, 1);
  assert.equal(mic.contexts[0].closed, 1);
  assert.equal(mic.contexts[0].processor.onaudioprocess, null);
  assert.equal(h.get("dictation-status").textContent, "Transcribing");
  assert.equal(h.get("dictation-toggle").disabled, true);
  assert.equal(h.get("draft").disabled, false);
  const request = h.calls.find(
    (call) => call.command === "dictation_transcribe",
  );
  assert.deepEqual(Object.keys(request.args), ["pcm16"]);
  const bytes = Buffer.from(request.args.pcm16, "base64");
  assert.equal(bytes.length, 1600 * 2);
  assert.ok(
    bytes.every((value) => value === 0),
    "Stereo is downmixed before resampling",
  );
  h.draft("Edited while transcribing");
  h.document.emit("keydown", { key: "Escape" });
  dictationShortcut(h);
  await h.app.send();
  assert.equal(h.get("dictation-status").textContent, "Transcribing");
  assert.equal(h.count("dictation_transcribe"), 1);
  assert.equal(h.posts().length, 0);
  pending.resolve({ text: "  appended words  ", language: "en" });
  await drain();
  assert.equal(
    h.app.store.current.draft,
    "Edited while transcribing appended words",
  );
  assert.deepEqual(h.app.store.current.attachments, [png]);
  assert.equal(h.app.store.current.messages.length, 0);
  assert.equal(h.get("dictation-feedback").hidden, true);
  await h.app.store.flush();
  const saved = h.calls
    .filter((call) => call.command === "workspace_save")
    .at(-1).args;
  assert.doesNotMatch(
    JSON.stringify(saved),
    /"(?:pcm16|modelPath|audio|transcribing)"\s*:/i,
  );
});

test("dictation shortcuts toggle only after a microphone click, and respect dialogs, composition and repeat", async (t) => {
  const h = await harness(t, { dictationModel: "/local/whisper-tiny" });
  const mic = mockMicrophone(h);
  await h.click("dictation-toggle");
  mic.feed();
  dictationShortcut(h, { metaKey: true, ctrlKey: false });
  await drain();
  assert.equal(h.count("dictation_transcribe"), 1);
  assert.equal(h.app.store.current.draft, "Dictated text");
  for (const extra of [
    { repeat: true },
    { isComposing: true },
    { altKey: true },
    { defaultPrevented: true },
  ])
    assert.equal(
      dictationShortcut(h, extra).defaultPrevented,
      !!extra.defaultPrevented,
    );
  assert.equal(mic.requests.length, 1);
  const modal = h.document.createElement("div");
  modal.setAttribute("aria-modal", "true");
  h.document.body.append(modal);
  assert.equal(dictationShortcut(h).defaultPrevented, false);
  modal.remove();
  h.app.openInspector(true);
  assert.equal(dictationShortcut(h).defaultPrevented, false);
  h.app.openInspector(false);
  assert.equal(dictationShortcut(h).defaultPrevented, true);
  await drain();
  assert.equal(mic.requests.length, 2);
  h.document.emit("keydown", { key: "Escape" });
  assert.equal(h.count("dictation_transcribe"), 1);
});

test("dictation stops microphone tracks on conversation/view changes, hidden pages, unload and dispose", async (t) => {
  for (const action of [
    "newChat",
    "models",
    "hidden",
    "beforeunload",
    "pagehide",
    "dispose",
  ]) {
    await t.test(action, async (t) => {
      const h = await harness(t, { dictationModel: "/local/whisper-tiny" });
      const mic = mockMicrophone(h);
      await h.click("dictation-toggle");
      mic.feed();
      if (action === "newChat") h.app.newChat();
      if (action === "models") h.app.setView("models");
      if (action === "hidden") {
        h.document.hidden = true;
        h.document.emit("visibilitychange");
      }
      if (["beforeunload", "pagehide"].includes(action)) h.window.emit(action);
      if (action === "dispose") h.app.dispose();
      assert.equal(mic.track.stopped, 1);
      assert.equal(mic.contexts[0].closed, 1);
      assert.equal(h.count("dictation_transcribe"), 0);
    });
  }
});

test("late microphone permission is cleaned up after Escape or navigation", async (t) => {
  for (const action of ["escape", "newChat", "dispose"]) {
    await t.test(action, async (t) => {
      const permission = deferred();
      const h = await harness(t, { dictationModel: "/local/whisper-tiny" });
      const mic = mockMicrophone(h, { permission });
      await h.click("dictation-toggle");
      assert.equal(
        h.get("dictation-status").textContent,
        "Microphone permission…",
      );
      if (action === "escape") h.document.emit("keydown", { key: "Escape" });
      if (action === "newChat") h.app.newChat();
      if (action === "dispose") h.app.dispose();
      permission.resolve(mic.stream);
      await drain();
      assert.equal(mic.track.stopped, 1);
      assert.equal(h.count("dictation_transcribe"), 0);
    });
  }
});

test("stale dictation results never move to another conversation or overwrite either draft", async (t) => {
  const pending = deferred();
  const h = await harness(t, {
    dictationModel: "/local/whisper-tiny",
    handlers: { dictation_transcribe: () => pending.promise },
  });
  const original = h.app.store.currentId;
  h.draft("Original draft");
  const mic = mockMicrophone(h);
  await h.click("dictation-toggle");
  mic.feed();
  await h.click("dictation-toggle");
  h.app.newChat();
  const other = h.app.store.currentId;
  h.draft("Other draft");
  assert.equal(h.get("dictation-status").textContent, "Transcribing");
  h.app.selectConversation(original);
  pending.resolve({ text: "Stale words" });
  await drain();
  assert.equal(h.app.store.current.draft, "Original draft");
  h.app.selectConversation(other);
  assert.equal(h.app.store.current.draft, "Other draft");
  assert.equal(h.posts().length, 0);
});

test("dictation permission, device and inference errors are actionable and release capture", async (t) => {
  for (const [name, message] of [
    ["NotAllowedError", /denied.*system/i],
    ["NotFoundError", /No microphone/],
    ["NotReadableError", /unavailable or in use/],
    ["SecurityError", /blocked/],
  ]) {
    await t.test(name, async (t) => {
      const h = await harness(t, { dictationModel: "/local/whisper-tiny" });
      const mic = mockMicrophone(h, {
        error: Object.assign(new Error("failed"), { name }),
      });
      await h.click("dictation-toggle");
      assert.match(h.get("dictation-error").textContent, message);
      assert.equal(mic.contexts[0].closed, 1);
      assert.equal(h.get("dictation-toggle").disabled, false);
    });
  }
  await t.test("unavailable API", async (t) => {
    const h = await harness(t, { dictationModel: "/local/whisper-tiny" });
    await h.click("dictation-toggle");
    assert.match(h.get("dictation-error").textContent, /unavailable.*webview/);
  });
  for (const error of ["empty", "inference", "disconnect", "silence"]) {
    await t.test(error, async (t) => {
      const h = await harness(t, {
        dictationModel: "/local/whisper-tiny",
        handlers: {
          dictation_transcribe: async () => {
            if (error === "inference")
              throw new Error("Whisper model is missing. Choose a model.");
            return { text: "" };
          },
        },
      });
      h.draft("Keep me");
      const mic = mockMicrophone(h);
      await h.click("dictation-toggle");
      if (error !== "empty") mic.feed();
      if (error === "disconnect") mic.track.onended();
      else await h.click("dictation-toggle");
      assert.equal(mic.track.stopped, 1);
      assert.match(
        h.get("dictation-error").textContent,
        /No microphone audio|Whisper model|disconnected|No speech/,
      );
      assert.equal(h.app.store.current.draft, "Keep me");
      assert.equal(h.get("dictation-toggle").disabled, false);
    });
  }
});

test("dictation model save and load failures do not trigger capture or discard drafts", async (t) => {
  const h = await harness(t, {
    handlers: {
      dictation_settings: async () => {
        throw new Error("Repair dictation.json");
      },
      dictation_set_model: async () => {
        throw new Error(
          "Choose a Hugging Face Whisper directory, not a GGML/GGUF file.",
        );
      },
    },
  });
  const mic = mockMicrophone(h);
  assert.match(h.get("dictation-error").textContent, /Repair dictation.json/);
  h.draft("Keep draft");
  await h.click("dictation-toggle");
  h.get("dictation-model-path").value = "/wrong.gguf";
  await h.click("dictation-save-model");
  assert.match(
    h.get("dictation-error").textContent,
    /Hugging Face Whisper directory/,
  );
  assert.equal(h.get("dictation-setup").hidden, false);
  assert.equal(mic.requests.length, 0);
  assert.equal(h.app.store.current.draft, "Keep draft");
  h.document.emit("keydown", { key: "Escape" });
  assert.equal(h.get("dictation-setup").hidden, true);
});

test("PCM16 encoder downmixes, clips, carries fractional rates across chunks and bounds IPC below 16 MiB", () => {
  for (const rate of [16000, 44100, 48000]) {
    const a = core.createPcm16Encoder(rate),
      b = core.createPcm16Encoder(rate);
    const input = Float32Array.from({ length: rate }, (_, i) =>
      Math.sin(i / 12),
    );
    a.push([input]);
    for (let i = 0; i < input.length; i += 127)
      b.push([input.subarray(i, i + 127)]);
    assert.equal(a.samples, 16000);
    assert.equal(b.samples, 16000);
    assert.equal(a.finish(), b.finish());
  }
  const encoder = core.createPcm16Encoder(16000);
  encoder.push([Float32Array.from([-2, 0, 2, NaN])]);
  const bytes = Buffer.from(encoder.finish(), "base64");
  assert.deepEqual([...bytes], [0, 128, 0, 0, 255, 127, 0, 0]);
  const bounded = core.createPcm16Encoder(16000);
  assert.equal(
    bounded.push([new Float32Array(core.DICTATION.samples + 1000)]),
    true,
  );
  assert.equal(bounded.samples, core.DICTATION.samples);
  const pcm16 = bounded.finish();
  assert.equal(Buffer.from(pcm16, "base64").length, 3840000);
  assert.ok(
    new TextEncoder().encode(JSON.stringify({ pcm16 })).length <
      16 * 1024 * 1024,
  );
  assert.throws(() => core.createPcm16Encoder(0), /sample rate/);
  assert.throws(
    () => core.createPcm16Encoder(16000).finish(),
    /No microphone audio/,
  );
});

test("recording stops automatically at 120 seconds and sends exactly one bounded clip", async (t) => {
  const h = await harness(t, { dictationModel: "/local/whisper-tiny" });
  const mic = mockMicrophone(h, { sampleRate: 16000 });
  await h.click("dictation-toggle");
  mic.feed([new Float32Array(core.DICTATION.samples + 1000)]);
  await drain();
  assert.equal(mic.track.stopped, 1);
  assert.equal(h.count("dictation_transcribe"), 1);
  assert.equal(h.app.store.current.draft, "Dictated text");
});

test("shell is desktop-local, injection-free, labelled, and has a quiet chat/library/inspector structure", async (t) => {
  const h = await harness(t);
  assert.equal(h.calls[0].command, "desktop_status");
  assert.equal(h.count("discover_models"), 1);
  assert.equal(h.count("source_update"), 0);
  assert.equal(h.confirmations.length, 0);
  assert.equal(h.get("models-view").hidden, true);
  assert.equal(h.get("inspector").hidden, true);
  assert.equal(h.get("manual-import").hidden, true);
  assert.equal(h.document.querySelectorAll("dialog").length, 0);
  for (const node of h.document.querySelectorAll(".icon-button"))
    if (node.tagName === "button") assert.ok(node.getAttribute("aria-label"));
  for (const file of ["app.js", "core.js", "workspace.js", "index.html"]) {
    const text = await readFile(
      new URL(`../desktop/ui/${file}`, import.meta.url),
      "utf8",
    );
    assert.doesNotMatch(
      text,
      /\.innerHTML\s*=|\.outerHTML\s*=|insertAdjacentHTML\s*\(|\beval\s*\(|new Function\s*\(|localStorage|sessionStorage|https:\/\/.+\.(?:js|css)/,
    );
  }
  assert.match(shell, /type="module" src="app.js"/);
  assert.doesNotMatch(shell, /<iframe|<dialog|<canvas|onclick=|cdn/i);
  await h.click("nav-models");
  assert.equal(h.get("chat-view").hidden, true);
  await h.click("nav-chat");
  assert.equal(h.get("chat-view").hidden, false);
  await h.click("conversation-options");
  assert.equal(h.get("inspector").hidden, false);
  h.document.emit("keydown", { key: "Escape" });
  assert.equal(h.get("inspector").hidden, true);
});

test("startup hydrates before writing and restores a saved current selection and draft", async (t) => {
  const loading = deferred(),
    c = newConversation({
      title: "Saved chat",
      draft: "unsent",
      attachments: [png],
      settings: { ...core.DEFAULT_SETTINGS, system: "system" },
      messages: [
        { role: "user", content: "earlier" },
        { role: "assistant", content: "Saved **answer**" },
      ],
    });
  const h = await harness(t, {
    start: false,
    handlers: { workspace_load: () => loading.promise },
  });
  const opening = h.app.start();
  await drain();
  await h.time.fire(400);
  assert.equal(h.count("workspace_save"), 0);
  assert.equal(h.get("new-chat").disabled, true);
  h.get("new-chat").click();
  assert.equal(h.app.store.workspace.conversations.length, 0);
  loading.resolve(workspaceOf(c));
  await opening;
  assert.equal(h.app.store.current.id, c.id);
  assert.equal(h.get("draft").value, "unsent");
  assert.equal(h.get("conversation-title").textContent, "Saved chat");
  assert.match(h.get("messages").textContent, /Saved answer/);
  assert.equal(h.get("attachments").querySelectorAll("img").length, 1);
  assert.equal(h.get("setting-system").value, "system");
  assert.equal(h.count("workspace_save"), 0);
  assert.equal(
    h.get("messages").querySelectorAll("button").at(-1).disabled,
    false,
  );
});

test("model references distinguish registry aliases without rewriting weight paths", () => {
  const first = {
    ...installed,
    id: "lma:team/first:latest",
    name: "team/first:latest",
  };
  const second = {
    ...first,
    id: "lma:team/second:latest",
    name: "team/second:latest",
  };
  assert.equal(modelReference(first), first.id);
  assert.equal(modelReference(second), second.id);
  assert.notEqual(modelIdentity(first), modelIdentity(second));
  assert.equal(
    modelIdentity(first),
    modelIdentity({ ...savedModel, model: first.id }),
  );
  assert.equal(modelReference(installed), installed.model);
  assert.equal(modelReference(ollama), ollama.model);
  assert.equal(
    modelReference({ ...first, id: "lma:not-the-selected-name" }),
    installed.model,
  );
  assert.equal(modelReference(null), "");
  assert.equal(first.model, installed.model);
  assert.equal(second.model, installed.model);
});

test("library loading and quick switch keep profiles sharing one weight file distinct", async (t) => {
  const first = { ...installed, id: "lma:first:latest", name: "first:latest" };
  const second = {
    ...installed,
    id: "lma:second:latest",
    name: "second:latest",
  };
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
    discovery: { models: [first, second], warnings: [], lastModel: null },
  });
  assert.equal(h.get("installed-models").children.length, 2);
  await h.click("load-model");
  assert.equal(
    h.calls.find((v) => v.command === "load_model").args.options.model,
    first.id,
  );
  assert.equal(h.app.state.selected.name, first.name);
  assert.match(h.get("installed-models").children[0].textContent, /Loaded/);
  assert.doesNotMatch(
    h.get("installed-models").children[1].textContent,
    /Loaded/,
  );
  h.get("model-quick-switch").value = modelIdentity(second);
  h.get("model-quick-switch").emit("change");
  await drain();
  assert.equal(h.count("load_model"), 2);
  assert.equal(
    h.calls.filter((v) => v.command === "load_model").at(-1).args.options.model,
    second.id,
  );
  assert.equal(h.app.state.selected.name, second.name);
  assert.equal(h.get("model-quick-switch").value, modelIdentity(second));
  assert.doesNotMatch(
    h.get("installed-models").children[0].textContent,
    /Loaded/,
  );
  assert.match(h.get("installed-models").children[1].textContent, /Loaded/);
  assert.deepEqual(
    h.app.state.models.map((model) => model.model),
    [installed.model, installed.model],
  );
});

test("restore selects the qualified profile rather than its shared weight sibling", async (t) => {
  const first = { ...installed, id: "lma:first:latest", name: "first:latest" };
  const second = {
    ...installed,
    id: "lma:second:latest",
    name: "second:latest",
  };
  const saved = { ...savedModel, model: second.id };
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
    discovery: { models: [first, second], warnings: [], lastModel: saved },
  });
  assert.deepEqual(
    h.calls.find((v) => v.command === "load_model").args.options,
    saved,
  );
  assert.equal(h.app.state.selected.name, second.name);
  assert.equal(h.get("model-quick-switch").value, modelIdentity(second));
  assert.equal(h.get("model-quick-switch").children.length, 3);
  assert.deepEqual(
    h.app.state.models.map((model) => model.model),
    [installed.model, installed.model],
  );
  await h.app.unload();
  await h.app.discover({ restore: true });
  assert.equal(h.count("load_model"), 1);
});

test("loaded status names do not mark every profile sharing weights as loaded", async (t) => {
  const first = { ...installed, id: "lma:first:latest", name: "first:latest" };
  const second = {
    ...installed,
    id: "lma:second:latest",
    name: "second:latest",
  };
  const h = await harness(t, {
    status: status({ model: second.name }),
    discovery: { models: [first, second], warnings: [], lastModel: null },
  });
  assert.equal(h.count("load_model"), 0);
  assert.equal(h.get("model-quick-switch").value, modelIdentity(second));
  assert.doesNotMatch(
    h.get("installed-models").children[0].textContent,
    /Loaded/,
  );
  assert.match(h.get("installed-models").children[1].textContent, /Loaded/);
});

test("discovery restores the exact remembered GPU/context once and never on subsequent scans", async (t) => {
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
    discovery: { models: [installed], warnings: [], lastModel: savedModel },
  });
  const load = h.calls.find((v) => v.command === "load_model");
  assert.deepEqual(load.args, { options: savedModel });
  assert.equal(h.get("load-gpu").value, "7");
  assert.equal(h.get("load-context").value, "512");
  assert.equal(h.count("load_model"), 1);
  await h.app.unload();
  await h.app.discover({ restore: true });
  assert.equal(h.count("load_model"), 1);
  assert.equal(h.get("inspector").hidden, true);
});

test("restore failures are visible without a load loop", async (t) => {
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
    discovery: {
      models: [installed],
      warnings: ["Read-only source skipped"],
      lastModel: savedModel,
    },
    handlers: {
      load_model: async () => {
        throw new Error("bad model");
      },
    },
  });
  assert.equal(h.count("load_model"), 1);
  assert.match(h.get("app-error-text").textContent, /restore.*bad model/s);
  assert.match(h.get("notifications").textContent, /Read-only/);
  assert.equal(h.get("discovery-warning").hidden, true);
  await h.app.discover({ restore: true });
  await h.time.fire();
  assert.equal(h.count("load_model"), 1);
  assert.equal(h.get("discovery-warning").textContent, "");
});

test("startup never restores over loaded, loading, or updating host state", async (t) => {
  for (const host of [
    status(),
    status({ loaded: false, model: null, loading: true }),
    status({ loaded: false, model: null, updating: true }),
  ]) {
    await t.test(JSON.stringify(host), async (t) => {
      const h = await harness(t, {
        status: host,
        discovery: { models: [installed], warnings: [], lastModel: savedModel },
      });
      assert.equal(h.count("load_model"), 0);
      assert.equal(h.count("source_update"), 0);
      assert.equal(h.confirmations.length, 0);
      if (host.loading || host.updating)
        assert.equal(h.get("load-model").disabled, true);
    });
  }
});

test("a fresh status check prevents restore if host becomes loaded during discovery", async (t) => {
  const discovery = deferred();
  const h = await harness(t, {
    start: false,
    status: status({ loaded: false, model: null }),
    handlers: { discover_models: () => discovery.promise },
  });
  const opening = h.app.start();
  await drain();
  h.setStatus(status({ model: "externally-loaded" }));
  discovery.resolve({
    models: [installed],
    warnings: [],
    lastModel: savedModel,
  });
  await opening;
  assert.equal(h.count("load_model"), 0);
  assert.equal(h.app.state.status.model, "externally-loaded");
});

test("model library searches installed sources, progressively exposes manual options, and uses one load/unload button", async (t) => {
  const h = await harness(t, { appSettings: { loadPolicy: "manual" } });
  await h.click("nav-models");
  assert.equal(h.get("installed-models").children.length, 2);
  assert.match(h.get("installed-models").textContent, /lma2.0 GiBLoaded/);
  h.get("model-search").value = "ollama";
  h.get("model-search").emit("input");
  assert.equal(h.get("installed-models").children.length, 1);
  h.get("installed-models").children[0].click();
  assert.equal(h.get("selected-model-name").textContent, ollama.name);
  assert.equal(
    h.get("installed-models").children[0].getAttribute("aria-pressed"),
    "true",
  );
  h.get("load-gpu").value = "cpu";
  h.get("load-context").value = "1024";
  await h.click("load-model");
  assert.deepEqual(
    h.calls.find((v) => v.command === "load_model").args.options,
    {
      model: ollama.model,
      ollama: true,
      ollamaDir: ollama.ollamaDir,
      projector: null,
      gpu: "cpu",
      context: 1024,
    },
  );
  assert.match(h.get("installed-models").textContent, /Loaded/);
  assert.equal(h.get("load-model").textContent, "Unload model");
  await h.click("load-model");
  assert.equal(h.app.state.status.loaded, false);
  h.get("model-quick-switch").value = modelIdentity(installed);
  h.get("model-quick-switch").emit("change");
  await drain();
  assert.equal(h.app.state.status.model, installed.model);
  assert.equal(
    h.calls.filter((v) => v.command === "load_model").at(-1).args.options
      .context,
    1024,
  );
  await h.click("load-model"); // Unload before validating the next load.
  h.get("load-gpu").value = "garbage";
  await h.click("load-model");
  assert.match(h.get("app-error-text").textContent, /GPU/);
  assert.equal(h.count("load_model"), 2);
});

test("Delete model confirms the selected catalog ID, clears stale choices and never loads or edits chats", async (t) => {
  const first = { ...installed, id: "lma:first", name: "first" };
  const second = { ...installed, id: "lma:second", name: "second" };
  const remembered = { ...savedModel, model: first.id };
  const conversation = newConversation({
    title: "Keep me",
    draft: "Untouched",
  });
  let deleted = false;
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
    appSettings: { autoRestore: false },
    workspace: workspaceOf(conversation),
    handlers: {
      discover_models: () => ({
        models: deleted ? [second] : [first, second],
        warnings: [],
        lastModel: remembered,
      }),
      model_delete: ({ id }) => {
        assert.equal(id, first.id);
        deleted = true;
        return { status: "success", retainedFiles: [], warnings: [] };
      },
    },
  });
  await h.click("nav-models");
  const before = h.app.store.workspace;
  assert.equal(h.get("delete-model").hidden, false);
  assert.equal(h.get("delete-model").disabled, false);
  assert.equal(
    h.get("delete-model").getAttribute("aria-describedby"),
    "delete-model-note",
  );
  h.get("manual-model").value = "/arbitrary/do-not-delete.gguf";
  await h.click("delete-model");
  assert.deepEqual(h.calls.find((v) => v.command === "model_delete").args, {
    id: first.id,
  });
  assert.match(h.confirmations[0], /Permanently delete "first" from lma/);
  assert.match(h.confirmations[0], /cannot be undone/);
  assert.match(
    h.confirmations[0],
    /Shared aliases.*entry removed.*external weights may be kept/,
  );
  assert.equal(h.count("discover_models"), 2);
  assert.equal(h.app.state.selected.id, second.id);
  assert.equal(h.app.state.remembered, null);
  assert.ok(
    !h
      .get("model-quick-switch")
      .children.some((v) => v.value === modelIdentity(first)),
  );
  assert.equal(h.app.state.modelBusy, false);
  assert.equal(h.app.state.modelOperation, null);
  assert.match(h.get("notifications").textContent, /first deleted/);
  await h.app.discover();
  assert.equal(
    h.app.state.remembered,
    null,
    "later scans cannot resurrect the deleted remembered choice",
  );
  assert.equal(h.count("load_model"), 0);
  assert.equal(h.count("unload_model"), 0);
  assert.equal(h.count("workspace_save"), 0);
  assert.deepEqual(h.app.store.workspace, before);
});

test("cancelling Delete model sends no IPC and leaves selection unchanged", async (t) => {
  const h = await harness(t, {
    confirm: false,
    status: status({ loaded: false, model: null }),
  });
  const before = h.calls.length;
  await h.click("delete-model");
  assert.equal(h.confirmations.length, 1);
  assert.equal(h.calls.length, before);
  assert.equal(h.app.state.selected.id, installed.id);
  assert.equal(h.app.state.modelBusy, false);
  assert.equal(h.get("delete-model").disabled, false);
});

test("deletion holds the model lock through rescan, rejects duplicate/load actions and preserves another selection", async (t) => {
  const deleting = deferred();
  const scanning = deferred();
  let scans = 0;
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
    handlers: {
      model_delete: () => deleting.promise,
      discover_models: () =>
        ++scans === 1
          ? { models: [installed, ollama], warnings: [], lastModel: null }
          : scanning.promise,
    },
  });
  h.get("delete-model").click();
  await drain();
  assert.equal(h.app.state.modelBusy, true);
  assert.equal(h.app.state.modelOperation, "delete");
  assert.equal(h.get("delete-model").textContent, "Deleting…");
  for (const id of [
    "delete-model",
    "load-model",
    "unload-model",
    "model-quick-switch",
    "rescan-models",
    "load-gpu",
    "load-context",
  ])
    assert.equal(h.get(id).disabled, true, id);
  h.get("delete-model").emit("click");
  h.get("load-model").emit("click");
  h.get("unload-model").emit("click");
  h.get("rescan-models").emit("click");
  await h.app.loadModel(savedModel);
  h.get("installed-models").children[1].click();
  h.get("model-quick-switch").value = modelIdentity(installed);
  h.get("model-quick-switch").emit("change");
  assert.equal(h.app.state.selected.id, ollama.id);
  h.draft("Editable during deletion");
  assert.equal(h.get("draft").disabled, false);
  deleting.resolve({ retainedFiles: [], warnings: [] });
  await drain();
  assert.equal(
    h.app.state.modelBusy,
    true,
    "rescan must complete before unlocking",
  );
  assert.equal(h.app.state.scanning, true);
  scanning.resolve({ models: [ollama], warnings: [], lastModel: null });
  await drain();
  assert.equal(h.app.state.selected.id, ollama.id);
  assert.equal(h.app.state.modelBusy, false);
  assert.equal(h.app.state.modelOperation, null);
  assert.equal(h.get("delete-model").disabled, false);
  assert.equal(h.app.store.current.draft, "Editable during deletion");
  assert.equal(h.count("model_delete"), 1);
  assert.equal(h.confirmations.length, 1);
  assert.equal(h.count("load_model"), 0);
  assert.equal(h.count("unload_model"), 0);
  assert.equal(h.count("discover_models"), 2);
});

test("deletion errors are actionable, refresh status, unlock and allow explicit retry", async (t) => {
  let fail = true;
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
    handlers: {
      model_delete: () => {
        if (fail)
          throw new Error("Source root changed; rescan before retrying.");
        return { retainedFiles: [], warnings: [] };
      },
      discover_models: () => ({
        models: fail ? [installed] : [],
        warnings: [],
        lastModel: null,
      }),
    },
  });
  const statuses = h.count("desktop_status");
  await h.click("delete-model");
  assert.match(
    h.get("app-error-text").textContent,
    /Could not delete Tiny model.*Source root changed/,
  );
  assert.equal(h.count("desktop_status"), statuses + 1);
  assert.equal(h.app.state.selected.id, installed.id);
  assert.equal(h.app.state.modelBusy, false);
  assert.equal(h.app.state.modelOperation, null);
  assert.equal(h.get("delete-model").disabled, false);
  fail = false;
  await h.click("delete-model");
  assert.equal(h.count("model_delete"), 2);
  assert.equal(h.app.state.selected, null);
  assert.equal(h.get("load-panel").hidden, true);
  assert.equal(h.get("delete-model").hidden, true);
  assert.equal(h.count("load_model"), 0);
});

test("retained deletion files and warnings are reported for camelCase and snake_case results", async (t) => {
  for (const field of ["retainedFiles", "retained_files"]) {
    await t.test(field, async (t) => {
      const h = await harness(t, {
        status: status({ loaded: false, model: null }),
        handlers: {
          model_delete: () => ({
            [field]: ["/external/shared.gguf"],
            warnings: ["Shared with another alias <script>not markup</script>"],
          }),
        },
      });
      await h.click("delete-model");
      assert.match(
        h.get("notifications").textContent,
        /entry removed.*1 shared or external file\(s\) kept/,
      );
      assert.match(
        h.get("notifications").textContent,
        /select their original source entry/,
      );
      assert.match(
        h.get("notifications").textContent,
        /Shared with another alias <script>not markup<\/script>/,
      );
    });
  }
});

test("ANY loaded model blocks deletion of another model until explicitly unloaded", async (t) => {
  const h = await harness(t);
  h.get("installed-models").children[1].click();
  assert.equal(h.app.state.selected.id, ollama.id);
  assert.equal(h.get("delete-model").hidden, false);
  assert.equal(h.get("delete-model").disabled, true);
  assert.match(
    h.get("delete-model-note").textContent,
    /Unload the loaded model first/,
  );
  h.get("delete-model").emit("click");
  await drain();
  assert.equal(h.confirmations.length, 0);
  assert.equal(h.count("model_delete"), 0);
  assert.equal(h.count("unload_model"), 0);
  await h.app.unload();
  assert.equal(h.get("delete-model").disabled, false);
});

test("busy, scanning, loading, updating and unknown status guard deletion even against synthetic clicks", async (t) => {
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
  });
  const cases = [
    { booting: true },
    { modelBusy: true },
    { importing: true },
    { scanning: true },
    { statusKnown: false },
    { status: status({ loaded: false, loading: true }) },
    { status: status({ loaded: false, updating: true }) },
    { active: { conversationId: h.app.store.currentId } },
    { preparing: true },
    { sourcePending: true },
    { update: { state: "Running", log: "", error: null } },
    { storeDownloading: "pending" },
  ];
  for (const patch of cases) {
    const before = Object.fromEntries(
      Object.keys(patch).map((key) => [key, h.app.state[key]]),
    );
    Object.assign(h.app.state, patch);
    h.app.render();
    assert.equal(h.get("delete-model").disabled, true, JSON.stringify(patch));
    h.get("delete-model").emit("click");
    await drain();
    assert.equal(h.count("model_delete"), 0);
    assert.equal(h.confirmations.length, 0);
    Object.assign(h.app.state, before);
  }
  h.app.render();
  assert.equal(h.get("delete-model").disabled, false);
});

test("remembered/manual model paths are not deletable catalog entries", async (t) => {
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
  });
  for (const selected of [
    savedModel,
    { ...installed, id: "/arbitrary/file.gguf" },
    { ...installed, model: "/arbitrary/file.gguf" },
    null,
  ]) {
    h.app.state.selected = selected;
    h.app.render();
    assert.equal(h.get("delete-model").hidden, true);
    assert.equal(h.get("delete-model").disabled, true);
    h.get("delete-model").emit("click");
    await drain();
    assert.equal(h.count("model_delete"), 0);
    assert.equal(h.confirmations.length, 0);
  }
});

test("a native deletion rejection refreshes a newly resident host and keeps deletion disabled", async (t) => {
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
    handlers: {
      model_delete: () => {
        throw new Error("Unload all models first.");
      },
    },
  });
  h.setStatus(status({ model: "another-model.gguf" }));
  await h.click("delete-model");
  assert.equal(h.count("model_delete"), 1);
  assert.equal(h.app.state.modelBusy, false);
  assert.equal(h.get("delete-model").disabled, true);
  assert.match(h.get("app-error-text").textContent, /Unload all models first/);
  assert.equal(h.count("unload_model"), 0);
});

test("late model deletion responses cannot rescan or notify a disposed app", async (t) => {
  const pending = deferred();
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
    handlers: { model_delete: () => pending.promise },
  });
  await h.click("delete-model");
  h.app.dispose();
  const calls = h.calls.length;
  const notices = h.get("notifications").textContent;
  pending.resolve({ retainedFiles: [], warnings: ["Late warning"] });
  await drain();
  assert.equal(h.calls.length, calls);
  assert.equal(h.get("notifications").textContent, notices);
});

test("chat sends whole nonstream responses, keeps next drafts, and exports JSON/Markdown", async (t) => {
  const pending = deferred();
  const h = await harness(t, {
    handlers: { api_request: () => pending.promise },
  });
  h.draft("First question");
  const generation = h.app.send();
  await drain();
  assert.equal(h.posts().length, 1);
  assert.equal(h.posts()[0].args.path, "/v1/chat/completions");
  assert.equal(h.posts()[0].args.body.stream, false);
  assert.deepEqual(h.posts()[0].args.body.messages, [
    { role: "user", content: "First question" },
  ]);
  assert.equal(h.app.store.current.messages.length, 1);
  assert.match(h.get("messages").textContent, /Generating…/);
  assert.equal(h.get("stop").hidden, false);
  h.draft("Next draft");
  pending.resolve(response("Answer with **care**."));
  await generation;
  assert.equal(h.app.store.current.messages.length, 2);
  assert.equal(h.app.store.current.draft, "Next draft");
  assert.equal(h.app.store.current.messages[1].usage.total_tokens, 5);
  assert.match(h.get("messages").textContent, /Answer with care/);
  assert.doesNotMatch(h.get("messages").textContent, /Generating/);
  const json = JSON.parse((await h.export("json")).text);
  assert.equal(json.draft, "Next draft");
  assert.equal(json.messages[1].content, "Answer with **care**.");
  assert.equal(json.error, undefined);
  const markdown = (await h.export("markdown")).text;
  assert.match(markdown, /## You\n\nFirst question/);
  assert.match(markdown, /## Draft\n\nNext draft/);
});

test("stop cancels exactly once and stale responses cannot append to retried or selected chats", async (t) => {
  const first = deferred(),
    second = deferred();
  let n = 0;
  const h = await harness(t, {
    handlers: {
      api_request: () => (++n === 1 ? first.promise : second.promise),
    },
  });
  h.draft("Question");
  const generation = h.app.send();
  const id = h.posts()[0].args.id;
  h.app.stop();
  h.app.stop();
  assert.equal(h.count("cancel_request"), 1);
  assert.deepEqual(h.calls.find((v) => v.command === "cancel_request").args, {
    id,
  });
  assert.equal(h.app.store.current.messages.length, 1);
  assert.match(h.get("request-error-text").textContent, /stopped/);
  const retry = h.app.retry();
  assert.equal(h.posts().length, 2);
  assert.notEqual(h.posts()[1].args.id, id);
  first.resolve(response("STALE"));
  await generation;
  assert.ok(h.app.state.active);
  const originalId = h.app.store.current.id;
  h.app.newChat();
  const newId = h.app.store.current.id;
  second.resolve(response("Fresh"));
  await retry;
  assert.equal(h.app.store.current.id, newId);
  assert.equal(h.app.store.current.messages.length, 0);
  const original = h.app.store.workspace.conversations.find(
    (v) => v.id === originalId,
  );
  assert.deepEqual(
    original.messages.map((m) => m.content),
    ["Question", "Fresh"],
  );
});

test("inference errors remain ephemeral and retry does not duplicate user turns", async (t) => {
  let fail = true;
  const h = await harness(t, {
    handlers: {
      api_request: async () =>
        fail
          ? { status: 400, body: { error: { message: "context exceeded" } } }
          : response("Recovered"),
    },
  });
  h.draft("Question");
  await h.app.send();
  assert.match(h.get("request-error-text").textContent, /context exceeded/);
  assert.equal(h.app.store.current.messages.length, 1);
  await h.app.store.flush();
  assert.doesNotMatch(
    JSON.stringify(
      h.calls.filter((c) => c.command === "workspace_save").at(-1).args
        .workspace,
    ),
    /context exceeded/,
  );
  fail = false;
  await h.app.retry();
  assert.deepEqual(
    h.app.store.current.messages.map((m) => m.role),
    ["user", "assistant"],
  );
  assert.equal(h.get("request-error").hidden, true);
});

test("edit/regenerate create persisted branches and retain the original conversation", async (t) => {
  const original = newConversation({
    title: "Original",
    messages: [
      { role: "user", content: "first", settings: core.DEFAULT_SETTINGS },
      { role: "assistant", content: "old answer" },
    ],
  });
  const h = await harness(t, { workspace: workspaceOf(original) });
  const originalId = h.app.store.current.id;
  const branchButtons = h.get("messages").querySelectorAll("button");
  branchButtons
    .find((n) => n.getAttribute("aria-label") === "Edit as a new branch")
    .click();
  await drain();
  const editId = h.app.store.current.id;
  assert.notEqual(editId, originalId);
  assert.equal(h.get("draft").value, "first");
  assert.deepEqual(h.app.store.current.messages, []);
  h.app.selectConversation(originalId);
  await h.app.branchAt(1, true);
  assert.equal(h.app.store.workspace.conversations.length, 3);
  assert.deepEqual(
    h.app.store.workspace.conversations.find((v) => v.id === originalId),
    original,
  );
  assert.deepEqual(
    h.app.store.current.messages.map((m) => m.content),
    ["first", "Whole response"],
  );
  await h.app.store.flush();
  assert.equal(
    h.calls.filter((v) => v.command === "workspace_save").at(-1).args.workspace
      .conversations.length,
    3,
  );
});

test("conversation selection, rename/search/delete, draft navigation and unload flush are wired to DOM", async (t) => {
  const first = newConversation({ title: "First" }),
    second = newConversation({ title: "Second", draft: "second draft" });
  const h = await harness(t, { workspace: workspaceOf(first, second) });
  h.draft("keep my draft");
  h.get("conversation-list")
    .children[1].querySelector(".conversation-select")
    .click();
  await drain();
  assert.equal(h.get("draft").value, "second draft");
  assert.equal(h.app.store.workspace.conversations[0].draft, "keep my draft");
  await h.click("conversation-options");
  h.get("rename-title").value = "Renamed chat";
  await h.click("rename-conversation");
  assert.equal(h.get("conversation-title").textContent, "Renamed chat");
  h.get("conversation-search").value = "renamed";
  h.get("conversation-search").emit("input");
  assert.equal(h.get("conversation-list").children.length, 1);
  h.draft("flush before close");
  h.window.emit("beforeunload");
  await drain();
  assert.equal(
    h.calls
      .filter((v) => v.command === "workspace_save")
      .at(-1)
      .args.workspace.conversations.find((c) => c.id === second.id).draft,
    "flush before close",
  );
  await h.click("delete-conversation");
  assert.equal(h.app.store.workspace.conversations.length, 1);
  assert.equal(h.app.store.current.id, first.id);
});

test("settings autosave and raw requests intentionally omit chat history, images and system", async (t) => {
  const h = await harness(t, {
    handlers: {
      api_request: async () => ({
        status: 200,
        body: { choices: [{ text: "raw output" }] },
      }),
    },
  });
  await h.click("conversation-options");
  h.get("setting-mode").value = "raw";
  h.get("setting-system").value = "must not enter raw prompt";
  h.get("setting-max-tokens").value = "64";
  h.get("setting-seed").value = "42";
  h.get("settings-form").emit("input");
  assert.equal(h.app.store.current.settings.mode, "raw");
  assert.equal(h.get("raw-note").hidden, false);
  assert.equal(h.get("attach-image").disabled, true);
  h.get("setting-temperature").value = "99";
  h.get("settings-form").emit("input");
  assert.equal(h.app.store.current.settings.temperature, 0.8);
  assert.equal(h.get("settings-error").hidden, false);
  h.get("setting-temperature").value = "0.25";
  h.get("settings-form").emit("input");
  h.draft("exact raw prompt");
  await h.app.send();
  assert.equal(h.posts()[0].args.path, "/v1/completions");
  assert.deepEqual(h.posts()[0].args.body, {
    model: "tiny.gguf",
    max_tokens: 64,
    temperature: 0.25,
    top_p: 0.95,
    seed: 42,
    stream: false,
    prompt: "exact raw prompt",
  });
  assert.equal(h.app.store.current.messages[1].content, "raw output");
  await h.time.fire(400);
  assert.equal(
    h.calls.filter((c) => c.command === "workspace_save").at(-1).args.workspace
      .conversations[0].settings.seed,
    42,
  );
});

test("PNG/JPEG attachments are safe, remain local, and async reads stay with their original chat", async (t) => {
  const h = await harness(t, { deferImages: true });
  const originalId = h.app.store.current.id;
  const attaching = h.app.attach([
    { name: "safe.png", type: "image/png", size: 8 },
  ]);
  assert.equal(h.get("send").disabled, true);
  h.app.newChat();
  const nextId = h.app.store.current.id;
  h.readers[0].onload();
  await attaching;
  assert.equal(h.app.store.current.id, nextId);
  assert.equal(h.app.store.current.attachments.length, 0);
  h.app.selectConversation(originalId);
  assert.equal(h.app.store.current.attachments.length, 1);
  assert.equal(
    h.get("attachments").querySelector("img").src,
    `data:image/png;base64,${png.base64}`,
  );
  await h.app.attach([{ name: "evil.svg", type: "image/svg+xml", size: 100 }]);
  assert.match(h.get("app-error-text").textContent, /PNG or JPEG/);
  assert.equal(h.app.store.current.attachments.length, 1);
  h.draft("what is this?");
  await h.app.send();
  assert.deepEqual(h.posts()[0].args.body.messages[0].images, [png.base64]);
  assert.equal(h.app.store.current.messages[0].images[0].mime, "image/png");
  assert.match((await h.export("markdown")).text, /data:image\/png;base64,/);
});

test("read/save failures are visible in UI and never replace local working data", async (t) => {
  let readBroken = true,
    saveBroken = true;
  const disk = newConversation({ title: "Disk" });
  const h = await harness(t, {
    handlers: {
      workspace_load: async () => {
        if (readBroken) throw new Error("read failure");
        return workspaceOf(disk);
      },
      workspace_save: async () => {
        if (saveBroken) throw new Error("write failure");
      },
    },
  });
  assert.match(h.get("persistence-error-text").textContent, /read failure/);
  h.draft("new local work");
  await h.time.fire(400);
  assert.equal(h.count("workspace_save"), 0);
  readBroken = false;
  await h.click("persistence-retry");
  assert.equal(h.app.store.workspace.conversations.length, 2);
  assert.equal(h.app.store.current.draft, "new local work");
  await h.time.fire(400);
  assert.match(h.get("persistence-error-text").textContent, /write failure/);
  saveBroken = false;
  await h.click("persistence-retry");
  assert.equal(h.get("persistence-error").hidden, true);
});

test("source updates only run after an explicit click and confirmation; progress and errors remain visible", async (t) => {
  const pending = deferred();
  let updateState = {
    state: "Running",
    log: "Compiling local source",
    error: null,
  };
  const h = await harness(t, {
    handlers: {
      source_update: () => pending.promise,
      update_status: async () => clone(updateState),
    },
  });
  await h.click("conversation-options");
  assert.equal(h.count("source_update"), 0);
  await h.click("source-update");
  assert.equal(h.confirmations.length, 1);
  assert.equal(h.count("source_update"), 1);
  await h.time.fire(700);
  assert.equal(h.get("update-log").hidden, false);
  assert.match(h.get("update-log").textContent, /Compiling/);
  assert.equal(h.get("send").disabled, true);
  updateState = {
    state: "Failed",
    log: "Compile failed",
    error: "compiler unavailable",
  };
  pending.resolve({ message: "failed" });
  await drain();
  assert.equal(h.get("update-state").textContent, "Failed");
  assert.match(h.get("update-log").textContent, /compiler unavailable/);
  assert.equal(h.get("source-update").disabled, false);
});

test("declining update confirmation does not invoke source_update", async (t) => {
  const h = await harness(t, { confirm: false });
  await h.click("source-update");
  assert.equal(h.count("source_update"), 0);
});

test("safe renderer treats model HTML, scripts, images and tool instructions strictly as text", async (t) => {
  const h = await harness(t);
  const attack =
    '<script>globalThis.compromised=true</script> <img src=x onerror=alert(1)>\n\n[bad](javascript:alert) [data](data:text/html,x) ![remote](https://evil.invalid/image.png) [ok](https://example.com/)\n\n```html\n<button onclick="evil()">run</button>\n```\n\n| A | B |\n| --- | --- |\n| <svg onload=x> | **bold** |';
  const fragment = core.renderMarkdown(attack);
  h.document.body.append(fragment);
  assert.equal(h.document.querySelectorAll("script").length, 1); // Checked-in module tag only.
  assert.equal(h.document.querySelectorAll("img").length, 0);
  assert.equal(h.document.querySelectorAll("iframe").length, 0);
  assert.match(h.document.body.textContent, /<script>globalThis.compromised/);
  const links = h.document.querySelectorAll("a");
  assert.deepEqual(
    links.map((v) => v.href),
    ["https://example.com/"],
  );
  assert.equal(links[0].rel, "noopener noreferrer");
  const block = h.document.querySelector(".code-block");
  block.querySelectorAll("button")[0].click();
  await drain();
  assert.equal(h.clipboard.at(-1), '<button onclick="evil()">run</button>');
  block.querySelectorAll("button")[1].click();
  const file = h.document.downloads.at(-1);
  assert.equal(file.name, "snippet.html");
  assert.equal(
    await h.blobs.get(file.url).text(),
    '<button onclick="evil()">run</button>',
  );
  assert.equal(globalThis.compromised, undefined);
  assert.equal(core.safeLink("https://user:password@example.com"), null);
  assert.equal(core.safeLink("//example.com"), null);
  assert.equal(core.safeLink("file:///etc/passwd"), null);
  const large = core.renderMarkdown("<svg>".repeat(60000));
  assert.equal(large.children.length, 1);
  assert.equal(large.firstChild.className, "plain-response");
});

test("request helpers validate generation, model options, byte budgets and image signatures", () => {
  for (const input of [
    { temperature: 3 },
    { max_tokens: 0 },
    { top_p: 0 },
    { seed: -1 },
    { mode: "tools" },
  ])
    assert.throws(() =>
      core.validateSettings({ ...core.DEFAULT_SETTINGS, ...input }),
    );
  for (const input of [
    { gpu: "-1" },
    { context: 0 },
    { context: 1.5 },
    { model: "" },
  ])
    assert.throws(() =>
      core.validateModelOptions({
        ...savedModel,
        projector: "",
        ollamaDir: "",
        ...input,
      }),
    );
  assert.throws(() =>
    core.normalizeImage(btoa("<svg onload=evil()>"), "image/png"),
  );
  assert.throws(() => core.normalizeImage(png.base64, "image/jpeg"));
  assert.throws(() =>
    core.buildRequest([], "raw", [png], {
      ...core.DEFAULT_SETTINGS,
      mode: "raw",
    }),
  );
  assert.throws(() =>
    core.buildRequest([], "x".repeat(1048576), [], {
      ...core.DEFAULT_SETTINGS,
      mode: "raw",
    }),
  );
  const request = JSON.parse(
    core.buildRequest([], "hello", [jpeg], {
      ...core.DEFAULT_SETTINGS,
      coding: true,
      system: "private",
    }).json,
  );
  assert.match(request.messages[0].content, /no execution tools/);
  assert.match(request.messages[0].content, /private/);
  assert.deepEqual(request.messages[1].images, [jpeg.base64]);
  assert.doesNotMatch(
    markdownExport(newConversation({ messages: [] })),
    /undefined/,
  );
});

test("workspace editing unlocks before a slow startup model restore completes", async (t) => {
  const loading = deferred();
  const h = await harness(t, {
    workspace: workspaceOf(
      newConversation({
        messages: [
          { role: "user", content: "Saved branch point" },
          { role: "assistant", content: "Saved reply" },
        ],
      }),
    ),
    start: false,
    status: status({ loaded: false, model: null }),
    discovery: { models: [installed], warnings: [], lastModel: savedModel },
    handlers: { load_model: () => loading.promise },
  });
  const opening = h.app.start();
  await drain();
  assert.equal(h.count("load_model"), 1);
  assert.equal(h.get("new-chat").disabled, false);
  assert.equal(h.get("draft").disabled, false);
  const edit = h
    .get("messages")
    .querySelectorAll("button")
    .find((v) => v.getAttribute("aria-label") === "Edit as a new branch");
  assert.equal(edit.disabled, false);
  h.draft("Write while the model is loading");
  assert.equal(h.app.store.current.draft, "Write while the model is loading");
  assert.equal(h.get("send").disabled, true);
  loading.resolve(status({ model: installed.model }));
  await opening;
  assert.equal(h.get("send").disabled, false);
  assert.equal(h.app.store.current.draft, "Write while the model is loading");
});

test("manual policy retains remembered GPU and context for the next quick switch", async (t) => {
  const h = await harness(t, {
    appSettings: { loadPolicy: "manual" },
    discovery: {
      models: [installed, ollama],
      warnings: [],
      lastModel: savedModel,
    },
  });
  assert.equal(h.count("load_model"), 0);
  assert.equal(h.get("load-gpu").value, "7");
  assert.equal(h.get("load-context").value, "512");
  h.get("model-quick-switch").value = modelIdentity(ollama);
  h.get("model-quick-switch").emit("change");
  await drain();
  const options = h.calls.find((v) => v.command === "load_model").args.options;
  assert.equal(options.gpu, "7");
  assert.equal(options.context, 512);
});

test("model loading and discovery failures remain visible and can be retried manually", async (t) => {
  let broken = true;
  const h = await harness(t, {
    handlers: {
      discover_models: async () => {
        if (broken) throw new Error("scan denied");
        return { models: [installed], warnings: [], lastModel: null };
      },
      load_model: async () => {
        if (broken) throw new Error("invalid GGUF");
        return status();
      },
    },
  });
  assert.match(h.get("app-error-text").textContent, /scan denied/);
  await h.app.loadModel(savedModel);
  assert.match(h.get("app-error-text").textContent, /invalid GGUF/);
  assert.equal(h.get("source-update").disabled, false);
  broken = false;
  await h.app.discover();
  await h.app.loadModel(savedModel);
  assert.equal(h.app.state.status.loaded, true);
  assert.equal(h.get("app-error").hidden, true);
});

test("unicode titles branch within the host byte cap and store snapshots are defensive", async () => {
  const c = newConversation({
    title: "字".repeat(85),
    messages: [{ role: "user", content: "question" }],
  });
  const branch = branchConversation(c, 0);
  assert.ok(new TextEncoder().encode(branch.title).length <= 256);
  assert.doesNotThrow(() => normalizeWorkspace(workspaceOf(c, branch)));
  const time = clock();
  const store = createWorkspaceStore({
    ...time,
    invoke: async () => workspaceOf(c),
  });
  await store.hydrate();
  store.current.messages[0].content = "outside mutation";
  store.workspace.conversations.length = 0;
  assert.equal(store.current.messages[0].content, "question");
  assert.equal(store.workspace.conversations.length, 1);
  store.dispose();
});

test("draft typing never decodes, serializes, clones or rebuilds unrelated image history", async (t) => {
  const bytes = Buffer.alloc(128 * 1024);
  Buffer.from(png.base64, "base64").copy(bytes);
  const image = core.normalizeImage(
    bytes.toString("base64"),
    "image/png",
    "large.png",
  );
  const first = newConversation({
    title: "Image thread",
    attachments: [image],
    messages: [
      { role: "user", content: "Analyze", images: [image] },
      { role: "assistant", content: "**Existing** answer" },
    ],
  });
  const other = newConversation({
    messages: [{ role: "user", content: "Unrelated image", images: [image] }],
  });
  const h = await harness(t, { workspace: workspaceOf(first, other) });
  const article = h.get("messages").firstChild;
  const thumbnail = h.get("attachments").firstChild;
  const sidebar = h.get("conversation-list").firstChild;
  const model = h.get("installed-models").firstChild;
  const revision = h.app.store.contentRevision;
  const decoder = t.mock.method(globalThis, "atob");
  const stringify = t.mock.method(JSON, "stringify");
  const snapshot = t.mock.getter(h.app.store, "current");
  const workspaceSnapshot = t.mock.getter(h.app.store, "workspace");
  const messages = t.mock.method(h.get("messages"), "replaceChildren");
  const attachments = t.mock.method(h.get("attachments"), "replaceChildren");
  for (let n = 1; n <= 40; n++) h.draft("Typing… ".repeat(n));
  assert.equal(decoder.mock.callCount(), 0);
  assert.equal(stringify.mock.callCount(), 0);
  assert.equal(snapshot.mock.callCount(), 0);
  assert.equal(workspaceSnapshot.mock.callCount(), 0);
  assert.equal(h.app.store.contentRevision, revision);
  assert.equal(h.get("send").disabled, false);
  assert.equal(h.get("messages").firstChild, article);
  assert.equal(h.get("attachments").firstChild, thumbnail);
  assert.equal(h.get("conversation-list").firstChild, sidebar);
  assert.equal(h.get("installed-models").firstChild, model);
  await h.time.fire(400);
  assert.ok(
    decoder.mock.callCount() > 0,
    "complete image validation still runs at the save boundary",
  );
  assert.equal(h.count("workspace_save"), 1);
  assert.equal(
    messages.mock.callCount(),
    0,
    "autosave notifications must not rebuild history",
  );
  assert.equal(attachments.mock.callCount(), 0);
  assert.equal(snapshot.mock.callCount(), 0);
  assert.equal(workspaceSnapshot.mock.callCount(), 0);
  const calls = stringify.mock.callCount();
  h.app.render(); // Same content version also reuses its defensive render snapshot.
  assert.equal(stringify.mock.callCount(), calls);
  assert.equal(snapshot.mock.callCount(), 0);
  assert.equal(messages.mock.callCount(), 0);
  const saved = h.calls.find((v) => v.command === "workspace_save").args
    .workspace;
  assert.equal(saved.conversations[0].draft, "Typing… ".repeat(40));
  assert.deepEqual(saved.conversations[0].messages, first.messages);
});

test("draft-only updates serialize pending saves, enforce types and keep all public snapshots defensive", async () => {
  const c = newConversation({
    attachments: [png],
    messages: [{ role: "user", content: "history", images: [jpeg] }],
  });
  const time = clock(),
    writes = [];
  const store = createWorkspaceStore({
    ...time,
    invoke(command, args) {
      if (command === "workspace_load") return Promise.resolve(workspaceOf(c));
      const pending = deferred();
      writes.push({ ...pending, workspace: args.workspace });
      return pending.promise;
    },
  });
  await store.hydrate();
  const contentRevision = store.contentRevision;
  store.updateDraft(c.id, "first");
  const saving = store.flush();
  store.updateDraft(c.id, "latest");
  assert.equal(store.flush(), saving);
  assert.equal(writes.length, 1);
  assert.equal(writes[0].workspace.conversations[0].draft, "first");
  assert.equal(store.contentRevision, contentRevision);
  assert.throws(
    () => store.updateDraft(c.id, { toString: () => "unsafe" }),
    /text/,
  );
  assert.throws(() => store.updateDraft("missing", "no"), /not found/);
  store.composer.settings.system = "outside mutation";
  store.current.attachments[0].base64 = "outside mutation";
  store.workspace.conversations[0].messages[0].content = "outside mutation";
  store.listConversations()[0].title = "outside mutation";
  assert.equal(store.composer.settings.system, "");
  assert.equal(store.current.attachments[0].base64, png.base64);
  assert.equal(store.current.messages[0].content, "history");
  assert.equal(store.composer.title, c.title);
  assert.equal(store.composer.draft, "latest");
  writes[0].resolve(null);
  await drain();
  assert.equal(writes.length, 2);
  assert.equal(writes[1].workspace.conversations[0].draft, "latest");
  assert.deepEqual(writes[1].workspace.conversations[0].messages, c.messages);
  writes[1].resolve(null);
  assert.equal(await saving, true);
  assert.equal(store.persistence.dirty, false);
  store.dispose();
});

test("oversized Unicode drafts remain recoverable across navigation/export and block invalid saves atomically", async (t) => {
  const first = newConversation({
    title: "Large draft",
    messages: [{ role: "user", content: "saved history", images: [png] }],
  });
  const other = newConversation({ title: "Other chat" });
  const h = await harness(t, { workspace: workspaceOf(first, other) });
  const oversized = "字".repeat(Math.floor(CAPS.text / 3) + 1);
  h.draft(oversized);
  assert.equal(h.app.store.current.draft, oversized);
  assert.equal(h.get("draft").value, oversized);
  assert.equal(h.get("persistence-error").hidden, false);
  assert.match(
    h.get("persistence-error-text").textContent,
    /1 MiB.*Shorten.*export/,
  );
  assert.equal(h.get("send").disabled, true);
  await assert.rejects(h.app.send(), /1 MiB/);
  await h.time.fire(400);
  assert.equal(h.count("workspace_save"), 0);
  assert.equal(h.posts().length, 0);
  assert.equal(h.app.store.persistence.dirty, true);
  assert.throws(() => normalizeWorkspace(h.app.store.workspace), /limit/);
  h.app.selectConversation(other.id);
  h.draft("Unrelated working draft");
  h.app.store.rename(other.id, "Still editable");
  await drain();
  assert.equal(h.app.store.current.title, "Still editable");
  assert.equal(h.app.store.current.draft, "Unrelated working draft");
  h.app.selectConversation(first.id);
  await drain();
  assert.equal(h.get("draft").value, oversized);
  const exported = JSON.parse((await h.export("json")).text);
  assert.equal(exported.draft, oversized);
  assert.deepEqual(exported.messages, first.messages);
  assert.equal(exported.draftError, undefined);
  assert.match((await h.export("markdown")).text, /## Draft/);
  h.draft("Shortened safely");
  assert.equal(await h.app.store.flush(), true);
  assert.equal(h.get("persistence-error").hidden, true);
  const saved = h.calls.filter((v) => v.command === "workspace_save").at(-1)
    .args.workspace;
  assert.equal(saved.conversations[0].draft, "Shortened safely");
  assert.equal(saved.conversations[1].draft, "Unrelated working draft");
  assert.deepEqual(saved.conversations[0].messages, first.messages);
  assert.doesNotThrow(() => normalizeWorkspace(saved));
});

test("an oversized draft entered during a pending save cannot leak to IPC or overwrite its acknowledged snapshot", async () => {
  const c = newConversation({ draft: "on disk" });
  const time = clock(),
    writes = [],
    pending = deferred();
  const store = createWorkspaceStore({
    ...time,
    invoke(command, args) {
      if (command === "workspace_load") return Promise.resolve(workspaceOf(c));
      writes.push(args.workspace);
      return writes.length === 1 ? pending.promise : Promise.resolve(null);
    },
  });
  await store.hydrate();
  store.updateDraft(c.id, "valid first edit");
  const saving = store.flush();
  const oversized = "x".repeat(CAPS.text + 1);
  store.updateDraft(c.id, oversized);
  pending.resolve(null);
  assert.equal(await saving, false);
  assert.equal(writes.length, 1);
  assert.equal(writes[0].conversations[0].draft, "valid first edit");
  assert.equal(store.composer.draft, oversized);
  store.add(newConversation({ title: "Unrelated work" }));
  assert.equal(store.workspace.conversations.length, 2);
  assert.equal(
    store.workspace.conversations.find((v) => v.id === c.id).draft,
    oversized,
  );
  store.updateDraft(c.id, "recovered");
  assert.equal(await store.flush(), true);
  assert.equal(
    writes.at(-1).conversations.find((v) => v.id === c.id).draft,
    "recovered",
  );
  store.dispose();
});

test("removing an oversized-draft conversation clears its local error and saves the remaining work", async () => {
  const first = newConversation({ title: "Oversized" });
  const other = newConversation({ title: "Keep", draft: "Retained work" });
  const time = clock(),
    writes = [];
  const store = createWorkspaceStore({
    ...time,
    invoke: async (command, args) => {
      if (command === "workspace_load") return workspaceOf(first, other);
      writes.push(args.workspace);
    },
  });
  await store.hydrate();
  store.updateDraft(first.id, "x".repeat(CAPS.text + 1));
  assert.equal(await store.flush(), false);
  assert.equal(writes.length, 0);
  store.remove(first.id);
  await drain();
  assert.equal(store.currentId, other.id);
  assert.equal(store.composer.draft, "Retained work");
  assert.equal(store.persistence.draftError, "");
  assert.equal(store.persistence.saveError, "");
  assert.equal(writes.length, 1);
  assert.equal(writes[0].conversations.length, 1);
  assert.equal(writes[0].conversations[0].draft, "Retained work");
  store.dispose();
});

test("native operations lock inference/model changes, not unrelated conversation editing", async (t) => {
  for (const command of [
    "api_request",
    "load_model",
    "unload_model",
    "source_update",
  ]) {
    await t.test(command, async (t) => {
      const pending = deferred();
      const original = newConversation({
        title: "Native work",
        draft: "Question",
      });
      const other = newConversation({
        title: "Editable chat",
        messages: [
          {
            role: "user",
            content: "Branch point",
            settings: core.DEFAULT_SETTINGS,
          },
          { role: "assistant", content: "Saved answer" },
        ],
      });
      const h = await harness(t, {
        workspace: workspaceOf(original, other),
        handlers: { [command]: () => pending.promise },
      });
      const operation =
        command === "api_request"
          ? h.app.send()
          : command === "load_model"
            ? h.app.loadModel(savedModel)
            : command === "unload_model"
              ? h.app.unload()
              : h.app.sourceUpdate();
      await drain();
      assert.equal(h.count(command), 1);
      h.app.selectConversation(other.id);
      assert.equal(h.get("draft").disabled, false);
      h.draft("Draft during native work");
      h.app.openInspector(true);
      h.get("setting-seed").value = "17";
      h.get("settings-form").emit("input");
      h.get("rename-title").value = "Edited during native work";
      await h.click("rename-conversation");
      assert.equal(h.app.store.current.draft, "Draft during native work");
      assert.equal(h.app.store.current.settings.seed, 17);
      assert.equal(h.app.store.current.title, "Edited during native work");
      const branches = h.get("messages").querySelectorAll("button");
      const edit = branches.find(
        (v) => v.getAttribute("aria-label") === "Edit as a new branch",
      );
      const regenerate = branches.find(
        (v) => v.getAttribute("aria-label") === "Regenerate as a new branch",
      );
      assert.equal(edit.disabled, false);
      assert.equal(regenerate.disabled, true);
      edit.click();
      await drain();
      assert.notEqual(h.app.store.current.id, other.id);
      assert.equal(h.app.store.current.draft, "Branch point");
      assert.equal(
        h.app.store.workspace.conversations.find((v) => v.id === other.id)
          .draft,
        "Draft during native work",
      );
      h.app.newChat();
      const newId = h.app.store.current.id;
      h.draft("New chat remains editable");
      assert.equal(h.get("draft").disabled, false);
      assert.equal(h.get("send").disabled, true);
      assert.equal(h.get("load-model").disabled, true);
      h.app.selectConversation(other.id);
      assert.equal(h.get("delete-conversation").disabled, false);
      await h.click("delete-conversation");
      assert.ok(
        !h.app.store.workspace.conversations.some((v) => v.id === other.id),
      );
      h.app.selectConversation(newId);
      pending.resolve(
        command === "api_request"
          ? response("Native reply")
          : command === "unload_model"
            ? status({ loaded: false, model: null })
            : command === "source_update"
              ? { message: "Done" }
              : status({ model: installed.model }),
      );
      await operation;
      assert.equal(h.app.store.current.id, newId);
      assert.equal(h.app.store.current.draft, "New chat remains editable");
      assert.equal(h.count(command), 1);
    });
  }
});

test("redesigned shell is quiet and keeps model selection in the composer", async (t) => {
  const h = await harness(t);
  assert.doesNotMatch(
    shell,
    /wordmark|brand-slash|On this device|Installed on this device|No messages yet|Enter to send/,
  );
  assert.match(shell, /What do you want to work on today\?/);
  assert.equal(h.get("welcome-view").hidden, false);
  assert.equal(h.get("view-label").textContent, "");
  assert.equal(h.get("composer-hint").textContent, "");
  assert.equal(h.get("model-state").textContent, "");
  let parent = h.get("model-quick-switch").parent;
  while (parent && parent.getAttribute("id") !== "composer-form")
    parent = parent.parent;
  assert.ok(parent, "model selection belongs to the composer");
  assert.equal(h.get("history-search-field").hidden, true);
  await h.click("history-search-toggle");
  assert.equal(h.get("history-search-field").hidden, false);
  assert.equal(h.document.activeElement, h.get("conversation-search"));
  h.get("conversation-search").value = "query";
  h.get("conversation-search").emit("keydown", { key: "Escape" });
  assert.equal(h.get("history-search-field").hidden, true);
  assert.equal(h.get("conversation-search").value, "");
});

test("composer model picker keeps a compact label in sync with its native select", async (t) => {
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
    discovery: { models: [installed, ollama], warnings: [], lastModel: null },
  });
  const select = h.get("model-quick-switch");
  const label = h.get("model-quick-switch-label");
  assert.equal(label.textContent, "Select model");
  assert.equal(label.getAttribute("aria-hidden"), "true");
  assert.equal(select.getAttribute("aria-label"), "Active model");
  assert.equal(select.title, "Select model");
  select.value = modelIdentity(ollama);
  select.emit("change");
  await drain();
  assert.equal(label.textContent, ollama.name);
  assert.equal(select.title, `${ollama.name} · Ollama`);
  assert.equal(select.value, modelIdentity(ollama));
  await h.click("unload-model");
  assert.equal(label.textContent, "Select model");
  assert.equal(select.value, "");
});

test("composer model picker label uses safe text and a loaded name absent from discovery", async (t) => {
  const name = "<img src=x onerror=alert(1)>";
  const h = await harness(t, {
    status: status({ model: name }),
    discovery: { models: [], warnings: [], lastModel: null },
  });
  assert.equal(h.get("model-quick-switch-label").textContent, name);
  assert.equal(h.get("model-quick-switch-label").children.length, 0);
});

test("composer picker hides native chrome and sizes from its label at every breakpoint", async () => {
  const css = await readFile(
    new URL("../desktop/ui/style.css", import.meta.url),
    "utf8",
  );
  assert.match(
    css,
    /#model-quick-switch\s*\{[^}]*position: absolute;[^}]*opacity: 0;/,
  );
  assert.match(
    css,
    /#model-quick-switch-label\s*\{[^}]*white-space: nowrap;[^}]*text-overflow: ellipsis;/,
  );
  assert.match(css, /\.composer-model-picker:focus-within\s*\{[^}]*outline:/);
  assert.match(css, /\.model-picker-chevron\s*\{[^}]*flex-shrink: 0;/);
  assert.doesNotMatch(css, /\.composer-model-picker\s*\{[^}]*flex-basis: 100%/);
});

test("all installed sources are selectable without importing or downloading", async (t) => {
  const lmStudio = {
    ...installed,
    id: "studio",
    name: "studio-model",
    model: "/studio/model.gguf",
    source: "lm-studio",
  };
  const h = await harness(t, {
    discovery: {
      models: [installed, ollama, lmStudio],
      warnings: [],
      lastModel: null,
    },
  });
  assert.match(h.get("installed-models").textContent, /LM Studio/);
  assert.match(h.get("model-quick-switch").textContent, /Ollama/);
  assert.equal(h.count("store_search"), 0);
  assert.equal(h.count("store_download"), 0);
  assert.equal(h.count("model_import"), 0);
  h.get("model-quick-switch").value = modelIdentity(lmStudio);
  h.get("model-quick-switch").emit("change");
  await drain();
  assert.equal(
    h.calls.find((call) => call.command === "load_model").args.options.model,
    lmStudio.model,
  );
  assert.equal(h.count("store_download"), 0);
});

test("discovery refreshes every thirty seconds, skips hidden windows, and is disposed", async (t) => {
  const h = await harness(t);
  assert.equal(h.count("discover_models"), 1);
  await h.time.fire(30000);
  assert.equal(h.count("discover_models"), 2);
  h.document.hidden = true;
  await h.time.fire(30000);
  assert.equal(h.count("discover_models"), 2);
  h.document.hidden = false;
  await h.time.fire(30000);
  assert.equal(h.count("discover_models"), 3);
  h.app.dispose();
  await h.time.fire(30000);
  assert.equal(h.count("discover_models"), 3);
});

test("startup warnings are safe deduplicated animated notifications, never library banners", async (t) => {
  const warning = "Duplicate model <img src=x onerror=alert(1)>";
  const h = await harness(t, {
    discovery: { models: [installed], warnings: [warning], lastModel: null },
  });
  assert.equal(h.get("discovery-warning").hidden, true);
  assert.equal(h.get("notifications").children.length, 1);
  assert.match(h.get("notifications").textContent, /<img src=x/);
  const toast = h.get("notifications").children[0];
  assert.equal(toast.classList.contains("entering"), true);
  await h.app.discover();
  assert.equal(h.get("notifications").children.length, 1);
  await h.time.fire(180);
  assert.equal(toast.classList.contains("entering"), false);
  await h.time.fire(6500);
  assert.equal(toast.classList.contains("leaving"), true);
  await h.time.fire(180);
  assert.equal(h.get("notifications").children.length, 0);
  await h.app.discover();
  assert.equal(h.get("notifications").children.length, 0);
});

test("application settings are separate, progressive, persisted, and do not reload the current model", async (t) => {
  const h = await harness(t);
  await h.click("nav-settings");
  assert.equal(h.get("settings-view").hidden, false);
  assert.equal(h.get("inspector").hidden, true);
  assert.equal(h.get("app-manual-load-fields").hidden, true);
  assert.equal(h.get("model-advanced").hidden, true);
  h.get("app-load-policy").value = "manual";
  h.get("app-settings-form").emit("change");
  assert.equal(h.get("app-manual-load-fields").hidden, false);
  h.get("app-gpu").value = "cpu";
  h.get("app-context").value = "2048";
  h.get("app-auto-restore").checked = false;
  h.get("app-model-dirs").value = "/extra/models\n/extra/other";
  h.get("app-lm-studio-dirs").value = "/custom/studio";
  h.get("app-settings-form").emit("submit");
  await drain();
  const settings = h.calls.find((call) => call.command === "app_settings_save")
    .args.settings;
  assert.equal(settings.loadPolicy, "manual");
  assert.equal(settings.context, 2048);
  assert.equal(settings.autoRestore, false);
  assert.deepEqual(settings.modelDirectories, [
    "/extra/models",
    "/extra/other",
  ]);
  assert.deepEqual(settings.lmStudioDirectories, ["/custom/studio"]);
  assert.equal(h.count("load_model"), 0);
  assert.equal(h.get("model-advanced").hidden, false);
  assert.equal(h.count("discover_models"), 2);
  h.get("app-load-policy").value = "defaults";
  h.get("app-settings-form").emit("change");
  assert.equal(h.get("app-defaults-row").hidden, false);
  assert.equal(h.get("app-manual-load-fields").hidden, true);
  h.get("app-defaults-path").value = "/config/defaults.json";
  await h.app.saveAppSettings();
  assert.equal(h.app.state.appSettings.defaultsPath, "/config/defaults.json");
});

test("disabled restore and failed settings reads never cause implicit model loads or settings overwrites", async (t) => {
  for (const failed of [false, true])
    await t.test(String(failed), async (t) => {
      const h = await harness(t, {
        status: status({ loaded: false, model: null }),
        appSettings: { autoRestore: false },
        discovery: { models: [installed], warnings: [], lastModel: savedModel },
        handlers: failed
          ? {
              app_settings_load: async () => {
                throw new Error("corrupt settings");
              },
            }
          : {},
      });
      assert.equal(h.count("load_model"), 0);
      if (failed) {
        assert.equal(h.get("save-app-settings").disabled, true);
        await h.app.saveAppSettings();
        assert.equal(h.count("app_settings_save"), 0);
        assert.match(h.get("notifications").textContent, /corrupt settings/);
      }
    });
});

test("automatic loading ignores hidden manual fields and one visible button toggles load/unload", async (t) => {
  const h = await harness(t, {
    status: status({ loaded: false, model: null }),
  });
  h.get("load-gpu").value = "bad hidden value";
  h.get("load-context").value = "bad hidden value";
  assert.equal(h.get("load-model").textContent, "Load model");
  await h.click("load-model");
  const options = h.calls.find((call) => call.command === "load_model").args
    .options;
  assert.equal(options.gpu, "auto");
  assert.equal(options.context, 4096);
  assert.equal(h.get("load-model").textContent, "Unload model");
  assert.equal(h.get("unload-model").hidden, true);
  await h.click("load-model");
  assert.equal(h.count("unload_model"), 1);
  assert.equal(h.get("load-model").textContent, "Load model");
});

test("the model store searches only on request, reuses installed models, and refreshes after download", async (t) => {
  const downloaded = {
    id: "owner/new",
    source: "lm-studio",
    name: "owner/new",
    exact: true,
    installed: false,
    totalBytes: 1024,
    files: [{ name: "new.gguf", bytes: 1024 }],
  };
  const present = {
    ...downloaded,
    id: "coder:latest",
    source: "ollama",
    name: "coder:latest",
    installed: true,
  };
  const h = await harness(t, {
    storeResults: { models: [present, downloaded], warnings: [] },
  });
  assert.equal(h.count("store_search"), 0);
  await h.click("nav-models");
  await h.click("models-store-tab");
  assert.equal(h.get("store-models-panel").hidden, false);
  assert.equal(h.count("store_search"), 0);
  h.get("store-query").value = "model";
  h.get("store-search-form").emit("submit");
  await drain();
  assert.equal(h.count("store_search"), 1);
  const cards = h.get("store-results").children;
  assert.equal(cards[0].querySelector("button").textContent, "Installed");
  assert.equal(cards[0].querySelector("button").disabled, true);
  cards[1].querySelector("button").click();
  await drain();
  assert.equal(h.count("store_download"), 1);
  assert.equal(h.count("discover_models"), 2);
  assert.equal(
    h.get("store-results").children[1].querySelector("button").textContent,
    "Installed",
  );
  assert.deepEqual(
    h.calls.find((call) => call.command === "store_download").args,
    { id: downloaded.id, source: "lm-studio", file: "new.gguf" },
  );
});

test("store variants require a file and never redownload an installed quantization", async (t) => {
  const model = {
    id: "owner/variants",
    source: "lm-studio",
    name: "owner/variants",
    exact: true,
    installed: true,
    files: [
      { name: "one.gguf", bytes: 10, installed: true },
      { name: "two.gguf", bytes: 20, installed: false },
    ],
  };
  const h = await harness(t, {
    storeResults: { models: [model], warnings: [] },
  });
  await h.app.searchStore("owner/variants", "lm-studio");
  const card = h.get("store-results").children[0],
    select = card.querySelector("select"),
    download = card.querySelector("button");
  assert.equal(download.disabled, true);
  select.value = "one.gguf";
  select.emit("change");
  assert.equal(download.textContent, "Installed");
  assert.equal(download.disabled, true);
  select.value = "two.gguf";
  select.emit("change");
  assert.equal(download.disabled, false);
  download.click();
  await drain();
  assert.equal(
    h.calls.find((call) => call.command === "store_download").args.file,
    "two.gguf",
  );
});

test("single-file store results require that exact file, not another installed quantization", async (t) => {
  const model = {
    id: "owner/repo",
    source: "lm-studio",
    name: "owner/repo",
    exact: true,
    installed: true,
    files: [{ name: "new-Q8.gguf", bytes: 20, installed: false }],
  };
  const h = await harness(t, {
    storeResults: { models: [model], warnings: [] },
  });
  await h.app.searchStore("owner/repo", "lm-studio");
  const download = h.get("store-results").children[0].querySelector("button");
  assert.equal(download.textContent, "Download");
  assert.equal(download.disabled, false);
  download.click();
  await drain();
  assert.equal(
    h.calls.find((call) => call.command === "store_download").args.file,
    "new-Q8.gguf",
  );
  assert.equal(
    h.get("store-results").children[0].querySelector("button").textContent,
    "Installed",
  );
});

test("store search results and errors cannot inject markup or update a disposed app", async (t) => {
  const pending = deferred();
  const h = await harness(t, {
    handlers: { store_search: () => pending.promise },
  });
  const search = h.app.searchStore("repo", "all");
  h.app.dispose();
  pending.resolve({
    models: [
      {
        id: "<img>",
        name: "<script>",
        source: "ollama",
        exact: true,
        files: [],
      },
    ],
    warnings: [],
  });
  await search;
  assert.equal(h.get("store-results").children.length, 0);
  assert.equal(h.get("notifications").children.length, 0);
});

test("Import model registers existing weights and refreshes selection without loading or downloading", async (t) => {
  const imported = {
    ...installed,
    id: "new",
    name: "new:latest",
    model: "/existing/new.gguf",
  };
  let scanned = 0;
  const h = await harness(t, {
    handlers: {
      discover_models: async () => ({
        models: ++scanned === 1 ? [installed] : [installed, imported],
        warnings: [],
        lastModel: null,
      }),
      model_import: async () => ({
        model: imported.name,
        path: imported.model,
      }),
    },
  });
  await h.click("toggle-manual-model");
  assert.equal(h.get("manual-import").hidden, false);
  h.get("manual-source").value = "lma";
  h.get("manual-model").value = imported.model;
  h.get("manual-name").value = "my-model:latest";
  h.get("manual-import").emit("submit");
  await drain();
  assert.equal(h.count("model_import"), 1);
  assert.equal(
    h.calls.find((call) => call.command === "model_import").args.name,
    "my-model:latest",
  );
  assert.equal(h.count("load_model"), 0);
  assert.equal(h.count("store_download"), 0);
  assert.equal(h.app.state.selected.name, imported.name);
  assert.equal(h.get("manual-import").hidden, true);
  assert.equal(h.app.state.importing, false);
});

test("application preferences validate limits and preserve only supported configuration", () => {
  assert.equal(
    validateAppSettings(DEFAULT_APP_SETTINGS).loadPolicy,
    "automatic",
  );
  for (const change of [
    { loadPolicy: "magic" },
    { gpu: "-1" },
    { context: 1 },
    { autoRestore: "yes" },
    { defaultsPath: "a\u0000b" },
    { modelDirectories: Array.from({ length: 65 }, (_, i) => `/models/${i}`) },
    {
      modelDirectories: Array.from({ length: 33 }, (_, i) => `/models/${i}`),
      lmStudioDirectories: Array.from({ length: 32 }, (_, i) => `/studio/${i}`),
    },
  ])
    assert.throws(() =>
      validateAppSettings({ ...DEFAULT_APP_SETTINGS, ...change }),
    );
  assert.equal(
    validateAppSettings({ ...DEFAULT_APP_SETTINGS, secret: "never" }).secret,
    undefined,
  );
});

function observeMessageEntries(t, h) {
  const entries = [];
  const container = h.get("messages");
  const append = container.append;
  t.mock.method(container, "append", function (...nodes) {
    for (const node of nodes) {
      const add = node.classList.add;
      t.mock.method(node.classList, "add", function (name) {
        if (name === "message-enter" && !node.classList.contains(name)) {
          entries.push({
            index: Number(node.dataset.messageIndex),
            content: node.querySelector(".message-body")?.textContent,
          });
        }
        return add.call(this, name);
      });
    }
    return append.apply(this, nodes);
  });
  return entries;
}

test("motion: initial hydrated history never receives message entry markers", async (t) => {
  const saved = newConversation({
    messages: [
      { role: "user", content: "Saved question" },
      { role: "assistant", content: "Saved answer" },
    ],
  });
  const h = await harness(t, {
    start: false,
    workspace: workspaceOf(saved),
  });
  const entries = observeMessageEntries(t, h);
  await h.app.start();
  assert.equal(h.get("messages").querySelectorAll(".message").length, 2);
  assert.match(h.get("messages").textContent, /Saved question/);
  assert.match(h.get("messages").textContent, /Saved answer/);
  assert.deepEqual(h.get("messages").querySelectorAll(".message-enter"), []);
  assert.deepEqual(
    entries,
    [],
    "hydration must not animate even a transient render",
  );
});

test("motion: send marks only the appended user and final assistant, without draft rerenders", async (t) => {
  const pending = deferred();
  const saved = newConversation({
    messages: [
      { role: "user", content: "Earlier question" },
      { role: "assistant", content: "Earlier answer" },
    ],
  });
  const h = await harness(t, {
    workspace: workspaceOf(saved),
    handlers: { api_request: () => pending.promise },
  });
  const entries = observeMessageEntries(t, h);
  h.draft("New question");
  const generation = h.app.send();
  await drain();
  assert.deepEqual(entries, [{ index: 2, content: "New question" }]);
  assert.deepEqual(
    h
      .get("messages")
      .querySelectorAll(".message-enter")
      .map((n) => n.dataset.messageIndex),
    ["2"],
    "locking for generation must not lose the new user entry or mark earlier history",
  );
  assert.match(h.get("messages").textContent, /Generating…/);
  const articles = h.get("messages").querySelectorAll(".message");
  const replacements = t.mock.method(h.get("messages"), "replaceChildren");
  h.draft("Next draft");
  assert.equal(replacements.mock.callCount(), 0);
  assert.deepEqual(h.get("messages").querySelectorAll(".message"), articles);
  assert.deepEqual(entries, [{ index: 2, content: "New question" }]);
  pending.resolve(response("New answer"));
  await generation;
  assert.deepEqual(
    entries,
    [
      { index: 2, content: "New question" },
      { index: 3, content: "New answer" },
    ],
    "each new turn must enter once, not replay during request state changes",
  );
  assert.deepEqual(
    h
      .get("messages")
      .querySelectorAll(".message-enter")
      .map((n) => n.dataset.messageIndex),
    ["3"],
    "the final response enters without replaying the user or hydrated history",
  );
  assert.equal(h.app.store.current.draft, "Next draft");
  assert.doesNotMatch(h.get("messages").textContent, /Generating/);
});

test("motion: unrelated updates and model lock rerenders clear entries without replaying history", async (t) => {
  const loading = deferred();
  const h = await harness(t, {
    handlers: { load_model: () => loading.promise },
  });
  h.draft("Question");
  await h.app.send();
  assert.equal(h.get("messages").querySelectorAll(".message-enter").length, 1);
  const entries = observeMessageEntries(t, h);
  const assertQuiet = () => {
    assert.equal(h.get("messages").querySelectorAll(".message").length, 2);
    assert.deepEqual(h.get("messages").querySelectorAll(".message-enter"), []);
    assert.deepEqual(
      entries,
      [],
      "existing messages must not re-enter on any rebuild",
    );
  };
  h.app.store.update(h.app.store.currentId, (c) => {
    c.title = "Renamed without a new turn";
  });
  assertQuiet();
  h.app.openInspector(true);
  h.get("setting-seed").value = "17";
  h.get("settings-form").emit("input");
  assert.equal(h.app.store.current.settings.seed, 17);
  assertQuiet();
  const operation = h.app.loadModel(savedModel);
  await drain();
  assert.equal(h.count("load_model"), 1);
  assert.equal(h.get("send").disabled, true);
  assertQuiet();
  loading.resolve(status({ model: installed.model }));
  await operation;
  assertQuiet();
  h.app.render();
  await h.app.refreshStatus();
  assertQuiet();
});

test("motion: switching to longer saved history and back never replays message entries", async (t) => {
  const first = newConversation({
    messages: [
      { role: "user", content: "First saved question" },
      { role: "assistant", content: "First saved answer" },
    ],
  });
  const other = newConversation({
    messages: Array.from({ length: 6 }, (_, index) => ({
      role: index % 2 ? "assistant" : "user",
      content: `Other saved turn ${index}`,
    })),
  });
  const h = await harness(t, { workspace: workspaceOf(first, other) });
  h.draft("Newest question");
  await h.app.send();
  assert.equal(h.get("messages").querySelectorAll(".message-enter").length, 1);
  const entries = observeMessageEntries(t, h);
  h.app.selectConversation(other.id);
  assert.equal(h.get("messages").querySelectorAll(".message").length, 6);
  assert.match(h.get("messages").textContent, /Other saved turn 5/);
  assert.deepEqual(h.get("messages").querySelectorAll(".message-enter"), []);
  h.app.selectConversation(first.id);
  assert.equal(h.get("messages").querySelectorAll(".message").length, 4);
  assert.match(h.get("messages").textContent, /Newest question/);
  assert.deepEqual(h.get("messages").querySelectorAll(".message-enter"), []);
  h.app.render();
  assert.deepEqual(
    entries,
    [],
    "navigation must not treat either history as appended turns",
  );
});

test("motion: CSS keeps global reduced-motion overrides and never animates every message", async () => {
  const css = (
    await readFile(new URL("../desktop/ui/style.css", import.meta.url), "utf8")
  ).replace(/\/\*[\s\S]*?\*\//g, "");
  assert.match(
    css,
    /@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)\s*\{\s*\*\s*,\s*\*::before\s*,\s*\*::after\s*\{[^}]*animation\s*:\s*none\s*!important\s*;[^}]*transition\s*:\s*none\s*!important\s*;/,
  );
  for (const [, selectors, declarations] of css.matchAll(
    /([^{}]+)\{([^{}]*)\}/g,
  )) {
    if (
      !selectors.split(",").some((selector) => selector.trim() === ".message")
    )
      continue;
    for (const [, value] of declarations.matchAll(
      /\banimation(?:-name)?\s*:\s*([^;}]*)/g,
    )) {
      assert.match(
        value.trim(),
        /^none(?:\s*!important)?$/,
        "entry animation must be opt-in, not on .message",
      );
    }
  }
});
