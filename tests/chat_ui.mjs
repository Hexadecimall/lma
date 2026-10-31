// Run with: node --test tests/chat_ui.mjs
// No browser, server, npm packages, or temporary source files are required.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { setImmediate as nextTurn } from "node:timers/promises";
import test from "node:test";

const source = readFileSync(
  new URL("../src/runtime/ui/chat.js", import.meta.url),
  "utf8",
);
const shell = readFileSync(
  new URL("../src/runtime/ui/index.html", import.meta.url),
  "utf8",
);
const moduleURL = `data:text/javascript;base64,${Buffer.from(source).toString("base64")}`;

// Import the real exports with startup disabled, even if a runner supplies a DOM.
const documentDescriptor = Object.getOwnPropertyDescriptor(
  globalThis,
  "document",
);
let ui;
try {
  Object.defineProperty(globalThis, "document", {
    configurable: true,
    value: undefined,
  });
  ui = await import(moduleURL);
} finally {
  if (documentDescriptor)
    Object.defineProperty(globalThis, "document", documentDescriptor);
  else delete globalThis.document;
}
const {
  DEFAULT_SETTINGS,
  CODING_PROMPT,
  LIMITS,
  validateSettings,
  normalizeImage,
  checkImageBudget,
  buildRequest,
  safeLink,
  apiFailure,
  renderMarkdown,
} = ui;
const settings = (overrides = {}) => ({ ...DEFAULT_SETTINGS, ...overrides });
const pngSignature = Buffer.from([137, 80, 78, 71, 13, 10, 26, 10]);
const pngBase64 = pngSignature.toString("base64");
const jpegBase64 = Buffer.from([255, 216, 255, 224]).toString("base64");
const png = () => normalizeImage(pngBase64, "image/png", "screenshot.png");
const pngOfSize = (size) => {
  const bytes = Buffer.alloc(size);
  pngSignature.copy(bytes);
  return bytes.toString("base64");
};
const bodyOf = (...args) => JSON.parse(buildRequest(...args).json);

await test(
  "pure request and security helpers",
  { concurrency: false },
  async (t) => {
    await t.test("defaults match the UI contract", () => {
      assert.deepEqual(DEFAULT_SETTINGS, {
        mode: "chat",
        system: "",
        coding: false,
        max_tokens: 128,
        temperature: 0.8,
        top_p: 0.95,
        seed: "",
      });
      assert.deepEqual(LIMITS, {
        images: 8,
        imageBytes: 8 * 1024 ** 2,
        totalImageBytes: 16 * 1024 ** 2,
        chatBytes: 24 * 1024 ** 2,
        rawBytes: 1024 ** 2,
      });
      assert.ok(Object.isFrozen(DEFAULT_SETTINGS));
      assert.ok(Object.isFrozen(LIMITS));
    });

    await t.test(
      "sampling accepts numeric form values and u32 boundaries without mutating input",
      () => {
        const input = settings({
          max_tokens: "1",
          temperature: "0",
          top_p: "0.000001",
          seed: "0",
        });
        const before = { ...input };
        assert.deepEqual(
          validateSettings(input),
          settings({ max_tokens: 1, temperature: 0, top_p: 0.000001, seed: 0 }),
        );
        assert.deepEqual(input, before);
        assert.equal(
          validateSettings(settings({ seed: "4294967295" })).seed,
          4294967295,
        );
        assert.equal(
          validateSettings(
            settings({ max_tokens: 4294967295, temperature: 2, top_p: 1 }),
          ).max_tokens,
          4294967295,
        );
        assert.equal(validateSettings(settings()).seed, "");
      },
    );

    await t.test(
      "sampling rejects missing, nonfinite, fractional, and out-of-range values",
      () => {
        for (const overrides of [
          { max_tokens: "" },
          { max_tokens: 0 },
          { max_tokens: -1 },
          { max_tokens: 1.5 },
          { max_tokens: 4294967296 },
          { temperature: "" },
          { temperature: -0.1 },
          { temperature: 2.01 },
          { temperature: Infinity },
          { top_p: NaN },
          { top_p: 0 },
          { top_p: 1.01 },
          { top_p: "not a number" },
          { seed: -1 },
          { seed: 0.5 },
          { seed: 4294967296 },
          { seed: Infinity },
          { mode: "tools" },
          { system: {} },
          { coding: "true" },
        ]) {
          assert.throws(
            () => validateSettings(settings(overrides)),
            undefined,
            `accepted ${JSON.stringify(overrides)}`,
          );
        }
      },
    );

    await t.test(
      "chat serializes history, explicit instructions and only API fields",
      () => {
        const history = [
          {
            role: "user",
            content: "Earlier prompt",
            mode: "chat",
            settings: settings(),
          },
          {
            role: "assistant",
            content: "Earlier response",
            elapsed: 42,
            usage: { total_tokens: 7 },
          },
        ];
        const before = structuredClone(history);
        const result = buildRequest(
          history,
          "  Next prompt\n",
          [],
          settings({ coding: true, system: "Custom instructions", seed: 0 }),
          "model-id",
        );
        assert.equal(result.endpoint, "/v1/chat/completions");
        assert.deepEqual(JSON.parse(result.json), {
          max_tokens: 128,
          temperature: 0.8,
          top_p: 0.95,
          stream: false,
          model: "model-id",
          seed: 0,
          messages: [
            {
              role: "system",
              content: `${CODING_PROMPT}\n\nCustom instructions`,
            },
            { role: "user", content: "Earlier prompt" },
            { role: "assistant", content: "Earlier response" },
            { role: "user", content: "  Next prompt\n" },
          ],
        });
        assert.deepEqual(history, before);
        const minimal = bodyOf([], "Hello", [], settings());
        assert.ok(!("model" in minimal));
        assert.ok(!("seed" in minimal));
        assert.deepEqual(minimal.messages, [
          { role: "user", content: "Hello" },
        ]);
      },
    );

    await t.test(
      "raw completion isolates exact composer text from history, images and system instructions",
      () => {
        const history = [
          {
            role: "user",
            content: "DO NOT SEND",
            images: Array(9).fill(png()),
          },
        ];
        const result = buildRequest(
          history,
          "  exact raw\ntext  ",
          [],
          settings({ mode: "raw", coding: true, system: "DO NOT SEND" }),
        );
        assert.equal(result.endpoint, "/v1/completions");
        assert.deepEqual(JSON.parse(result.json), {
          max_tokens: 128,
          temperature: 0.8,
          top_p: 0.95,
          stream: false,
          prompt: "  exact raw\ntext  ",
        });
        assert.throws(
          () => buildRequest([], "Raw", [png()], settings({ mode: "raw" })),
          /cannot send images/,
        );
      },
    );

    await t.test(
      "empty drafts require an image and only user turns may carry images",
      () => {
        for (const content of ["", "  \n", undefined]) {
          assert.throws(
            () => buildRequest([], content, [], settings()),
            /Write a message/,
          );
        }
        assert.deepEqual(bodyOf([], "", [png()], settings()).messages, [
          { role: "user", content: "", images: [pngBase64] },
        ]);
        for (const role of ["assistant", "system"]) {
          assert.throws(
            () =>
              buildRequest(
                [{ role, content: "x", images: [png()] }],
                "x",
                [],
                settings(),
              ),
            /Only user/,
          );
        }
        assert.throws(
          () =>
            buildRequest([{ role: "tool", content: "x" }], "x", [], settings()),
          /Invalid conversation/,
        );
        assert.throws(
          () =>
            buildRequest([{ role: "user", content: {} }], "x", [], settings()),
          /Invalid conversation/,
        );
      },
    );

    await t.test(
      "PNG/JPEG normalization strips data prefixes and checks MIME against bytes",
      () => {
        assert.deepEqual(
          normalizeImage(
            `data:image/png;base64,${pngBase64}`,
            "image/png",
            "screenshot.png",
          ),
          png(),
        );
        assert.equal(normalizeImage(jpegBase64).mime, "image/jpeg");
        assert.equal(
          normalizeImage(`data:image/jpeg;base64,${jpegBase64}`).size,
          4,
        );
        assert.equal(
          normalizeImage(pngBase64, "", "x".repeat(201)).name.length,
          200,
        );
        for (const [value, mime] of [
          [pngBase64, "image/jpeg"],
          [jpegBase64, "image/png"],
          [pngBase64, "image/gif"],
          [`data:image/png;base64,${pngBase64}`, "image/jpeg"],
          ["data:image/svg+xml;base64,PHN2Zz4=", ""],
          ["data:image/png,not-base64", ""],
          ["%%%%", ""],
          ["YWJj=", ""],
          ["", ""],
          [null, ""],
          [Buffer.from("not an image").toString("base64"), ""],
        ])
          assert.throws(
            () => normalizeImage(value, mime),
            undefined,
            String(value),
          );
      },
    );

    await t.test(
      "decoded per-file limit catches padding at the 8 MiB boundary",
      () => {
        assert.equal(
          normalizeImage(pngOfSize(LIMITS.imageBytes)).size,
          LIMITS.imageBytes,
        );
        // Both lengths can have the same base64 length; checking encoded size alone is insufficient.
        assert.throws(
          () => normalizeImage(pngOfSize(LIMITS.imageBytes + 1)),
          /8 MiB/,
        );
      },
    );

    await t.test(
      "image count and decoded aggregate budgets include exact boundaries",
      () => {
        assert.equal(checkImageBudget([]), 0);
        assert.equal(checkImageBudget(Array(8).fill({ size: 1 })), 8);
        assert.equal(
          checkImageBudget([
            { size: LIMITS.imageBytes },
            { size: LIMITS.imageBytes },
          ]),
          LIMITS.totalImageBytes,
        );
        assert.throws(
          () => checkImageBudget(Array(9).fill({ size: 1 })),
          /8 images/,
        );
        assert.throws(
          () =>
            checkImageBudget([
              { size: LIMITS.imageBytes },
              { size: LIMITS.imageBytes },
              { size: 1 },
            ]),
          /16 MiB/,
        );
        for (const size of [
          0,
          -1,
          1.5,
          NaN,
          Infinity,
          undefined,
          LIMITS.imageBytes + 1,
        ]) {
          assert.throws(() => checkImageBudget([{ size }]), /8 MiB/);
        }
      },
    );

    await t.test(
      "request budgets include historical images and recompute decoded size",
      () => {
        const history = [
          { role: "user", content: "Earlier", images: Array(7).fill(png()) },
        ];
        const body = bodyOf(history, "Next", [png()], settings());
        assert.equal(
          body.messages[0].images.length + body.messages[1].images.length,
          8,
        );
        assert.throws(
          () => buildRequest(history, "Next", [png(), png()], settings()),
          /including conversation history/,
        );
        const large = {
          base64: pngOfSize(LIMITS.imageBytes),
          mime: "image/png",
          size: 1,
        };
        assert.throws(
          () =>
            buildRequest(
              [{ role: "user", content: "Earlier", images: [large, large] }],
              "Next",
              [png()],
              settings(),
            ),
          /16 MiB/,
        );
      },
    );

    await t.test(
      "serialized limits account for UTF-8, JSON escaping and history",
      () => {
        const raw = settings({ mode: "raw" });
        const overhead =
          Buffer.byteLength(buildRequest([], "x", [], raw).json) - 1;
        const exact = "x".repeat(LIMITS.rawBytes - overhead);
        assert.equal(
          Buffer.byteLength(buildRequest([], exact, [], raw).json),
          LIMITS.rawBytes,
        );
        assert.throws(() => buildRequest([], `${exact}x`, [], raw), /1 MiB/);
        for (const text of [
          "é".repeat(LIMITS.rawBytes / 2),
          '"'.repeat(LIMITS.rawBytes / 2),
        ]) {
          assert.throws(() => buildRequest([], text, [], raw), /1 MiB/);
        }
        assert.throws(
          () =>
            buildRequest(
              [{ role: "user", content: "x".repeat(LIMITS.chatBytes) }],
              "Next",
              [],
              settings(),
            ),
          /24 MiB/,
        );
      },
    );

    await t.test(
      "links allow only explicit HTTP(S), never active schemes or credentials",
      () => {
        assert.equal(
          safeLink("https://example.com/a?q=1#code"),
          "https://example.com/a?q=1#code",
        );
        assert.equal(safeLink("HTTP://example.com"), "http://example.com/");
        for (const url of [
          "javascript:alert(1)",
          "JaVaScRiPt:alert(1)",
          "data:text/html,<script>x</script>",
          "blob:https://example.com/id",
          "file:///tmp/a",
          "ftp://example.com",
          "/v1/models",
          "//example.com",
          "https://user:pass@example.com",
          "https://user@example.com",
          "https://example.com/\nfoo",
          "https://example.com/\0foo",
          " https://example.com",
          "https://",
        ])
          assert.equal(safeLink(url), null, url);
      },
    );

    await t.test(
      "API failures give actionable advice and bound untrusted server details",
      () => {
        for (const [status, pattern] of [
          [400, /context window/],
          [401, /API key/],
          [413, /too large/],
          [429, /Retry shortly/],
          [503, /unavailable/],
          [500, /server logs/],
        ]) {
          const error = apiFailure(status, "<script>untrusted</script>");
          assert.match(error.title, new RegExp(String(status)));
          assert.match(error.detail, pattern);
          assert.ok(
            error.detail.endsWith("Server: <script>untrusted</script>"),
          );
        }
        assert.equal(
          apiFailure(500, "x".repeat(2000)).detail.split("Server: ")[1].length,
          1600,
        );
      },
    );
  },
);

// This is intentionally a small DOM test double, not an HTML engine. It uses the
// real shell's IDs/tree, dispatches the real UI handlers, and fails on unsafe DOM
// sinks/storage access. Layout, native browser focus behavior, and CSP enforcement
// still require browser testing. Tests using browser globals run serially.
class MockNode {
  constructor(tag, document) {
    this.tagName = tag;
    this.ownerDocument = document;
    this.children = [];
    this.attributes = {};
    this.events = new Map();
    this.className = "";
    this.dataset = {};
    this.value = "";
    this.hidden = false;
    this.disabled = false;
    this.checked = false;
    this.clientWidth = 600;
    this.clientHeight = 600;
    this.scrollHeight = 1200;
    this.scrollTop = 0;
    this._text = "";
    this.classList = {
      contains: (name) => this.className.split(/\s+/).includes(name),
      add: (name) => {
        if (!this.classList.contains(name)) this.className += ` ${name}`;
      },
      remove: (name) => {
        this.className = this.className
          .split(/\s+/)
          .filter((item) => item !== name)
          .join(" ");
      },
    };
  }
  get textContent() {
    return this._text + this.children.map((node) => node.textContent).join("");
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
      this.parent.children = this.parent.children.filter(
        (node) => node !== this,
      );
    this.parent = null;
  }
  setAttribute(name, value) {
    if (/^on/i.test(name) || name === "style")
      this.ownerDocument.unsafe(`attribute ${name}`);
    this.attributes[name] = String(value);
    if (name === "class") this.className = String(value);
    if (name === "value") this.value = String(value);
    if (["hidden", "disabled", "checked"].includes(name)) this[name] = true;
    if (name.startsWith("data-")) this.dataset[name.slice(5)] = String(value);
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
    if (this.tagName === "a" && this.download) {
      this.ownerDocument.downloads.push({
        name: this.download,
        url: this.href,
      });
    }
    this.emit("click");
  }
  focus() {
    this.ownerDocument.activeElement = this;
  }
  select() {}
  showModal() {
    this.open = true;
  }
  close() {
    this.open = false;
  }
  querySelectorAll(selector) {
    const selectors = selector.split(",").map((part) => part.trim());
    const matches = (node) =>
      selectors.some((part) => {
        if (part.startsWith(".")) return node.classList.contains(part.slice(1));
        if (part === "a[href]")
          return (
            node.tagName === "a" && Boolean(node.href || node.attributes.href)
          );
        if (part === "button:not(:disabled)")
          return node.tagName === "button" && !node.disabled;
        return node.tagName === part;
      });
    const result = [];
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
  const document = new MockNode("#document");
  document.ownerDocument = document;
  document.ids = new Map();
  document.downloads = [];
  document.violations = [];
  document.unsafe = (operation) => {
    document.violations.push(operation);
    throw new Error(`Unsafe UI operation: ${operation}`);
  };
  document.createElement = (tag) => new MockNode(tag, document);
  document.createElementNS = (namespace, tag) => {
    assert.equal(namespace, "http://www.w3.org/2000/svg");
    assert.ok(["svg", "path"].includes(tag));
    return new MockNode(tag, document);
  };
  document.createTextNode = (text) => {
    const node = new MockNode("#text", document);
    node.textContent = text;
    return node;
  };
  document.createDocumentFragment = () => new MockNode("#fragment", document);
  document.getElementById = (id) => {
    assert.ok(document.ids.has(id), `UI references missing shell ID: ${id}`);
    return document.ids.get(id);
  };
  Object.defineProperty(document, "cookie", {
    get: () => document.unsafe("cookie read"),
    set: () => document.unsafe("cookie write"),
  });
  const voidTags = new Set(["meta", "link", "input", "br", "img", "hr"]);
  const stack = [document];
  // Only parse the trusted, checked-in shell, never model/user content.
  for (const match of shell.matchAll(/<\/?([a-z][\w-]*)\b[^>]*>/gi)) {
    const tag = match[1].toLowerCase();
    if (match[0].startsWith("</")) {
      assert.equal(stack.pop().tagName, tag, `unbalanced shell tag: ${tag}`);
      continue;
    }
    const node = new MockNode(tag, document);
    const attributes = match[0].slice(match[0].indexOf(tag) + tag.length, -1);
    for (const attr of attributes.matchAll(/([\w:-]+)(?:\s*=\s*"([^"]*)")?/g)) {
      node.setAttribute(attr[1], attr[2] ?? "");
      if (attr[1] === "id") {
        assert.ok(!document.ids.has(attr[2]), `duplicate shell ID: ${attr[2]}`);
        document.ids.set(attr[2], node);
      }
    }
    stack.at(-1).append(node);
    if (tag === "body") document.body = node;
    if (!voidTags.has(tag)) stack.push(node);
  }
  assert.deepEqual(stack, [document]);
  return document;
}

const flush = async () => {
  await nextTurn();
  await nextTurn();
};
const completion = (content = "Model response") => ({
  model: "local-model",
  choices: [{ message: { content }, finish_reason: "stop" }],
  usage: { prompt_tokens: 5, completion_tokens: 7, total_tokens: 12 },
});
let harnessSequence = 0;

async function makeHarness(
  t,
  {
    autoConnect = true,
    desktop = false,
    status = {},
    autoDiscover = true,
    discovery = { models: [], warnings: [], lastModel: null },
  } = {},
) {
  const document = makeDocument();
  const saved = new Map();
  const install = (name, descriptor) => {
    saved.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, {
      configurable: true,
      ...descriptor,
    });
  };
  const setGlobal = (name, value) => install(name, { writable: true, value });
  const timers = new Map(),
    intervals = new Map(),
    blobs = new Map();
  const calls = [];
  const requests = [],
    readers = [],
    clipboard = [];
  let timerID = 0;
  const h = {
    document,
    calls,
    requests,
    readers,
    clipboard,
    timers,
    intervals,
    blobs,
    update() {
      this.get("source-update").click();
      assert.equal(this.get("update-confirm-dialog").open, true);
      this.get("confirm-source-update").click();
      assert.equal(this.get("update-confirm-dialog").open, false);
    },
    get: (id) => document.getElementById(id),
    pending: () => requests.filter((request) => !request.settled),
    posts: () =>
      requests.filter((request) => request.options.method === "POST"),
    setDraft(text) {
      this.get("prompt").value = text;
      this.get("prompt").emit("input");
    },
    send() {
      this.get("composer").emit("submit");
    },
    messages: () =>
      document.getElementById("messages").querySelectorAll("article"),
    async respond(data, status = 200, { raw = false } = {}) {
      assert.equal(
        this.pending().length,
        1,
        "expected exactly one pending request",
      );
      this.pending()[0].resolve({
        ok: status >= 200 && status < 300,
        status,
        text: async () => (raw ? data : JSON.stringify(data)),
      });
      await flush();
    },
    pendingIPC(command) {
      return calls.filter(
        (call) => !call.settled && (!command || call.command === command),
      );
    },
    async replyIPC(command, value, reject = false) {
      const pending = this.pendingIPC(command);
      assert.equal(pending.length, 1, `expected one pending ${command}`);
      pending[0][reject ? "reject" : "resolve"](value);
      await flush();
      if (
        command === "desktop_status" &&
        autoDiscover &&
        this.pendingIPC("discover_models").length
      )
        await this.replyIPC("discover_models", discovery);
    },
    async reject(error) {
      assert.equal(this.pending().length, 1);
      this.pending()[0].reject(error);
      await flush();
    },
    async fireTimeouts(ms) {
      for (const [id, timer] of [...timers]) {
        if (timer.ms === ms) {
          timers.delete(id);
          timer.callback();
        }
      }
      await flush();
    },
    setKey(key) {
      this.get("api-key").value = key;
      this.get("api-key").emit("input");
    },
    async exportJSON() {
      this.get("export-json").click();
      const download = document.downloads.at(-1);
      assert.match(download.name, /\.json$/);
      return JSON.parse(await blobs.get(download.url).text());
    },
  };
  // Register cleanup before startup so even assertion failures restore globals.
  t.after(async () => {
    try {
      for (const request of h.pending())
        request.reject(new DOMException("Test cleanup", "AbortError"));
      for (const call of h.pendingIPC()) call.reject(new Error("Test cleanup"));
      await flush();
    } finally {
      timers.clear();
      intervals.clear();
      blobs.clear();
      for (const [name, descriptor] of saved) {
        if (descriptor) Object.defineProperty(globalThis, name, descriptor);
        else delete globalThis[name];
      }
    }
    assert.deepEqual(
      document.violations,
      [],
      "UI touched an unsafe DOM or persistence API",
    );
  });
  setGlobal("document", document);
  const window = new MockNode("#window", document);
  window.matchMedia = () => ({ matches: true, addEventListener() {} });
  if (desktop)
    window.__TAURI__ = {
      core: {
        invoke(command, args) {
          const call = { command, args, settled: false };
          calls.push(call);
          return new Promise((resolve, reject) => {
            call.resolve = (value) => {
              call.settled = true;
              resolve(value);
            };
            call.reject = (error) => {
              call.settled = true;
              reject(error);
            };
          });
        },
      },
    };
  setGlobal("window", window);
  setGlobal("navigator", {
    platform: "MacIntel",
    clipboard: {
      writeText: async (text) => {
        clipboard.push(text);
      },
    },
  });
  setGlobal("confirm", () => true);
  setGlobal("requestAnimationFrame", (callback) => {
    callback();
    return 0;
  });
  setGlobal("setTimeout", (callback, ms) => {
    const id = ++timerID;
    timers.set(id, { callback, ms });
    return id;
  });
  setGlobal("clearTimeout", (id) => timers.delete(id));
  setGlobal("setInterval", (callback, ms) => {
    const id = ++timerID;
    intervals.set(id, { callback, ms });
    return id;
  });
  setGlobal("clearInterval", (id) => intervals.delete(id));
  for (const name of ["localStorage", "sessionStorage", "indexedDB"]) {
    install(name, { get: () => document.unsafe(name) });
  }
  const NativeURL = globalThis.URL;
  setGlobal(
    "URL",
    class extends NativeURL {
      static createObjectURL(blob) {
        const url = `blob:test/${blobs.size + 1}`;
        blobs.set(url, blob);
        return url;
      }
      static revokeObjectURL(url) {
        blobs.delete(url);
      }
    },
  );
  setGlobal(
    "FileReader",
    class {
      readAsDataURL(file) {
        this.file = file;
        readers.push(this);
      }
      finish(data) {
        this.result = data;
        this.onload();
      }
      fail() {
        this.onerror();
      }
    },
  );
  setGlobal("fetch", (url, options = {}) => {
    assert.equal(desktop, false, "desktop must never fall back to fetch");
    assert.ok(
      ["/v1/models", "/v1/chat/completions", "/v1/completions"].includes(url),
      `unexpected network target: ${url}`,
    );
    const request = {
      url,
      options,
      body: options.body ? JSON.parse(options.body) : undefined,
      settled: false,
    };
    requests.push(request);
    return new Promise((resolve, reject) => {
      const finish = (callback, value) => {
        if (request.settled) return;
        request.settled = true;
        options.signal?.removeEventListener("abort", abort);
        callback(value);
      };
      const abort = () =>
        finish(reject, new DOMException("Stopped", "AbortError"));
      request.resolve = (value) => finish(resolve, value);
      request.reject = (error) => finish(reject, error);
      if (options.signal?.aborted) abort();
      else options.signal?.addEventListener("abort", abort, { once: true });
    });
  });
  // A unique fragment reruns the unmodified module's startup for each isolated DOM.
  await import(`${moduleURL}#lifecycle-${++harnessSequence}`);
  if (desktop) {
    assert.equal(h.pendingIPC()[0]?.command, "desktop_status");
    if (autoConnect)
      await h.replyIPC("desktop_status", {
        loaded: false,
        model: null,
        backend: "llama.cpp",
        platform: "macos",
        updating: false,
        ...status,
      });
  } else {
    assert.equal(h.pending()[0]?.url, "/v1/models");
    if (autoConnect) await h.respond({ data: [{ id: "local-model" }] });
  }
  return h;
}

const desktopStatus = (overrides = {}) => ({
  loaded: true,
  model: "local-model",
  backend: "llama.cpp",
  platform: "macos",
  updating: false,
  ...overrides,
});

await test("model options are validated before crossing IPC", () => {
  const input = {
    model: " /models/a.gguf ",
    ollama: false,
    ollamaDir: "/ignored",
    projector: " ",
    gpu: "AUTO",
    context: "4096",
  };
  assert.deepEqual(ui.validateModelOptions(input), {
    model: "/models/a.gguf",
    ollama: false,
    ollamaDir: null,
    projector: null,
    gpu: "auto",
    context: 4096,
  });
  for (const gpu of ["cpu", "all", "0", "32"])
    assert.equal(ui.validateModelOptions({ ...input, gpu }).gpu, gpu);
  for (const gpu of ["", "-1", "1.5", "1e2", "2147483648", "auto; rm"])
    assert.throws(
      () => ui.validateModelOptions({ ...input, gpu }),
      /GPU layers/,
    );
  for (const context of ["", "0", "-1", "1.5", "NaN", "Infinity", "4294967296"])
    assert.throws(
      () => ui.validateModelOptions({ ...input, context }),
      /Context tokens/,
    );
  assert.throws(
    () => ui.validateModelOptions({ ...input, model: " " }),
    /model path/,
  );
  assert.deepEqual(
    ui.validateModelOptions({
      ...input,
      model: "qwen3:8b",
      ollama: true,
      projector: " /vision.gguf ",
    }),
    {
      model: "qwen3:8b",
      ollama: true,
      ollamaDir: "/ignored",
      projector: "/vision.gguf",
      gpu: "auto",
      context: 4096,
    },
  );
});

await test("desktop transport skips already aborted requests and preserves IPC errors", async () => {
  const calls = [];
  const transport = ui.createTransport({
    __TAURI__: {
      core: {
        invoke: async (command, args) => {
          calls.push({ command, args });
          throw "Runtime disconnected";
        },
      },
    },
  });
  const controller = new AbortController();
  controller.abort();
  await assert.rejects(
    transport.request("/v1/models", { signal: controller.signal }),
    { name: "AbortError" },
  );
  assert.equal(calls.length, 0);
  await assert.rejects(transport.request("/v1/models"), /Runtime disconnected/);
  assert.equal(calls[0].command, "api_request");
  assert.equal(calls[0].args.method, "GET");
  assert.equal(calls[0].args.body, null);
  assert.equal(ui.createTransport({}).desktop, false);
});

const installedModel = (overrides = {}) => ({
  id: "lma:/models/small.gguf",
  name: "Small model",
  source: "lma",
  model: "/models/small.gguf",
  ollama: false,
  ollamaDir: null,
  projector: null,
  sizeBytes: 2 * 1024 ** 3,
  ...overrides,
});
const savedModel = (overrides = {}) => ({
  model: "/custom/large.gguf",
  ollama: false,
  ollamaDir: null,
  projector: "/custom/vision.gguf",
  gpu: "1",
  context: 16384,
  ...overrides,
});
const discoveryResult = (models = [], lastModel = null, warnings = []) => ({
  models,
  warnings,
  lastModel,
});
const modelRows = (h) =>
  h.get("installed-models").querySelectorAll(".installed-model");

await test(
  "desktop discovery and remembered selection",
  { concurrency: false },
  async (t) => {
    await t.test(
      "native status comes first; scanning never forces a manual dialog or arbitrary load",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          autoConnect: false,
          autoDiscover: false,
        });
        assert.deepEqual(
          h.calls.map((call) => call.command),
          ["desktop_status"],
        );
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loaded: false, model: null }),
        );
        assert.equal(h.pendingIPC("discover_models").length, 1);
        assert.equal(h.pendingIPC("discover_models")[0].args, undefined);
        assert.notEqual(h.get("model-dialog").open, true);
        assert.match(h.get("discovery-status").textContent, /Scanning/);
        await h.replyIPC(
          "discover_models",
          discoveryResult([
            installedModel(),
            installedModel({
              id: "big",
              name: "Large",
              model: "/models/large.gguf",
              sizeBytes: 100 * 1024 ** 3,
            }),
          ]),
        );
        assert.equal(modelRows(h).length, 2);
        assert.equal(h.get("model-path").value, "");
        assert.equal(h.get("load-model").disabled, true);
        assert.equal(h.pendingIPC("load_model").length, 0);
        assert.notEqual(h.get("model-dialog").open, true);
        h.get("empty-load").click();
        assert.equal(h.get("model-dialog").open, true);
        assert.equal(h.document.activeElement, h.get("model-search"));
        assert.notEqual(h.get("model-advanced").open, true);
        assert.match(modelRows(h)[1].textContent, /100.0 GiB/);
      },
    );

    await t.test(
      "a unique installed model is preselected but never automatically loaded",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          discovery: discoveryResult([installedModel()]),
        });
        assert.equal(h.get("model-path").value, "/models/small.gguf");
        assert.equal(modelRows(h)[0].getAttribute("aria-pressed"), "true");
        assert.equal(h.get("load-model").disabled, false);
        assert.equal(h.pendingIPC("load_model").length, 0);
        assert.notEqual(h.get("model-dialog").open, true);
      },
    );

    await t.test(
      "rows select only; GPU/context remain user-controlled and source/projector options stay exact",
      async (t) => {
        const ollama = installedModel({
          id: "ollama:vision",
          name: "Vision <img onerror=bad()>",
          source: "ollama",
          model: "vision:large",
          ollama: true,
          ollamaDir: "/custom/ollama",
          projector: "/custom/mmproj.gguf",
          sizeBytes: 512 * 1024 ** 2,
        });
        const h = await makeHarness(t, {
          desktop: true,
          discovery: discoveryResult([installedModel(), ollama]),
        });
        h.get("empty-load").click();
        h.get("model-gpu").value = "1";
        h.get("model-gpu").emit("input");
        h.get("model-context").value = "32768";
        h.get("model-context").emit("input");
        h.get("model-search").value = "OLLAMA";
        h.get("model-search").emit("input");
        assert.equal(modelRows(h).length, 1);
        assert.match(modelRows(h)[0].textContent, /Ollama.*Vision.*512.0 MiB/);
        assert.equal(modelRows(h)[0].querySelectorAll("img").length, 0);
        modelRows(h)[0].focus();
        modelRows(h)[0].click();
        assert.equal(
          h.document.activeElement,
          modelRows(h)[0],
          "keyboard focus survives selection",
        );
        assert.equal(h.pendingIPC("load_model").length, 0);
        assert.equal(h.get("model-source").value, "ollama");
        assert.equal(h.get("ollama-dir-field").hidden, false);
        assert.equal(h.get("model-gpu").value, "1");
        assert.equal(h.get("model-context").value, "32768");
        h.get("model-form").emit("submit");
        assert.deepEqual(h.pendingIPC("load_model")[0].args, {
          options: {
            model: "vision:large",
            ollama: true,
            ollamaDir: "/custom/ollama",
            projector: "/custom/mmproj.gguf",
            gpu: "1",
            context: 32768,
          },
        });
        assert.equal(modelRows(h)[0].disabled, true);
        assert.equal(h.get("rescan-models").disabled, true);
        await h.replyIPC(
          "load_model",
          desktopStatus({ model: "vision:large" }),
        );
        h.get("model-search").value = "small";
        h.get("model-search").emit("input");
        modelRows(h)[0].click();
        assert.equal(h.get("model-source").value, "local");
        assert.equal(h.get("ollama-dir").value, "");
        assert.equal(h.get("model-projector").value, "");
        h.get("model-search").value = "no match";
        h.get("model-search").emit("input");
        assert.equal(modelRows(h).length, 0);
        assert.equal(
          h.get("discovery-status").textContent,
          "No matching models",
        );
      },
    );

    for (const saved of [
      savedModel(),
      savedModel({
        model: "qwen:large",
        ollama: true,
        ollamaDir: "/other/ollama",
        gpu: "cpu",
      }),
    ]) {
      await t.test(
        `remembered ${saved.ollama ? "Ollama" : "custom file"} restores once with saved options, not defaults`,
        async (t) => {
          const h = await makeHarness(t, {
            desktop: true,
            discovery: discoveryResult([installedModel()], saved),
          });
          assert.deepEqual(h.pendingIPC("load_model")[0].args, {
            options: saved,
          });
          assert.notEqual(h.get("model-dialog").open, true);
          assert.equal(h.get("model-gpu").value, saved.gpu);
          assert.equal(h.get("model-context").value, String(saved.context));
          assert.match(
            h.get("installed-models").textContent,
            /Previous selection/,
          );
          assert.equal(modelRows(h)[0].getAttribute("aria-pressed"), "true");
          await h.replyIPC("load_model", desktopStatus({ model: saved.model }));
          h.get("model-selector").click();
          h.get("unload-model").click();
          assert.equal(h.pendingIPC("unload_model")[0].args, undefined);
          await h.replyIPC(
            "unload_model",
            desktopStatus({ loaded: false, model: null }),
          );
          h.get("rescan-models").click();
          await h.replyIPC("discover_models", discoveryResult([], saved));
          h.get("refresh-model").click();
          await h.replyIPC(
            "desktop_status",
            desktopStatus({ loaded: false, model: null }),
          );
          assert.equal(
            h.calls.filter((call) => call.command === "load_model").length,
            1,
          );
          assert.equal(
            h.calls.filter((call) => call.command === "discover_models").length,
            2,
          );
          assert.equal(h.get("model-path").value, saved.model);
          assert.equal(modelRows(h).length, 1);
        },
      );
    }

    await t.test(
      "stale remembered load shows a recoverable chooser and never retries itself",
      async (t) => {
        const saved = savedModel();
        const h = await makeHarness(t, {
          desktop: true,
          discovery: discoveryResult([installedModel()], saved),
        });
        await h.replyIPC("load_model", "Missing <script>model</script>", true);
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loaded: false, model: null }),
        );
        assert.equal(h.get("model-dialog").open, true);
        assert.equal(h.get("model-error").hidden, false);
        assert.match(h.get("model-error").textContent, /Missing <script>/);
        assert.equal(h.get("model-error").children.length, 0);
        assert.equal(h.get("model-path").disabled, false);
        assert.equal(h.get("model-gpu").value, "1");
        assert.equal(modelRows(h).length, 2);
        h.get("rescan-models").click();
        await h.replyIPC(
          "discover_models",
          discoveryResult([installedModel()], saved),
        );
        await h.fireTimeouts(750);
        assert.equal(
          h.calls.filter((call) => call.command === "load_model").length,
          1,
        );
        h.get("model-path").value = "/fixed/manual.gguf";
        h.get("model-path").emit("input");
        h.get("model-form").emit("submit");
        assert.equal(
          h.pendingIPC("load_model")[0].args.options.model,
          "/fixed/manual.gguf",
        );
        await h.replyIPC("load_model", desktopStatus());
        assert.equal(h.get("model-error").hidden, true);
      },
    );

    await t.test(
      "discovery failure and warnings are safe text; rescan preserves manual input and controls",
      async (t) => {
        const h = await makeHarness(t, { desktop: true, autoDiscover: false });
        await h.replyIPC("discover_models", "Scan <img> failed", true);
        assert.notEqual(h.get("model-dialog").open, true);
        assert.match(
          h.get("discovery-warnings").textContent,
          /Scan <img> failed/,
        );
        assert.equal(h.get("discovery-warnings").children.length, 0);
        assert.equal(h.get("rescan-models").disabled, false);
        h.get("empty-load").click();
        h.get("model-path").value = "/manual/outside.gguf";
        h.get("model-path").emit("input");
        h.get("rescan-models").click();
        assert.equal(h.get("model-path").disabled, false);
        assert.equal(h.get("rescan-models").disabled, true);
        await h.replyIPC(
          "discover_models",
          discoveryResult([installedModel()], null, [
            "Ollama <script> unavailable",
          ]),
        );
        assert.equal(h.get("model-path").value, "/manual/outside.gguf");
        assert.equal(modelRows(h).length, 1);
        assert.match(
          h.get("discovery-warnings").textContent,
          /Ollama <script>/,
        );
        h.get("rescan-models").click();
        await h.replyIPC("discover_models", {
          models: "malformed",
          warnings: [],
          lastModel: null,
        });
        assert.match(
          h.get("discovery-warnings").textContent,
          /Invalid model discovery/,
        );
        assert.equal(modelRows(h).length, 1);
        assert.equal(h.get("load-model").disabled, false);
      },
    );

    for (const action of [
      "navigation",
      "manual",
      "load",
      "update",
      "dismiss",
    ]) {
      await t.test(
        `late startup discovery cannot override ${action} intent`,
        async (t) => {
          const h = await makeHarness(t, {
            desktop: true,
            autoDiscover: false,
          });
          if (action === "navigation") h.get("new-chat").click();
          if (action === "manual" || action === "load") {
            h.get("model-path").value = "/user/chosen.gguf";
            h.get("model-path").emit("input");
            if (action === "load") h.get("model-form").emit("submit");
          }
          if (action === "update") h.update();
          if (action === "dismiss") {
            h.get("empty-load").click();
            h.get("close-model").click();
          }
          await h.replyIPC(
            "discover_models",
            discoveryResult([installedModel()], savedModel()),
          );
          assert.notEqual(h.get("model-path").value, "/custom/large.gguf");
          assert.equal(
            h.calls.filter((call) => call.command === "load_model").length,
            action === "load" ? 1 : 0,
          );
          assert.notEqual(h.get("model-dialog").open, true);
          if (action === "load") {
            await h.replyIPC("load_model", desktopStatus());
            assert.equal(h.get("model-path").value, "/user/chosen.gguf");
          }
          if (action === "update") {
            await h.replyIPC("source_update", { message: "Started" });
            assert.equal(modelRows(h)[0].disabled, true);
          }
        },
      );
    }

    for (const status of [
      desktopStatus(),
      desktopStatus({ loaded: false, model: null, loading: true }),
      desktopStatus({ updating: true }),
    ]) {
      await t.test(
        `startup with native loaded/loading/updating state never double-loads: ${JSON.stringify(status)}`,
        async (t) => {
          const h = await makeHarness(t, {
            desktop: true,
            status,
            autoDiscover: false,
          });
          // Complete an existing native operation before the scan returns.
          if (status.loading) {
            await h.fireTimeouts(750);
            await h.replyIPC(
              "desktop_status",
              desktopStatus({ loaded: false, model: null }),
            );
          }
          await h.replyIPC(
            "discover_models",
            discoveryResult([installedModel()], savedModel()),
          );
          assert.equal(h.pendingIPC("load_model").length, 0);
          assert.notEqual(h.get("model-dialog").open, true);
        },
      );
    }

    await t.test(
      "a native load observed during scanning invalidates delayed automatic restore",
      async (t) => {
        const h = await makeHarness(t, { desktop: true, autoDiscover: false });
        h.get("refresh-model").click();
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loaded: false, model: null, loading: true }),
        );
        await h.fireTimeouts(750);
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loaded: false, model: null }),
        );
        await h.replyIPC(
          "discover_models",
          discoveryResult([installedModel()], savedModel()),
        );
        assert.equal(
          h.calls.filter((call) => call.command === "load_model").length,
          0,
        );
        assert.equal(h.get("model-path").value, "");
        assert.equal(modelRows(h).length, 2);
      },
    );

    await t.test(
      "failed restore respects navigation while error recheck is pending",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          discovery: discoveryResult([], savedModel()),
        });
        await h.replyIPC("load_model", "Model missing", true);
        h.get("close-model").emit("click");
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loaded: false, model: null }),
        );
        assert.notEqual(h.get("model-dialog").open, true);
        assert.match(h.get("model-error").textContent, /Model missing/);
      },
    );

    await t.test(
      "preference save warnings are optional, unobtrusive, and never invalidate a successful load",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          discovery: discoveryResult([installedModel()]),
        });
        h.get("model-form").emit("submit");
        await h.replyIPC(
          "load_model",
          desktopStatus({ preferenceWarning: "Cannot save <preferences>" }),
        );
        assert.equal(h.get("loaded-model").hidden, false);
        assert.match(
          h.get("model-error").textContent,
          /Cannot save <preferences>/,
        );
        assert.equal(h.get("model-error").getAttribute("role"), "status");
        assert.equal(h.get("model-error").classList.contains("muted"), true);
        h.get("refresh-model").click();
        await h.replyIPC("desktop_status", desktopStatus());
        assert.equal(h.get("model-error").hidden, true);
      },
    );

    await t.test(
      "late scans and failed preference saves never erase a newly loaded custom selection",
      async (t) => {
        const h = await makeHarness(t, { desktop: true, autoDiscover: false });
        h.get("model-path").value = "/new/custom.gguf";
        h.get("model-path").emit("input");
        h.get("model-form").emit("submit");
        await h.replyIPC(
          "load_model",
          desktopStatus({ preferenceWarning: "Preferences are read-only" }),
        );
        await h.replyIPC("discover_models", discoveryResult([], savedModel()));
        assert.equal(h.get("model-path").value, "/new/custom.gguf");
        assert.equal(modelRows(h).length, 1);
        assert.match(modelRows(h)[0].textContent, /\/new\/custom.gguf/);
        h.get("rescan-models").click();
        await h.replyIPC("discover_models", discoveryResult([], savedModel()));
        assert.match(modelRows(h)[0].textContent, /\/new\/custom.gguf/);
        h.get("rescan-models").click();
        await h.replyIPC("discover_models", discoveryResult());
        assert.match(modelRows(h)[0].textContent, /\/new\/custom.gguf/);
        assert.equal(
          h.calls.filter((call) => call.command === "load_model").length,
          1,
        );
      },
    );

    await t.test(
      "removed or corrupt preferences clear scan-cached selections without restoring them",
      async (t) => {
        const h = await makeHarness(t, { desktop: true, autoDiscover: false });
        h.get("new-chat").click();
        await h.replyIPC(
          "discover_models",
          discoveryResult([installedModel()], savedModel()),
        );
        assert.match(
          h.get("installed-models").textContent,
          /Previous selection/,
        );
        assert.equal(h.pendingIPC("load_model").length, 0);
        h.get("rescan-models").click();
        await h.replyIPC(
          "discover_models",
          discoveryResult([installedModel()], null, [
            "Saved preferences are invalid",
          ]),
        );
        assert.equal(modelRows(h).length, 1);
        assert.doesNotMatch(
          h.get("installed-models").textContent,
          /Previous selection|custom\/large/,
        );
        assert.match(
          h.get("discovery-warnings").textContent,
          /Saved preferences are invalid/,
        );
        assert.equal(h.get("model-path").value, "/models/small.gguf");
        assert.equal(h.get("model-path").disabled, false);
        h.get("rescan-models").click();
        await h.replyIPC("discover_models", discoveryResult());
        assert.equal(modelRows(h).length, 0);
        assert.equal(
          h.calls.filter((call) => call.command === "load_model").length,
          0,
        );
      },
    );

    await t.test(
      "invalid saved options never fall back to guessed defaults",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          discovery: discoveryResult(
            [installedModel()],
            savedModel({ gpu: null }),
          ),
        });
        assert.equal(h.pendingIPC("load_model").length, 0);
        assert.match(
          h.get("discovery-warnings").textContent,
          /Invalid saved model options/,
        );
        assert.equal(modelRows(h).length, 1);
        assert.equal(h.get("model-path").disabled, false);
      },
    );

    await t.test(
      "browser mode never scans or uses native model controls",
      async (t) => {
        const h = await makeHarness(t);
        h.get("rescan-models").click();
        h.get("model-selector").emit("click");
        h.get("model-form").emit("submit");
        h.get("refresh-model").click();
        await h.respond({ data: [{ id: "browser-model" }] });
        assert.deepEqual(h.calls, []);
        assert.equal(h.get("model-selector").hidden, true);
        assert.equal(h.get("model-status").textContent, "browser-model");
        assert.notEqual(h.get("model-dialog").open, true);
      },
    );
  },
);

await test(
  "desktop lifecycle uses only the documented IPC contract",
  { concurrency: false },
  async (t) => {
    await t.test(
      "unloaded startup scans without forcing the picker; local load and unload preserve the draft",
      async (t) => {
        const h = await makeHarness(t, { desktop: true });
        assert.notEqual(h.get("model-dialog").open, true);
        assert.equal(
          h.calls.filter((call) => call.command === "discover_models").length,
          1,
        );
        assert.equal(
          h.calls.find((call) => call.command === "discover_models").args,
          undefined,
        );
        h.get("empty-load").click();
        assert.equal(h.get("model-dialog").open, true);
        assert.equal(h.get("model-selector").hidden, false);
        assert.equal(h.get("web-settings").hidden, true);
        assert.equal(h.get("app-settings").hidden, false);
        assert.equal(h.get("empty-load").hidden, false);
        h.setDraft("Keep this draft");
        assert.equal(h.get("send").disabled, true);
        h.send();
        assert.equal(h.pendingIPC("api_request").length, 0);
        h.get("model-path").value = "/models/test.gguf";
        h.get("model-form").emit("submit");
        assert.equal(h.get("load-model").disabled, true);
        assert.equal(h.get("new-chat").disabled, true);
        assert.equal(h.get("prompt").readOnly, true);
        assert.deepEqual(h.pendingIPC("load_model")[0].args, {
          options: {
            model: "/models/test.gguf",
            ollama: false,
            ollamaDir: null,
            projector: null,
            gpu: "auto",
            context: 4096,
          },
        });
        await h.replyIPC("load_model", desktopStatus());
        assert.equal(h.get("model-dialog").open, false);
        assert.equal(h.get("desktop-model-name").textContent, "local-model");
        assert.equal(h.get("prompt").value, "Keep this draft");
        assert.equal(h.get("send").disabled, false);
        assert.equal(h.get("starters").hidden, false);
        h.get("model-selector").click();
        h.get("unload-model").click();
        assert.equal(h.pendingIPC("unload_model")[0].args, undefined);
        await h.replyIPC(
          "unload_model",
          desktopStatus({ loaded: false, model: null }),
        );
        assert.equal(h.get("loaded-model").hidden, true);
        assert.equal(h.get("send").disabled, true);
        assert.equal(h.get("prompt").value, "Keep this draft");
        assert.equal(h.requests.length, 0);
      },
    );

    await t.test(
      "Ollama and advanced options, validation, and failed replacement recovery",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          status: desktopStatus(),
        });
        h.get("model-selector").click();
        h.get("model-source").value = "ollama";
        h.get("model-source").emit("change");
        assert.equal(h.get("ollama-dir-field").hidden, false);
        assert.equal(
          h.get("model-path-label").textContent,
          "Ollama model name",
        );
        h.get("model-path").value = "qwen3:8b";
        h.get("ollama-dir").value = "/models/ollama";
        h.get("model-projector").value = "/models/vision.gguf";
        h.get("model-gpu").value = "-1";
        h.get("model-form").emit("submit");
        assert.match(h.get("model-error").textContent, /GPU layers/);
        assert.equal(h.pendingIPC("load_model").length, 0);
        h.get("model-gpu").value = "32";
        h.get("model-context").value = "8192";
        h.get("model-form").emit("submit");
        assert.deepEqual(h.pendingIPC("load_model")[0].args.options, {
          model: "qwen3:8b",
          ollama: true,
          ollamaDir: "/models/ollama",
          projector: "/models/vision.gguf",
          gpu: "32",
          context: 8192,
        });
        await h.replyIPC(
          "load_model",
          "<img src=x onerror=bad()> failed load",
          true,
        );
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loaded: false, model: null }),
        );
        assert.match(h.get("model-error").textContent, /<img/);
        assert.equal(h.get("model-error").children.length, 0);
        assert.equal(h.get("model-dialog").open, true);
        assert.equal(h.get("load-model").disabled, false);
        assert.equal(h.get("loaded-model").hidden, true);
        h.setDraft("cannot send to the old model");
        assert.equal(h.get("send").disabled, true);
      },
    );

    await t.test(
      "chat responses, API errors and raw completion share the safe rendering path",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          status: desktopStatus(),
        });
        h.setDraft("Hello");
        h.send();
        await flush();
        const first = h.pendingIPC("api_request")[0];
        assert.deepEqual(Object.keys(first.args).sort(), [
          "body",
          "id",
          "method",
          "path",
        ]);
        assert.equal(first.args.method, "POST");
        assert.equal(first.args.path, "/v1/chat/completions");
        assert.equal(first.args.body.stream, false);
        assert.equal(first.args.body.model, "local-model");
        assert.deepEqual(first.args.body.messages, [
          { role: "user", content: "Hello" },
        ]);
        assert.equal(h.get("model-selector").disabled, true);
        await h.replyIPC("api_request", {
          status: 200,
          body: completion(
            "<script>bad()</script>\n\n```js\nalert('text only')\n```",
          ),
        });
        assert.equal(h.messages().length, 2);
        assert.equal(h.get("messages").querySelectorAll("script").length, 0);
        assert.equal(
          h.get("messages").querySelector("code").textContent,
          "alert('text only')",
        );
        assert.equal(h.get("model-selector").disabled, false);
        h.setDraft("Retry draft");
        h.send();
        await flush();
        assert.notEqual(h.pendingIPC("api_request")[0].args.id, first.args.id);
        await h.replyIPC("api_request", {
          status: 429,
          body: { error: { message: "<script>busy</script>" } },
        });
        assert.match(h.get("error-detail").textContent, /Retry shortly/);
        assert.equal(h.get("error-detail").children.length, 0);
        assert.equal(h.get("prompt").value, "Retry draft");
        assert.equal(h.messages().length, 2);
        h.get("open-settings").click();
        h.get("input-mode").value = "raw";
        h.get("settings-form").emit("submit");
        h.send();
        await flush();
        const raw = h.pendingIPC("api_request")[0].args;
        assert.equal(raw.path, "/v1/completions");
        assert.equal(raw.body.prompt, "Retry draft");
        assert.equal(raw.body.messages, undefined);
        await h.replyIPC("api_request", {
          status: 200,
          body: { choices: [{ text: "raw result" }] },
        });
        assert.equal(h.messages().length, 2);
        const exported = await h.exportJSON();
        assert.equal(exported.settings.mode, "raw");
        assert.equal(exported.messages[1].content, "raw result");
        assert.equal(h.get("toast").hidden, true);
      },
    );

    await t.test(
      "stop maps the request ID to cancel; late completion cannot commit or overwrite retry",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          status: desktopStatus(),
        });
        h.setDraft("Keep on stop");
        h.send();
        await flush();
        const original = h.pendingIPC("api_request")[0];
        h.get("stop").click();
        await flush();
        assert.deepEqual(h.pendingIPC("cancel_request")[0].args, {
          id: original.args.id,
        });
        assert.equal(h.get("prompt").value, "Keep on stop");
        assert.equal(h.messages().length, 0);
        assert.equal(h.get("stop").hidden, true);
        await h.replyIPC("cancel_request", "Already finished", true);
        h.get("retry-send").click();
        await flush();
        const retry = h
          .pendingIPC("api_request")
          .find((call) => call !== original);
        original.resolve({ status: 200, body: completion("stale reply") });
        await flush();
        assert.equal(
          h.messages().length,
          1,
          "only the retry's pending user turn is visible",
        );
        assert.equal(
          h.get("messages").textContent.includes("stale reply"),
          false,
        );
        assert.equal(h.get("stop").hidden, false);
        retry.resolve({ status: 200, body: completion("fresh reply") });
        await flush();
        assert.equal(h.messages().length, 2);
        assert.match(h.get("messages").textContent, /fresh reply/);
        assert.equal(h.intervals.size, 0);
      },
    );

    await t.test(
      "desktop startup errors and malformed API replies stay recoverable",
      async (t) => {
        const h = await makeHarness(t, { desktop: true, autoConnect: false });
        await h.replyIPC("desktop_status", "IPC unavailable", true);
        assert.match(h.get("connection-error").textContent, /IPC unavailable/);
        h.get("refresh-model").click();
        await h.replyIPC("desktop_status", desktopStatus());
        h.setDraft("Keep for retry");
        h.send();
        await flush();
        await h.replyIPC("api_request", { status: 200 });
        assert.match(
          h.get("error-detail").textContent,
          /Invalid desktop API response/,
        );
        assert.equal(h.get("prompt").value, "Keep for retry");
        assert.equal(h.messages().length, 0);
        h.get("retry-send").click();
        await flush();
        await h.replyIPC("api_request", "Native runtime failed", true);
        assert.match(
          h.get("error-detail").textContent,
          /Native runtime failed/,
        );
        assert.equal(h.get("prompt").value, "Keep for retry");
      },
    );

    await t.test(
      "reloaded loading status locks model actions until a full status refresh",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          status: desktopStatus({ loading: true }),
        });
        assert.notEqual(h.get("model-dialog").open, true);
        assert.equal(h.get("desktop-model-name").textContent, "Loading model…");
        assert.equal(h.get("loaded-model").hidden, true);
        h.setDraft("Wait for the model");
        for (const id of [
          "send",
          "model-selector",
          "load-model",
          "unload-model",
          "source-update",
        ])
          assert.equal(h.get(id).disabled, true, id);
        h.send();
        h.get("model-form").emit("submit");
        assert.equal(h.pendingIPC("api_request").length, 0);
        assert.equal(h.pendingIPC("load_model").length, 0);
        await h.fireTimeouts(750);
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loading: true, loaded: false, model: null }),
        );
        assert.equal(h.get("send").disabled, true);
        await h.fireTimeouts(750);
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loading: false, model: "new-model" }),
        );
        assert.equal(h.get("desktop-model-name").textContent, "new-model");
        assert.equal(h.get("loaded-model").hidden, false);
        assert.equal(h.get("send").disabled, false);
        assert.equal(h.get("model-selector").disabled, false);
        assert.equal(h.get("prompt").value, "Wait for the model");
        await h.fireTimeouts(750);
        assert.equal(h.pendingIPC().length, 0);
      },
    );

    await t.test(
      "loading-status failure clears readiness and allows reconnect",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          status: desktopStatus({ loading: true }),
        });
        await h.fireTimeouts(750);
        await h.replyIPC("desktop_status", "Runtime disconnected", true);
        assert.equal(h.get("model-selector").disabled, false);
        assert.equal(h.get("refresh-model").disabled, false);
        assert.equal(h.get("loaded-model").hidden, true);
        h.setDraft("Keep while offline");
        assert.equal(h.get("send").disabled, true);
        h.get("refresh-model").click();
        await h.replyIPC("desktop_status", desktopStatus({ loading: false }));
        assert.equal(h.get("send").disabled, false);
      },
    );

    await t.test(
      "source update confirms only on click; cancelling has no side effects",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          status: desktopStatus(),
        });
        h.setDraft("Unchanged draft");
        h.get("open-settings").click();
        assert.notEqual(h.get("update-confirm-dialog").open, true);
        h.get("confirm-source-update").emit("click");
        assert.equal(h.pendingIPC().length, 0);
        h.get("source-update").click();
        assert.equal(h.get("update-confirm-dialog").open, true);
        assert.equal(h.document.activeElement, h.get("cancel-source-update"));
        assert.equal(h.pendingIPC().length, 0);
        h.get("cancel-source-update").click();
        assert.equal(h.get("update-confirm-dialog").open, false);
        assert.equal(h.pendingIPC().length, 0);
        assert.equal(h.get("send").disabled, false);
        assert.equal(h.get("source-update").disabled, false);
        assert.equal(h.get("prompt").value, "Unchanged draft");
        await h.fireTimeouts(750);
        assert.equal(h.pendingIPC().length, 0);
      },
    );

    await t.test(
      "Windows updater instructions stay visible as errors after status refresh",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          status: desktopStatus({ platform: "windows" }),
        });
        h.get("open-settings").click();
        h.update();
        h.get("source-update").emit("click");
        h.get("confirm-source-update").emit("click");
        assert.equal(h.pendingIPC("source_update").length, 1);
        assert.equal(h.get("update-confirm-dialog").open, false);
        const instructions =
          "Run install/install.ps1 --update externally. <script>not HTML</script>";
        await h.replyIPC("source_update", instructions, true);
        assert.equal(h.get("update-error").textContent, instructions);
        assert.equal(h.get("update-error").children.length, 0);
        assert.equal(h.get("source-update").disabled, true);
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ platform: "windows" }),
        );
        assert.equal(h.get("update-error").textContent, instructions);
        assert.equal(h.get("update-error").hidden, false);
        h.setDraft("Model was not unloaded");
        assert.equal(h.get("send").disabled, false);
        assert.equal(h.get("source-update").disabled, false);
        assert.equal(h.get("toast").hidden, true);
      },
    );

    await t.test(
      "Failed without an error and a failed recheck never restore stale model readiness",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          status: desktopStatus(),
        });
        h.update();
        await h.replyIPC("source_update", { message: "Started" });
        await h.fireTimeouts(750);
        await h.replyIPC("update_status", {
          state: "Failed",
          log: "Build failed",
          error: null,
        });
        assert.equal(
          h.get("update-error").textContent,
          "Source update failed.",
        );
        await h.replyIPC("desktop_status", "Status unavailable", true);
        assert.equal(h.get("loaded-model").hidden, true);
        assert.equal(h.get("loaded-model-name").textContent, "");
        assert.equal(
          h.get("desktop-model-name").textContent,
          "Model unavailable",
        );
        h.setDraft("Keep after failure");
        assert.equal(h.get("send").disabled, true);
        assert.equal(h.get("refresh-model").disabled, false);
        assert.equal(
          h.get("update-error").textContent,
          "Source update failed.",
        );
        assert.match(
          h.get("connection-error").textContent,
          /Status unavailable/,
        );
        await h.fireTimeouts(750);
        assert.equal(h.pendingIPC().length, 0);
      },
    );

    await t.test(
      "update polls while active, renders logs as text, and stops on success",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          status: desktopStatus(),
        });
        h.get("open-settings").click();
        h.update();
        assert.equal(h.get("source-update").disabled, true);
        assert.equal(h.get("model-selector").disabled, true);
        assert.equal(h.pendingIPC("source_update")[0].args, undefined);
        await h.fireTimeouts(750);
        await h.replyIPC("update_status", {
          state: "Idle",
          log: "",
          error: null,
        });
        await h.replyIPC("desktop_status", desktopStatus());
        assert.equal(
          h.get("source-update").disabled,
          true,
          "source command is still pending",
        );
        await h.replyIPC("source_update", { message: "Building" });
        h.get("close-settings").click();
        await h.fireTimeouts(750);
        await h.replyIPC("update_status", {
          state: "Running",
          log: "<script>build output</script>",
          error: null,
        });
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ updating: true, loaded: false, model: null }),
        );
        assert.equal(h.get("loaded-model").hidden, true);
        assert.equal(h.get("desktop-model-name").textContent, "Select model");
        assert.equal(h.get("update-log").children.length, 0);
        assert.match(h.get("update-log").textContent, /<script>/);
        h.get("open-settings").click();
        assert.equal(h.get("settings-dialog").open, true);
        await h.fireTimeouts(750);
        await h.replyIPC("update_status", {
          state: "Completed",
          log: "Finished",
          error: null,
        });
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loaded: false, model: null }),
        );
        assert.equal(h.get("loaded-model").hidden, true);
        assert.equal(h.get("desktop-model-name").textContent, "Select model");
        h.setDraft("Keep after updating");
        assert.equal(h.get("send").disabled, true);
        assert.equal(h.get("empty-load").hidden, false);
        assert.equal(h.get("source-update").disabled, false);
        assert.equal(h.get("model-selector").disabled, false);
        await h.fireTimeouts(750);
        assert.equal(h.pendingIPC("update_status").length, 0);
        assert.equal(h.get("toast").hidden, true);
      },
    );

    await t.test(
      "unsupported updates and polling errors fail cleanly; existing updates resume polling",
      async (t) => {
        const h = await makeHarness(t, {
          desktop: true,
          status: desktopStatus({ updating: true }),
        });
        await h.fireTimeouts(750);
        await h.replyIPC("update_status", {
          state: "Failed",
          log: "build failed",
          error: "Compiler unavailable",
        });
        assert.match(h.get("update-error").textContent, /Compiler unavailable/);
        assert.equal(h.get("source-update").disabled, true);
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loaded: false, model: null }),
        );
        assert.equal(h.get("loaded-model").hidden, true);
        assert.equal(h.get("desktop-model-name").textContent, "Select model");
        h.setDraft("Keep after failed update");
        assert.equal(h.get("send").disabled, true);
        assert.equal(h.get("source-update").disabled, false);
        h.get("open-settings").click();
        h.update();
        await h.replyIPC(
          "source_update",
          "Source updates are disabled in this build",
          true,
        );
        assert.match(
          h.get("update-error").textContent,
          /disabled in this build/,
        );
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loaded: false, model: null }),
        );
        assert.equal(h.get("source-update").disabled, false);
        await h.fireTimeouts(750);
        assert.equal(h.pendingIPC("update_status").length, 0);
        h.update();
        await h.replyIPC("source_update", { message: "Started" });
        await h.fireTimeouts(750);
        await h.replyIPC("update_status", "Lost status connection", true);
        assert.match(
          h.get("update-error").textContent,
          /Lost status connection/,
        );
        await h.replyIPC(
          "desktop_status",
          desktopStatus({ loaded: false, model: null }),
        );
        assert.equal(h.get("source-update").disabled, false);
      },
    );
  },
);

await test("chat shell keeps instructions out of the workspace", () => {
  assert.doesNotMatch(
    shell,
    /Your workspace, not an archive|A LITTLE SPACE TO THINK|MAKE IT YOURS|welcome-note|settings-mode-help|id="mode-note"/,
  );
  assert.match(shell, /placeholder="Message…"/);
  assert.match(shell, /Temporary chats/);
  assert.match(shell, /id="send-shortcut"/);
  assert.match(shell, /Update from source/);
  assert.doesNotMatch(shell, /Update app/);
  assert.doesNotMatch(
    shell.match(/<main\b[^>]*>[\s\S]*?<\/main>/)[0],
    /This unloads the model/,
  );
  assert.match(
    shell,
    /This unloads the model and builds the latest checkout\.\s+Restart\s+when complete\./,
  );
  assert.match(shell, /id="error-detail"/);
  assert.match(shell, /aria-label="Conversation messages"/);
  assert.match(shell, /<script type="module" src="\/chat.js"><\/script>/);
  assert.doesNotMatch(
    shell,
    /\s(?:style|on\w+)\s*=|https?:\/\/|http-equiv="Content-Security-Policy"/i,
  );
  assert.doesNotMatch(
    source,
    /\.innerHTML|\.outerHTML|insertAdjacentHTML|\beval\s*\(|new Function\s*\(/,
  );
});

await test(
  "mocked DOM lifecycle and security",
  { concurrency: false },
  async (t) => {
    await t.test(
      "icon controls and mobile sidebar retain keyboard access",
      async (t) => {
        const h = await makeHarness(t);
        for (const node of h.document.querySelectorAll(".icon-button")) {
          assert.ok(
            node.getAttribute("aria-label"),
            "icon controls need an accessible name",
          );
          const icon = node.querySelector("svg");
          assert.ok(icon);
          assert.equal(icon.getAttribute("aria-hidden"), "true");
        }
        h.get("open-sidebar").click();
        assert.equal(h.get("sidebar").getAttribute("aria-modal"), "true");
        assert.equal(h.document.querySelector(".workspace").inert, true);
        h.get("close-sidebar").focus();
        const backTab = h.document.emit("keydown", {
          key: "Tab",
          shiftKey: true,
        });
        assert.equal(backTab.defaultPrevented, true);
        assert.equal(h.document.activeElement, h.get("open-settings"));
        h.document.emit("keydown", { key: "Tab" });
        assert.equal(h.document.activeElement, h.get("close-sidebar"));
        h.document.emit("keydown", { key: "Escape" });
        assert.equal(h.document.querySelector(".workspace").inert, false);
        assert.equal(h.document.activeElement, h.get("open-sidebar"));
        assert.equal(
          h.get("open-sidebar").getAttribute("aria-expanded"),
          "false",
        );
        h.get("open-sidebar").click();
        h.get("open-settings").click();
        assert.equal(h.get("settings-dialog").open, true);
        assert.equal(h.document.querySelector(".workspace").inert, false);
        assert.equal(h.get("sidebar").classList.contains("open"), false);
      },
    );
    await t.test(
      "routine actions have concise status without narration or success toasts",
      async (t) => {
        const h = await makeHarness(t);
        assert.equal(h.get("prompt").placeholder, "Message…");
        assert.equal(h.get("model-selector").hidden, true);
        assert.equal(h.get("app-settings").hidden, true);
        assert.equal(h.get("web-settings").hidden, false);
        assert.equal(h.get("empty-load").hidden, true);
        assert.notEqual(h.get("model-dialog").open, true);
        assert.equal(h.calls.length, 0);
        assert.equal(h.get("model-status").textContent, "local-model");
        h.setDraft("Hello");
        h.send();
        assert.equal(
          h.get("messages").querySelector(".pending-note").textContent,
          "Generating…",
        );
        assert.equal(h.get("elapsed").textContent, "0.0s");
        await h.respond(completion("Hello back"));
        assert.equal(h.get("toast").hidden, true);
        assert.equal(h.get("request-progress").hidden, true);
        assert.equal(h.get("request-error").hidden, true);
        assert.equal(
          h.get("messages").querySelectorAll(".branch-button")[0].textContent,
          "Edit",
        );
        assert.equal(
          h.get("messages").querySelectorAll(".branch-button")[1].textContent,
          "Regenerate",
        );
        await h.exportJSON();
        assert.equal(h.get("toast").hidden, true);
      },
    );
    await t.test(
      "model readiness, Bearer authentication and untrusted auth errors",
      async (t) => {
        const h = await makeHarness(t, { autoConnect: false });
        assert.equal(h.get("send").disabled, true);
        assert.equal(h.requests[0].options.headers.Authorization, undefined);
        assert.equal(h.requests[0].options.redirect, "error");
        await h.respond({ error: { message: "Key required" } }, 401);
        assert.match(h.get("connection-error").textContent, /API key/);
        h.setKey("test-only-secret");
        h.get("connect").click();
        assert.equal(
          h.pending()[0].options.headers.Authorization,
          "Bearer test-only-secret",
        );
        const model = '<img src=x onerror="alert(1)">';
        await h.respond({ data: [{ id: model }] });
        assert.equal(h.get("model-status").textContent, model);
        assert.equal(h.get("model-status").children.length, 0);
        h.setDraft("Authenticate this request");
        h.send();
        assert.equal(h.posts()[0].body.model, model);
        assert.equal(
          h.posts()[0].options.headers.Authorization,
          "Bearer test-only-secret",
        );
        assert.equal(h.posts()[0].options.cache, "no-store");
        await h.respond(
          { error: { message: "<script>bad key</script>" } },
          401,
        );
        assert.match(
          h.get("error-detail").textContent,
          /<script>bad key<\/script>/,
        );
        assert.equal(h.get("error-detail").children.length, 0);
        assert.equal(h.get("prompt").value, "Authenticate this request");
        assert.equal(h.messages().length, 0);
        assert.match(h.get("model-status").textContent, /API key/);
      },
    );

    await t.test(
      "missing models and a timed-out model check are recoverable",
      async (t) => {
        const h = await makeHarness(t, { autoConnect: false });
        await h.respond({ data: [] });
        assert.match(
          h.get("connection-error").textContent,
          /did not advertise a model/,
        );
        h.get("refresh-model").click();
        await h.fireTimeouts(15000);
        assert.match(h.get("connection-error").textContent, /timed out/);
        assert.equal(h.pending().length, 0);
        assert.equal(h.get("refresh-model").disabled, false);
        h.get("refresh-model").click();
        await h.respond({ data: [{ id: "recovered-model" }] });
        assert.equal(h.get("model-status").textContent, "recovered-model");
      },
    );

    await t.test(
      "concurrent sends/navigation are guarded and 429 retries are manual",
      async (t) => {
        const h = await makeHarness(t);
        h.setDraft("First prompt");
        h.send();
        h.send();
        h.get("new-chat").emit("click");
        h.get("conversation-list")
          .querySelector(".conversation-select")
          .emit("click");
        assert.equal(h.posts().length, 1);
        assert.equal(h.get("chat-count").textContent, "1");
        assert.equal(h.get("new-chat").disabled, true);
        assert.equal(h.get("prompt").readOnly, true);
        await h.respond(completion("First response"));
        assert.equal(h.messages().length, 2);
        assert.match(h.get("messages").textContent, /5 input tokens/);
        assert.match(h.get("messages").textContent, /7 output tokens/);
        h.setDraft("Keep this draft");
        h.send();
        await h.respond({ error: { message: "Busy" } }, 429);
        assert.match(h.get("error-detail").textContent, /Retry shortly/);
        assert.equal(h.get("prompt").value, "Keep this draft");
        assert.equal(h.messages().length, 2);
        await h.fireTimeouts(4500);
        assert.equal(h.posts().length, 2);
        h.get("retry-send").click();
        assert.equal(h.posts().at(-1).body.messages.length, 3);
        await h.respond(completion("Retry succeeded"));
        assert.equal(h.messages().length, 4);
        assert.equal(h.get("prompt").value, "");
        assert.equal(h.intervals.size, 0);
      },
    );

    await t.test(
      "invalid, empty, non-JSON, unavailable and network responses retain the draft",
      async (t) => {
        const h = await makeHarness(t);
        h.setDraft("Never discard this");
        for (const [response, status, raw, expected] of [
          ["<html>not JSON</html>", 200, true, /non-JSON/],
          [{ choices: [] }, 200, false, /expected completion text/],
          [completion(""), 200, false, /empty response/],
          ["<html>Unavailable</html>", 503, true, /model is unavailable/i],
        ]) {
          h.send();
          await h.respond(response, status, { raw });
          assert.match(h.get("error-detail").textContent, expected);
          assert.equal(h.get("prompt").value, "Never discard this");
          assert.equal(h.messages().length, 0);
          assert.equal(h.intervals.size, 0);
        }
        h.send();
        await h.reject(new TypeError("Network lost"));
        assert.match(
          h.get("error-title").textContent,
          /Connection interrupted/,
        );
        assert.equal(h.get("prompt").value, "Never discard this");
        h.get("retry-send").click();
        assert.equal(h.posts().at(-1).body.messages.length, 1);
        await h.respond(completion());
        assert.equal(h.messages().length, 2);
      },
    );

    await t.test(
      "stop aborts the request without committing a turn or retrying",
      async (t) => {
        const h = await makeHarness(t);
        h.setDraft("Keep after stop");
        h.send();
        const request = h.pending()[0];
        h.get("stop").click();
        await flush();
        assert.equal(request.options.signal.aborted, true);
        assert.equal(h.get("prompt").value, "Keep after stop");
        assert.match(h.get("error-title").textContent, /stopped/);
        assert.equal(h.messages().length, 0);
        assert.equal(h.pending().length, 0);
        assert.equal(h.posts().length, 1);
        assert.equal(h.intervals.size, 0);
        assert.equal(h.get("send").disabled, false);
      },
    );

    await t.test(
      "editing/regeneration preserve originals and exports never include the API key",
      async (t) => {
        const h = await makeHarness(t);
        h.setKey("private-key-must-not-be-exported");
        h.setDraft("Original first prompt");
        h.send();
        await h.respond(completion("Original first reply"));
        h.setDraft("Original second prompt");
        h.send();
        await h.respond(completion("Original second reply"));
        h.setDraft("Original unsent draft");
        const original = await h.exportJSON();
        assert.equal(original.format, "lma.conversation");
        assert.equal(original.version, 1);
        assert.equal(original.messages.length, 4);
        assert.equal(original.draft.content, "Original unsent draft");
        assert.ok(
          !JSON.stringify(original).includes(
            "private-key-must-not-be-exported",
          ),
        );
        const selectOriginal = () =>
          h
            .get("conversation-list")
            .querySelectorAll(".conversation-select")
            .find((node) => node.title === original.title)
            .click();
        h.get("messages").querySelectorAll(".branch-button")[0].click();
        assert.equal(h.get("chat-count").textContent, "2");
        assert.equal(h.get("prompt").value, "Original first prompt");
        assert.equal(h.messages().length, 0);
        h.setDraft("Edited first prompt");
        h.send();
        await h.respond(completion("Edited answer"));
        selectOriginal();
        const afterEdit = await h.exportJSON();
        assert.deepEqual(afterEdit.messages, original.messages);
        assert.deepEqual(afterEdit.draft, original.draft);
        h.get("messages").querySelectorAll(".branch-button").at(-1).click();
        assert.equal(h.get("chat-count").textContent, "3");
        assert.equal(h.posts().at(-1).body.messages.length, 3);
        assert.equal(
          h.posts().at(-1).body.messages.at(-1).content,
          "Original second prompt",
        );
        await h.respond(completion("Regenerated answer"));
        const regenerated = await h.exportJSON();
        assert.equal(regenerated.messages.at(-1).content, "Regenerated answer");
        selectOriginal();
        assert.deepEqual((await h.exportJSON()).messages, original.messages);
        h.get("export-markdown").click();
        const download = h.document.downloads.at(-1);
        assert.match(download.name, /\.md$/);
        const markdown = await h.blobs.get(download.url).text();
        assert.match(markdown, /Original first reply/);
        assert.ok(!markdown.includes("private-key-must-not-be-exported"));
        await h.fireTimeouts(60000);
        assert.equal(h.blobs.size, 0, "download URLs should be revoked");
      },
    );

    await t.test(
      "raw mode starts separate history and sends only the composer text",
      async (t) => {
        const h = await makeHarness(t);
        h.setDraft("Prior chat prompt");
        h.send();
        await h.respond(completion("Prior chat response"));
        const oldTitle = h.get("conversation-title").textContent;
        h.get("open-settings").click();
        h.get("input-mode").value = "raw";
        h.get("input-mode").emit("change");
        h.get("system-prompt").value = "MUST NOT SEND";
        h.get("coding-mode").checked = true;
        h.get("settings-form").emit("submit");
        assert.equal(h.get("chat-count").textContent, "2");
        assert.equal(h.get("attach-button").disabled, true);
        assert.equal(h.messages().length, 0);
        h.setDraft("  exact raw prompt\n");
        h.send();
        assert.equal(h.posts().at(-1).url, "/v1/completions");
        assert.deepEqual(h.posts().at(-1).body, {
          model: "local-model",
          max_tokens: 128,
          temperature: 0.8,
          top_p: 0.95,
          stream: false,
          prompt: "  exact raw prompt\n",
        });
        await h.respond({ choices: [{ text: "Raw result" }] });
        assert.match(h.get("messages").textContent, /Raw result/);
        h.get("conversation-list")
          .querySelectorAll(".conversation-select")
          .find((node) => node.title === oldTitle)
          .click();
        assert.equal(h.messages().length, 2);
        assert.match(h.get("messages").textContent, /Prior chat response/);
      },
    );

    await t.test(
      "image preparation locks send, retains images on error and budgets history",
      async (t) => {
        const h = await makeHarness(t);
        const file = {
          name: "screenshot.png",
          type: "image/png",
          size: pngSignature.length,
        };
        h.setDraft("Inspect screenshot");
        h.get("image-input").files = [file];
        h.get("image-input").emit("change");
        assert.equal(h.get("send").disabled, true);
        assert.equal(h.get("new-chat").disabled, true);
        h.send();
        h.get("new-chat").emit("click");
        assert.equal(h.posts().length, 0);
        assert.equal(h.get("chat-count").textContent, "1");
        h.readers[0].finish(`data:image/png;base64,${pngBase64}`);
        await flush();
        assert.equal(h.get("attachments").children.length, 1);
        assert.equal(h.get("send").disabled, false);
        h.send();
        assert.deepEqual(h.posts()[0].body.messages[0].images, [pngBase64]);
        await h.respond({ error: { message: "context too small" } }, 400);
        assert.equal(h.get("attachments").children.length, 1);
        assert.equal(h.get("prompt").value, "Inspect screenshot");
        assert.match(h.get("error-detail").textContent, /context too small/);
        h.get("retry-send").click();
        await h.respond(completion(""));
        assert.equal(h.get("attachments").children.length, 1);
        h.get("retry-send").click();
        await h.respond(completion("Screenshot description"));
        assert.equal(h.get("attachments").children.length, 0);
        const exported = await h.exportJSON();
        assert.equal(exported.messages[0].images[0].base64, pngBase64);
        h.get("image-input").files = Array(8).fill(file);
        h.get("image-input").emit("change");
        await flush();
        assert.match(
          h.get("error-detail").textContent,
          /including conversation history/,
        );
        assert.equal(
          h.readers.length,
          1,
          "over-budget files must fail before FileReader allocation",
        );
      },
    );

    await t.test(
      "multi-file preparation is atomic on MIME mismatch or read failure",
      async (t) => {
        const h = await makeHarness(t);
        const file = { name: "file.png", type: "image/png", size: 8 };
        h.get("image-input").files = [file, file];
        h.get("image-input").emit("change");
        h.readers[0].finish(`data:image/png;base64,${pngBase64}`);
        await flush();
        assert.equal(h.get("send").disabled, true);
        h.readers[1].finish(`data:image/png;base64,${jpegBase64}`);
        await flush();
        assert.match(h.get("error-detail").textContent, /MIME/);
        assert.equal(h.get("attachments").children.length, 0);
        h.get("image-input").files = [file];
        h.get("image-input").emit("change");
        h.readers[2].fail();
        await flush();
        assert.match(h.get("error-detail").textContent, /Could not read/);
        assert.equal(h.get("attachments").children.length, 0);
        assert.equal(h.get("attach-button").disabled, false);
      },
    );

    await t.test(
      "Markdown treats HTML/code as data, allows safe links and supports copy/download",
      async (t) => {
        const h = await makeHarness(t);
        const code =
          '<script>alert("never execute")</script>\n<img src=x onerror=alert(1)>';
        const fragment = renderMarkdown(
          [
            '<img src="https://evil.invalid/tracker" onerror="alert(1)">',
            "**Strong** and `inline code`",
            "",
            "```html",
            code,
            "```",
            "",
            "[safe](https://example.com/docs) [bad](javascript:alert) [data](data:text/html,evil)",
            "![remote image](https://evil.invalid/tracker)",
            "",
            "- One",
            "- Two",
            "",
            "| Name | Value |",
            "| --- | --- |",
            "| safe | `text` |",
          ].join("\n"),
        );
        assert.equal(
          fragment.querySelectorAll("script, img, iframe, object").length,
          0,
        );
        assert.match(fragment.textContent, /onerror/);
        assert.equal(fragment.querySelectorAll("strong").length, 1);
        assert.equal(fragment.querySelectorAll("li").length, 2);
        assert.equal(fragment.querySelectorAll("table").length, 1);
        const [link] = fragment.querySelectorAll("a");
        assert.equal(fragment.querySelectorAll("a").length, 1);
        assert.equal(link.href, "https://example.com/docs");
        assert.equal(link.target, "_blank");
        assert.equal(link.rel, "noopener noreferrer");
        assert.equal(link.referrerPolicy, "no-referrer");
        const block = fragment.querySelector(".code-block");
        assert.equal(block.querySelector("code").textContent, code);
        const [copy, download] = block.querySelectorAll("button");
        copy.click();
        await flush();
        assert.deepEqual(h.clipboard, [code]);
        assert.equal(copy.getAttribute("aria-label"), "Copied");
        assert.equal(h.get("toast").hidden, true);
        await h.fireTimeouts(1800);
        assert.equal(copy.getAttribute("aria-label"), "Copy");
        download.click();
        const saved = h.document.downloads.at(-1);
        assert.equal(saved.name, "snippet.html");
        assert.equal(await h.blobs.get(saved.url).text(), code);
        const large = "<script>still text</script>".repeat(11000);
        assert.equal(renderMarkdown(large).textContent, large);
        const wideTable = `${Array(31).fill("Header").join("|")}\n${Array(31).fill("---").join("|")}\n${Array(31).fill("Cell").join("|")}`;
        assert.equal(
          renderMarkdown(wideTable).textContent,
          wideTable,
          "wide table fallback must not discard content",
        );
      },
    );
  },
);
