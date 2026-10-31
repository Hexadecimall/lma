export const DEFAULT_APP_SETTINGS = Object.freeze({
  loadPolicy: "automatic",
  defaultsPath: null,
  autoRestore: true,
  gpu: "auto",
  context: 4096,
  modelDirectories: [],
  ollamaDirectory: null,
  lmStudioDirectories: [],
});

export function validateAppSettings(input) {
  if (!input || typeof input !== "object")
    throw new Error("Invalid application settings.");
  const loadPolicy = input.loadPolicy;
  if (!["automatic", "defaults", "manual"].includes(loadPolicy))
    throw new Error("Choose Automatic, Defaults file, or Manual.");
  const gpu = String(input.gpu ?? "auto")
    .trim()
    .toLowerCase();
  const context = Number(input.context);
  if (
    !/^(auto|cpu|all|\d+)$/.test(gpu) ||
    (/^\d+$/.test(gpu) && Number(gpu) > 2147483647)
  )
    throw new Error(
      "GPU layers must be auto, cpu, all, or a non-negative layer count.",
    );
  if (!Number.isInteger(context) || context < 2 || context > 4294967295)
    throw new Error("Context must be a whole number of at least 2 tokens.");
  const path = (value) => {
    if (value == null || value === "") return null;
    if (
      typeof value !== "string" ||
      value.length > 4096 ||
      /[\u0000-\u001f]/.test(value)
    )
      throw new Error("Invalid settings path.");
    return value.trim() || null;
  };
  const paths = (values) => {
    if (!Array.isArray(values) || values.length > 64)
      throw new Error("Use at most 64 custom model folders.");
    return [...new Set(values.map(path).filter(Boolean))];
  };
  if (typeof input.autoRestore !== "boolean")
    throw new Error("Invalid model restore setting.");
  const modelDirectories = paths(input.modelDirectories ?? []);
  const lmStudioDirectories = paths(input.lmStudioDirectories ?? []);
  if (modelDirectories.length + lmStudioDirectories.length > 64)
    throw new Error("Use at most 64 custom model folders.");
  return {
    loadPolicy,
    defaultsPath: path(input.defaultsPath),
    autoRestore: input.autoRestore,
    gpu,
    context,
    modelDirectories,
    ollamaDirectory: path(input.ollamaDirectory),
    lmStudioDirectories,
  };
}

export function folderLines(value) {
  return String(value)
    .split(/\r?\n/)
    .map((line) => line.trim())
    .filter(Boolean);
}

export function sourceLabel(source) {
  return source === "ollama"
    ? "Ollama"
    : source === "lm-studio"
      ? "LM Studio"
      : "lma";
}
