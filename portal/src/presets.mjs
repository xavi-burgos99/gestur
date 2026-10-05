export function presetName(value) {
  return String(value ?? "")
    .trim()
    .normalize("NFC");
}

export function presetNameError(value) {
  const name = presetName(value);
  if (!name) return "Escribe un nombre para el preset.";
  if (name.length > 80) return "El nombre no puede superar 80 caracteres.";
  if (/[\x00-\x1f\x7f-\x9f]/u.test(name))
    return "El nombre contiene caracteres no válidos.";
  return "";
}

export function presetParameters(config) {
  return structuredClone({
    tracking: config.tracking,
    render: config.render,
    controls: config.controls,
  });
}

export function presetSaveBody(name, draft) {
  const error = presetNameError(name);
  if (error) throw new Error(error);
  return { name: presetName(name), parameters: presetParameters(draft) };
}

export function parametersEqual(first, second) {
  return ["tracking", "render", "controls"].every(
    (group) => JSON.stringify(first[group]) === JSON.stringify(second[group]),
  );
}
