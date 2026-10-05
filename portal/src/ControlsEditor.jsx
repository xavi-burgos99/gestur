import React, { useEffect, useRef, useState } from "react";
import {
  Accordion,
  Alert,
  Button,
  Group,
  Modal,
  NumberInput,
  Paper,
  Select,
  SimpleGrid,
  Stack,
  Switch,
  Text,
  Title,
} from "@mantine/core";
import {
  IconArrowLeft,
  IconCheck,
  IconChevronDown,
  IconPlus,
  IconTrash,
} from "@tabler/icons-react";
import {
  CONTROL_OPTIONS,
  GESTURE_OPTIONS,
  emptyGestureSelection,
  selectionForGesture,
  gestureFromSelection,
  changeGestureSelection,
  gestureSteps,
  responseOptions,
  modeChanges,
  createControlMapping,
} from "./control-catalog.mjs";
import MotionIcon from "./MotionIcon.jsx";
import "./controls-editor.css";

const controlFor = (id) => CONTROL_OPTIONS.find((option) => option.id === id);
const gestureFor = (id) => GESTURE_OPTIONS.find((option) => option.id === id);
const gestureLabel = (option) =>
  option ? `${option.category} · ${option.label}` : "Gesto no disponible";

function Numeric({
  value,
  onChange,
  min = 0,
  max = 360,
  step = 0.1,
  ...props
}) {
  return (
    <NumberInput
      {...props}
      value={value}
      onChange={(next) => onChange(next === "" ? 0 : Number(next))}
      min={min}
      max={max}
      step={step}
      decimalScale={3}
    />
  );
}

function ChoiceCard({
  option,
  selected,
  disabled,
  onClick,
  detail,
  compact = false,
}) {
  return (
    <button
      type="button"
      className={`motion-choice${compact ? " motion-choice-compact" : ""}`}
      aria-pressed={selected}
      disabled={disabled}
      onClick={onClick}
    >
      <span className="motion-choice-art" aria-hidden="true">
        <MotionIcon {...option.icon} size={compact ? 48 : 68} />
        {selected && <IconCheck className="motion-choice-check" size={18} />}
      </span>
      <span className="motion-choice-label">{option.label}</span>
      {(detail || option.description) && (
        <span className="motion-choice-description">
          {detail || option.description}
        </span>
      )}
    </button>
  );
}

function GestureStep({ step, value, onChange }) {
  const section = useRef(null);
  const selected = step.options.find((option) => option.id === value);
  useEffect(() => {
    // The chosen card is replaced by a summary. Continue keyboard navigation
    // at the next question instead of losing focus back to the modal header.
    if (!value) section.current?.querySelector("button")?.focus();
  }, [value]);
  return (
    <section ref={section} className="gesture-step" aria-label={step.label}>
      <Text fw={600} size="sm" mb={8}>
        {step.label}
      </Text>
      {selected ? (
        <button
          type="button"
          className="gesture-step-selected"
          aria-label={`Cambiar ${step.label.toLowerCase()}: ${selected.label}`}
          onClick={() => onChange(null)}
        >
          <MotionIcon {...selected.icon} size={36} />
          <span>{selected.label}</span>
          <IconChevronDown size={17} aria-hidden="true" />
        </button>
      ) : (
        <div
          className="gesture-step-options"
          role="group"
          aria-label={step.label}
          style={{ "--choice-count": step.options.length }}
        >
          {step.options.map((option) => (
            <ChoiceCard
              key={option.id}
              option={option}
              compact
              selected={false}
              onClick={() => onChange(option.id)}
            />
          ))}
        </div>
      )}
    </section>
  );
}

function TrackingNotice({ input, tracking }) {
  if (!input) return null;
  if (input.includes("hand") && !tracking.use_hands)
    return (
      <Alert color="yellow">
        Activa «Reconocer las manos» para utilizar este gesto.
      </Alert>
    );
  if (
    (input.startsWith("head") || input.startsWith("torso")) &&
    !tracking.use_pose
  )
    return (
      <Alert color="yellow">
        Activa «Seguir cabeza y cuerpo» para utilizar este gesto.
      </Alert>
    );
  return null;
}

export default function ControlsEditor({ draft, update, mapping }) {
  const controls = draft.controls.mappings;
  const [dialog, setDialog] = useState(null);
  const [selectedOutput, setSelectedOutput] = useState(null);
  const [selection, setSelection] = useState(emptyGestureSelection);
  const selectedGesture = gestureFromSelection(selection);
  const [opened, setOpened] = useState([]);
  const editingIndex =
    dialog?.kind === "gesture"
      ? controls.findIndex((control) => control.id === dialog.id)
      : -1;
  const selectedControl = controlFor(selectedOutput);
  const occupied = (output, exceptId) =>
    controls.some(
      (control) =>
        control.output === output && control.enabled && control.id !== exceptId,
    );
  const adding = dialog?.kind === "add";

  function closeDialog() {
    setDialog(null);
    setSelectedOutput(null);
    setSelection(emptyGestureSelection());
  }
  function addControl() {
    setSelectedOutput(null);
    setSelection(emptyGestureSelection());
    setDialog({ kind: "add" });
  }
  function editGesture(control) {
    const restored = selectionForGesture(control.input);
    setSelectedOutput(control.output);
    setSelection(restored);
    setDialog({ kind: "gesture", id: control.id });
  }
  function applySelection() {
    if (!selectedOutput || !selectedGesture) return;
    if (!adding) {
      if (editingIndex < 0) return;
      mapping(editingIndex, { input: selectedGesture });
      closeDialog();
      return;
    }
    if (controls.length >= 32 || occupied(selectedOutput)) return;
    // The access point serves HTTP, where crypto.randomUUID may be unavailable.
    const id = `control_${Array.from(crypto.getRandomValues(new Uint32Array(3)), (n) => n.toString(16)).join("_")}`;
    const next = createControlMapping(id, selectedOutput, selectedGesture);
    if (!next) return;
    update("controls", "mappings", [...controls, next]);
    setOpened((current) => [...current, id]);
    closeDialog();
  }

  return (
    <section className="controls-editor" aria-labelledby="controls-title">
      <Group justify="space-between" mb="md">
        <div>
          <Title id="controls-title" order={2}>
            Controles
          </Title>
          <Text c="dimmed" size="sm" mt={4}>
            Elige un movimiento del modelo y asígnale un gesto.
          </Text>
        </div>
        <Button
          variant="light"
          leftSection={<IconPlus size={17} />}
          disabled={controls.length >= 32}
          onClick={addControl}
        >
          Añadir control
        </Button>
      </Group>
      {controls.length === 0 && (
        <Paper p="xl" withBorder>
          <Text fw={600}>No hay controles</Text>
          <Text c="dimmed" size="sm" mt={4}>
            Añade un control para mover el modelo con gestos.
          </Text>
        </Paper>
      )}
      <Accordion
        multiple
        variant="separated"
        radius="md"
        value={opened}
        onChange={setOpened}
      >
        {controls.map((control, index) => {
          const output = controlFor(control.output);
          const gesture = gestureFor(control.input);
          const blocked = occupied(control.output, control.id);
          return (
            <Accordion.Item key={control.id} value={control.id}>
              <div className="control-heading">
                <Accordion.Control>
                  <Group gap="md" wrap="nowrap">
                    <span className="control-art" aria-hidden="true">
                      {output && <MotionIcon {...output.icon} size={44} />}
                    </span>
                    <div>
                      <Text fw={600}>{output?.label || control.output}</Text>
                      <Text size="sm" c="dimmed">
                        {gestureLabel(gesture)}
                      </Text>
                    </div>
                  </Group>
                </Accordion.Control>
                <Switch
                  className="control-enabled"
                  label={control.enabled ? "Activo" : "Desactivado"}
                  aria-label={`${output?.label || "Control"}: activar control`}
                  checked={control.enabled}
                  disabled={!control.enabled && blocked}
                  onChange={(event) => {
                    if (!event.currentTarget.checked || !blocked)
                      mapping(index, { enabled: event.currentTarget.checked });
                  }}
                />
              </div>
              <Accordion.Panel>
                <Stack pt="sm" gap="lg">
                  {blocked && !control.enabled && (
                    <Alert color="yellow">
                      Ya hay un control activo para este movimiento. Desactívalo
                      antes de activar este.
                    </Alert>
                  )}
                  <Paper className="assigned-gesture" p="md" withBorder>
                    <Group justify="space-between" wrap="wrap">
                      <Group gap="md" wrap="nowrap">
                        {gesture && (
                          <span
                            className="assigned-gesture-art"
                            aria-hidden="true"
                          >
                            <MotionIcon {...gesture.icon} size={52} />
                          </span>
                        )}
                        <div>
                          <Text size="xs" c="dimmed">
                            Gesto asignado
                          </Text>
                          <Text fw={600}>{gestureLabel(gesture)}</Text>
                          {gesture?.description && (
                            <Text size="sm" c="dimmed">
                              {gesture.description}
                            </Text>
                          )}
                        </div>
                      </Group>
                      <Button
                        variant="default"
                        onClick={() => editGesture(control)}
                      >
                        Cambiar gesto
                      </Button>
                    </Group>
                  </Paper>
                  <TrackingNotice
                    input={control.input}
                    tracking={draft.tracking}
                  />
                  <SimpleGrid cols={{ base: 1, sm: 2 }}>
                    <Select
                      label="Respuesta"
                      value={control.mode}
                      data={responseOptions(control.output)}
                      allowDeselect={false}
                      onChange={(value) =>
                        value && mapping(index, modeChanges(control, value))
                      }
                    />
                    <div className="control-invert">
                      <Switch
                        label="Invertir dirección"
                        checked={control.invert}
                        onChange={(event) =>
                          mapping(index, {
                            invert: event.currentTarget.checked,
                          })
                        }
                      />
                    </div>
                  </SimpleGrid>
                  <SimpleGrid cols={{ base: 2, md: 4 }}>
                    {control.mode !== "stepped" && (
                      <>
                        <Numeric
                          label="Intensidad"
                          value={control.scale}
                          onChange={(value) => mapping(index, { scale: value })}
                        />
                        <Numeric
                          label="Centro neutro"
                          value={control.center}
                          min={control.mode === "hybrid" ? 0.02 : 0}
                          max={control.mode === "hybrid" ? 0.98 : 1}
                          step={0.05}
                          onChange={(value) =>
                            mapping(
                              index,
                              control.mode === "hybrid"
                                ? modeChanges(
                                    { ...control, center: value },
                                    "hybrid",
                                  )
                                : { center: value },
                            )
                          }
                        />
                      </>
                    )}
                    {control.mode === "hybrid" && (
                      <>
                        <Numeric
                          label="Umbral inferior"
                          value={control.left_threshold}
                          min={0.01}
                          max={Math.max(0.01, control.center - 0.01)}
                          step={0.05}
                          onChange={(value) =>
                            mapping(index, { left_threshold: value })
                          }
                        />
                        <Numeric
                          label="Umbral superior"
                          value={control.right_threshold}
                          min={Math.min(0.99, control.center + 0.01)}
                          max={0.99}
                          step={0.05}
                          onChange={(value) =>
                            mapping(index, { right_threshold: value })
                          }
                        />
                        <Numeric
                          label="Velocidad continua"
                          suffix=" °/s"
                          value={control.continuous_speed}
                          step={1}
                          onChange={(value) =>
                            mapping(index, { continuous_speed: value })
                          }
                        />
                      </>
                    )}
                    {control.mode === "stepped" && (
                      <>
                        <Numeric
                          label="Umbral de cambio"
                          value={control.threshold}
                          max={1}
                          step={0.05}
                          onChange={(value) =>
                            mapping(index, { threshold: value })
                          }
                        />
                        <Numeric
                          label="Margen del umbral"
                          value={control.hysteresis}
                          max={0.2}
                          step={0.01}
                          onChange={(value) =>
                            mapping(index, { hysteresis: value })
                          }
                        />
                        <Numeric
                          label="Tamaño pequeño"
                          value={control.small_scale}
                          min={0.1}
                          max={5}
                          onChange={(value) =>
                            mapping(index, { small_scale: value })
                          }
                        />
                        <Numeric
                          label="Tamaño grande"
                          value={control.large_scale}
                          min={0.1}
                          max={5}
                          onChange={(value) =>
                            mapping(index, { large_scale: value })
                          }
                        />
                        <Numeric
                          label="Transición"
                          value={control.transition_ms}
                          max={5000}
                          step={50}
                          suffix=" ms"
                          onChange={(value) =>
                            mapping(index, { transition_ms: value })
                          }
                        />
                      </>
                    )}
                  </SimpleGrid>
                  <Group justify="flex-end">
                    <Button
                      variant="subtle"
                      color="red"
                      leftSection={<IconTrash size={16} />}
                      onClick={() =>
                        update(
                          "controls",
                          "mappings",
                          controls.filter((item) => item.id !== control.id),
                        )
                      }
                    >
                      Eliminar control
                    </Button>
                  </Group>
                </Stack>
              </Accordion.Panel>
            </Accordion.Item>
          );
        })}
      </Accordion>

      <Modal
        opened={!!dialog}
        onClose={closeDialog}
        title={adding ? "Añadir control" : "Cambiar gesto"}
        centered
        size="xl"
        classNames={{
          content: "control-dialog",
          body: "control-dialog-content",
        }}
        closeButtonProps={{ "aria-label": "Cerrar selector de control" }}
      >
        <Stack gap="lg" className="control-dialog-body">
          <Stack gap="lg" className="control-dialog-scroll">
            {!selectedOutput ? (
              <>
                <Text size="sm" c="dimmed">
                  Selecciona el movimiento del modelo.
                </Text>
                <div
                  className="motion-choice-grid"
                  role="group"
                  aria-label="Movimiento del modelo"
                >
                  {CONTROL_OPTIONS.map((option) => (
                    <ChoiceCard
                      key={option.id}
                      option={option}
                      selected={false}
                      disabled={occupied(option.id)}
                      detail={
                        occupied(option.id)
                          ? "Ya tiene un control activo"
                          : undefined
                      }
                      onClick={() => setSelectedOutput(option.id)}
                    />
                  ))}
                </div>
                {CONTROL_OPTIONS.some((option) => occupied(option.id)) && (
                  <Text size="xs" c="dimmed">
                    Para sustituir un control activo, cambia su gesto o
                    desactívalo antes de añadir otro.
                  </Text>
                )}
              </>
            ) : (
              <>
                <Group
                  justify="space-between"
                  className="selected-control"
                  wrap="wrap"
                >
                  <Group gap="sm" wrap="nowrap">
                    {selectedControl && (
                      <span aria-hidden="true">
                        <MotionIcon {...selectedControl.icon} size={44} />
                      </span>
                    )}
                    <div>
                      <Text size="xs" c="dimmed">
                        Control
                      </Text>
                      <Text fw={600}>{selectedControl?.label}</Text>
                    </div>
                  </Group>
                  {adding && (
                    <Button
                      variant="subtle"
                      size="xs"
                      leftSection={<IconArrowLeft size={14} />}
                      onClick={() => {
                        setSelectedOutput(null);
                        setSelection(emptyGestureSelection());
                      }}
                    >
                      Cambiar control
                    </Button>
                  )}
                </Group>
                <div className="gesture-selector">
                  {gestureSteps(selection).map((step) => (
                    <GestureStep
                      key={step.id}
                      step={step}
                      value={selection[step.id]}
                      onChange={(value) =>
                        setSelection((current) =>
                          changeGestureSelection(current, step.id, value),
                        )
                      }
                    />
                  ))}
                </div>
                <TrackingNotice
                  input={selectedGesture}
                  tracking={draft.tracking}
                />
                {adding && occupied(selectedOutput) && (
                  <Alert color="yellow">
                    Este movimiento ya tiene un control activo.
                  </Alert>
                )}
              </>
            )}
          </Stack>
          <Group justify="flex-end" className="control-dialog-actions">
            <Button variant="default" onClick={closeDialog}>
              Cancelar
            </Button>
            {selectedOutput && (
              <Button
                onClick={applySelection}
                disabled={
                  !selectedGesture ||
                  (adding
                    ? occupied(selectedOutput) || controls.length >= 32
                    : editingIndex < 0)
                }
              >
                {adding ? "Añadir control" : "Asignar gesto"}
              </Button>
            )}
          </Group>
        </Stack>
      </Modal>
    </section>
  );
}
