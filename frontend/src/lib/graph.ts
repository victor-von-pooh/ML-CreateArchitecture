// モデルの定義 (ML-ModelControl の YAML と同じ形) と, キャンバスのノードと線の変換

/** 部品の引数の定義 (ML-ModelControl の ParamSpec) */
export interface ParamSpec {
  name: string
  kind: string
  default: unknown
  required: boolean
  description: string
  minimum: number | null
  maximum: number | null
  choices: unknown[] | null
}

/** 部品の口の定義 (ML-ModelControl の PortSpec) */
export interface PortSpec {
  name: string
  description: string
  optional: boolean
  variadic: boolean
}

/** 部品の定義 (ML-ModelControl の describe_units の 1 件) */
export interface UnitSpec {
  name: string
  category: string
  source: string
  description: string
  params: ParamSpec[]
  inputs: PortSpec[]
  outputs: PortSpec[]
  forward_params: ParamSpec[]
  extra_params: boolean
  available: boolean
  reason: string
}

/** 位置などの画面の情報 */
export interface Ui {
  x?: number
  y?: number
  [key: string]: unknown
}

export interface InputDef {
  shape?: number[]
  dtype?: string
  fill?: string
  description?: string
  ui?: Ui
}

export interface NodeDef {
  unit: string
  params?: Record<string, unknown>
  forward_params?: Record<string, unknown>
  inputs?: Record<string, string | string[]> | string
  description?: string
  ui?: Ui
}

/** モデルの定義 */
export interface Spec {
  format_version: number
  name?: string
  description?: string
  inputs: Record<string, InputDef>
  nodes: Record<string, NodeDef>
  outputs: Record<string, string>
  ui?: { outputs?: Record<string, Ui>; [key: string]: unknown }
}

/** キャンバスの線 (どのノードのどの口から, どのノードのどの口へ) */
export interface Wire {
  id: string
  source: string
  sourcePort: string
  target: string
  targetPort: string
}

/** 空のモデルの定義 */
export function emptySpec(name = 'model'): Spec {
  return {
    format_version: 1,
    name,
    inputs: { x: { shape: [4, 16], ui: { x: 40, y: 160 } } },
    nodes: {},
    outputs: {},
    ui: { outputs: {} },
  }
}

/** つなぎ先の文字 (名前か 名前:口) を, 名前と口に分ける */
export function parseRef(ref: string): { name: string; port: string | null } {
  const index = ref.indexOf(':')
  if (index < 0) return { name: ref, port: null }
  return { name: ref.slice(0, index), port: ref.slice(index + 1) }
}

/** ノードの inputs を, 省略形も含めて口ごとのつなぎ先のリストにする */
export function nodeInputs(node: NodeDef, unit: UnitSpec | undefined): Record<string, string[]> {
  const inputs = node.inputs
  if (inputs === undefined) return {}
  if (typeof inputs === 'string') {
    // 省略形は必須の口がちょうど 1 つの部品だけ (ML-ModelControl と同じ規則)
    const required = (unit?.inputs ?? []).filter((port) => !port.optional && !port.variadic)
    const port = required.length === 1 ? required[0]!.name : (unit?.inputs[0]?.name ?? 'input')
    return { [port]: [inputs] }
  }
  return Object.fromEntries(
    Object.entries(inputs).map(([port, value]) => [port, Array.isArray(value) ? value : [value]]),
  )
}

/** 出力の口の名前 (参照の口が無い場合は最初の口) */
function sourcePortOf(ref: string, spec: Spec, units: Map<string, UnitSpec>): string {
  const { name, port } = parseRef(ref)
  if (name in spec.inputs) return 'out'
  if (port) return port
  const unit = units.get(spec.nodes[name]?.unit ?? '')
  return unit?.outputs[0]?.name ?? 'out'
}

/** 定義のつなぎ先から, キャンバスの線の一覧を作る */
export function specWires(spec: Spec, units: Map<string, UnitSpec>): Wire[] {
  const wires: Wire[] = []
  for (const [name, node] of Object.entries(spec.nodes)) {
    const unit = units.get(node.unit)
    for (const [port, refs] of Object.entries(nodeInputs(node, unit))) {
      refs.forEach((ref, index) => {
        const source = parseRef(ref).name
        const sourcePort = sourcePortOf(ref, spec, units)
        wires.push({
          id: `${source}:${sourcePort}->${name}:${port}:${index}`,
          source,
          sourcePort,
          target: name,
          targetPort: port,
        })
      })
    }
  }
  for (const [name, ref] of Object.entries(spec.outputs)) {
    const source = parseRef(ref).name
    const sourcePort = sourcePortOf(ref, spec, units)
    wires.push({
      id: `${source}:${sourcePort}->output:${name}`,
      source,
      sourcePort,
      target: `output:${name}`,
      targetPort: 'in',
    })
  }
  return wires
}

/** 出力の口の参照の文字 (出力が 1 つの部品と入力は名前だけ) */
function refFor(spec: Spec, units: Map<string, UnitSpec>, source: string, port: string): string {
  if (source in spec.inputs) return source
  const unit = units.get(spec.nodes[source]?.unit ?? '')
  if (unit && unit.outputs.length <= 1) return source
  return `${source}:${port}`
}

/** 線を 1 本加えた定義を返す (可変の数の口は後ろに加え, それ以外は置き換え) */
export function connect(spec: Spec, units: Map<string, UnitSpec>, wire: Omit<Wire, 'id'>): Spec {
  const ref = refFor(spec, units, wire.source, wire.sourcePort)
  if (wire.target.startsWith('output:')) {
    const output = wire.target.slice('output:'.length)
    return { ...spec, outputs: { ...spec.outputs, [output]: ref } }
  }
  const node = spec.nodes[wire.target]
  if (!node) return spec
  const unit = units.get(node.unit)
  const port = unit?.inputs.find((item) => item.name === wire.targetPort)
  const current = nodeInputs(node, unit)
  const refs = port?.variadic ? [...(current[wire.targetPort] ?? []), ref] : [ref]
  const inputs: Record<string, string | string[]> = {}
  for (const [name, values] of Object.entries({ ...current, [wire.targetPort]: refs })) {
    const variadic = unit?.inputs.find((item) => item.name === name)?.variadic
    inputs[name] = variadic ? values : values[0]!
  }
  return { ...spec, nodes: { ...spec.nodes, [wire.target]: { ...node, inputs } } }
}

/** 線を外した定義を返す */
export function disconnect(spec: Spec, units: Map<string, UnitSpec>, wireId: string): Spec {
  const wire = specWires(spec, units).find((item) => item.id === wireId)
  if (!wire) return spec
  if (wire.target.startsWith('output:')) {
    const output = wire.target.slice('output:'.length)
    const outputs = { ...spec.outputs }
    delete outputs[output]
    const uiOutputs = { ...(spec.ui?.outputs ?? {}) }
    // 出力の名前と位置は残す (つなぎ直せるように空のつなぎ先にはしない)
    return { ...spec, outputs, ui: { ...spec.ui, outputs: uiOutputs } }
  }
  const node = spec.nodes[wire.target]!
  const unit = units.get(node.unit)
  const current = nodeInputs(node, unit)
  const index = Number(wireId.split(':').pop())
  const refs = (current[wire.targetPort] ?? []).filter((_, position) => position !== index)
  const inputs: Record<string, string | string[]> = {}
  for (const [name, values] of Object.entries({ ...current, [wire.targetPort]: refs })) {
    if (values.length === 0) continue
    const variadic = unit?.inputs.find((item) => item.name === name)?.variadic
    inputs[name] = variadic ? values : values[0]!
  }
  return { ...spec, nodes: { ...spec.nodes, [wire.target]: { ...node, inputs } } }
}

/** 名前の空間 (入力とノード) で使われていない, 部品の名前から作った名前 */
export function freshName(spec: Spec, base: string): string {
  const stem =
    base
      .split('.')
      .pop()!
      .replace(/[^A-Za-z0-9_]/g, '_')
      .replace(/^([0-9])/, '_$1')
      .toLowerCase() || 'node'
  const used = new Set([...Object.keys(spec.inputs), ...Object.keys(spec.nodes)])
  for (let index = 1; ; index += 1) {
    const name = `${stem}${index}`
    if (!used.has(name)) return name
  }
}

/** 部品のノードを加えた定義を返す (必須の引数は既定値の無い空欄のまま) */
export function addNode(spec: Spec, unit: UnitSpec, position: { x: number; y: number }): Spec {
  const name = freshName(spec, unit.name)
  const node: NodeDef = { unit: unit.name, params: {}, inputs: {}, ui: { ...position } }
  return { ...spec, nodes: { ...spec.nodes, [name]: node } }
}

/** ノードか入力を消し, それを参照するつなぎ先も外した定義を返す */
export function removeItem(spec: Spec, units: Map<string, UnitSpec>, name: string): Spec {
  if (name.startsWith('output:')) {
    const output = name.slice('output:'.length)
    const outputs = { ...spec.outputs }
    delete outputs[output]
    const uiOutputs = { ...(spec.ui?.outputs ?? {}) }
    delete uiOutputs[output]
    return { ...spec, outputs, ui: { ...spec.ui, outputs: uiOutputs } }
  }
  const nodes = { ...spec.nodes }
  const inputs = { ...spec.inputs }
  delete nodes[name]
  delete inputs[name]
  let next: Spec = { ...spec, nodes, inputs }
  // 消したものを参照する線を外す
  for (const wire of specWires(next, units)) {
    if (wire.source === name) next = disconnect(next, units, wire.id)
  }
  return next
}

/** 名前を変え, それを参照するつなぎ先も書き換えた定義を返す */
export function renameItem(spec: Spec, oldName: string, newName: string): Spec {
  if (oldName === newName) return spec
  const rename = (ref: string) => {
    const { name, port } = parseRef(ref)
    if (name !== oldName) return ref
    return port ? `${newName}:${port}` : newName
  }
  const renameKey = <T>(record: Record<string, T>) =>
    Object.fromEntries(
      Object.entries(record).map(([key, value]) => [key === oldName ? newName : key, value]),
    )
  const nodes = Object.fromEntries(
    Object.entries(renameKey(spec.nodes)).map(([key, node]) => {
      const inputs = node.inputs
      let next: NodeDef['inputs'] = inputs
      if (typeof inputs === 'string') next = rename(inputs)
      else if (inputs)
        next = Object.fromEntries(
          Object.entries(inputs).map(([port, value]) => [
            port,
            Array.isArray(value) ? value.map(rename) : rename(value),
          ]),
        )
      return [key, { ...node, inputs: next }]
    }),
  )
  const outputs = Object.fromEntries(
    Object.entries(spec.outputs).map(([key, value]) => [key, rename(value)]),
  )
  return { ...spec, inputs: renameKey(spec.inputs), nodes, outputs }
}

/** 入力, ノード, 出力の名前の規則 (ML-ModelControl と同じ識別子の形) */
export function isValidName(name: string): boolean {
  return /^[A-Za-z_][A-Za-z0-9_]*$/.test(name)
}
