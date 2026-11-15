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
  /** 値の候補 (候補以外も書ける. モデルの名前など) */
  suggestions?: unknown[] | null
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
  /** 部分モデル (保存した定義を写したもの. ノードの unit に sub.<名前> と書いて使う) */
  subgraphs?: Record<string, Subgraph>
  ui?: { outputs?: Record<string, Ui>; [key: string]: unknown }
}

/** 部分モデルの定義 (版と部分モデルを除いた, 普通の定義と同じ形) */
export type Subgraph = Omit<Spec, 'format_version' | 'subgraphs'>

/** 部分モデルの部品の名前の頭 */
export const SUBGRAPH_PREFIX = 'sub.'

/**
 * 定義の部分モデルを部品にした一覧 (ML-ModelControl の describe_spec_units と
 * 同じ口. 入力は全て必須の口, 出力は出力の口. 中の誤りは確認の結果で示す)
 */
export function subgraphUnits(spec: Spec): UnitSpec[] {
  return Object.entries(spec.subgraphs ?? {}).map(([name, sub]) => ({
    name: `${SUBGRAPH_PREFIX}${name}`,
    category: 'sub',
    source: 'subgraph',
    description: sub.description || subgraphSummary(sub),
    params: [],
    inputs: Object.entries(sub.inputs ?? {}).map(([port, input]) => ({
      name: port,
      description: input?.description ?? '',
      optional: false,
      variadic: false,
    })),
    outputs: Object.keys(sub.outputs ?? {}).map((port) => ({
      name: port,
      description: '',
      optional: false,
      variadic: false,
    })),
    forward_params: [],
    extra_params: false,
    available: true,
    reason: '',
  }))
}

/** 説明の無い部分モデルの説明 (入力, 出力, ノードの数) */
export function subgraphSummary(sub: Subgraph): string {
  const inputs = Object.keys(sub.inputs ?? {}).join(', ')
  const outputs = Object.keys(sub.outputs ?? {}).join(', ')
  const count = Object.keys(sub.nodes ?? {}).length
  return `入力 ${inputs} → 出力 ${outputs} (ノード ${count} 個)`
}

// Python の予約語 (部分モデルの名前に使えない)
const PY_KEYWORDS = new Set(
  (
    'False None True and as assert async await break class continue def del elif else ' +
    'except finally for from global if import in is lambda nonlocal not or pass raise ' +
    'return try while with yield'
  ).split(' '),
)

/** 書き出すコードの部分モデルのクラスの名前 (ML-ModelControl の class_name_of と同じ) */
function className(name: string): string {
  const joined = name
    .split('_')
    .filter(Boolean)
    .map((part) => part[0]!.toUpperCase() + part.slice(1))
    .join('')
  return /^[A-Za-z_]/.test(joined) && !PY_KEYWORDS.has(joined) ? joined : `Subgraph${joined}`
}

/** 部分モデルの名前に使える形にした名前 (保存した定義の名前から) */
function subgraphStem(name: string): string {
  const stem = name.replace(/[^A-Za-z0-9_]/g, '_').replace(/^([0-9])/, '_$1') || 'block'
  return PY_KEYWORDS.has(stem) ? `${stem}_` : stem
}

/** 順に依らずに 2 つの値が同じか (辞書のキーの順は比べない) */
function sameValue(a: unknown, b: unknown): boolean {
  if (Array.isArray(a) || Array.isArray(b)) {
    return (
      Array.isArray(a) &&
      Array.isArray(b) &&
      a.length === b.length &&
      a.every((item, index) => sameValue(item, b[index]))
    )
  }
  if (a && b && typeof a === 'object' && typeof b === 'object') {
    const left = Object.keys(a)
    const right = Object.keys(b)
    return (
      left.length === right.length &&
      left.every(
        (key) =>
          key in b &&
          sameValue((a as Record<string, unknown>)[key], (b as Record<string, unknown>)[key]),
      )
    )
  }
  return Object.is(a, b)
}

/** 部分モデルの中の sub.<名前> を付け直した部分モデル */
function renameUnits(graph: Subgraph, renames: Map<string, string>): Subgraph {
  if (renames.size === 0) return graph
  return {
    ...graph,
    nodes: Object.fromEntries(
      Object.entries(graph.nodes ?? {}).map(([key, node]) => {
        const target = node.unit.startsWith(SUBGRAPH_PREFIX)
          ? renames.get(node.unit.slice(SUBGRAPH_PREFIX.length))
          : undefined
        return [key, target ? { ...node, unit: `${SUBGRAPH_PREFIX}${target}` } : node]
      }),
    ),
  }
}

/**
 * 別の定義を部分モデルとして写した定義と, 付けた名前を返す
 *
 * 写す定義の部分モデルも一緒に写す. 同じ名前で中身の同じものは 1 つに
 * まとめ, 中身の違うものは番号を付けた名前にする (写したものの中の
 * sub.<名前> も付け直す). 書き出すコードのクラスの名前 (本体の Model と
 * 他の部分モデル) と重なる名前も番号を付ける
 */
export function embedSubgraph(
  spec: Spec,
  name: string,
  source: Spec,
): { spec: Spec; name: string } {
  const subgraphs = { ...(spec.subgraphs ?? {}) }
  const { format_version: _version, subgraphs: nested = {}, ...body } = structuredClone(source)
  void _version

  // 使える名前 (同じ中身のものと同じ名前か, 空いていてクラスの名前も重ならない名前)
  const fits = (candidate: string, value: Subgraph) => {
    if (candidate in subgraphs) return sameValue(subgraphs[candidate], value)
    const classes = new Set(['Model', ...Object.keys(subgraphs).map(className)])
    return !classes.has(className(candidate))
  }
  const pick = (base: string, value: Subgraph) => {
    let used = base
    for (let index = 2; !fits(used, value); index += 1) used = `${base}_${index}`
    return used
  }

  // 写す定義の部分モデルの名前を決めてから, 中の参照を付け直して写す
  const renames = new Map<string, string>()
  for (const [key, value] of Object.entries(nested)) {
    const used = pick(key, value)
    if (used !== key) renames.set(key, used)
    subgraphs[used] = value
  }
  for (const [key, value] of Object.entries(nested)) {
    subgraphs[renames.get(key) ?? key] = renameUnits(value, renames)
  }
  const own = renameUnits(body, renames)
  const used = pick(subgraphStem(name), own)
  subgraphs[used] = own
  return { spec: { ...spec, subgraphs }, name: used }
}

/**
 * 部分モデルを 1 つの定義として編集するための定義 (版と, 中で使う部分モデルを
 * 引くための subgraphs を付けたもの)
 */
export function subgraphView(document: Spec, name: string): Spec {
  const sub = document.subgraphs?.[name]
  return {
    format_version: document.format_version,
    ...sub,
    inputs: sub?.inputs ?? {},
    nodes: sub?.nodes ?? {},
    outputs: sub?.outputs ?? {},
    subgraphs: document.subgraphs,
  }
}

/** 定義 (本体か部分モデル) のノードが直接使う部分モデルの名前 */
function directSubgraphs(graph: { nodes?: Record<string, NodeDef> } | undefined): string[] {
  return Object.values(graph?.nodes ?? {})
    .map((node) => node.unit)
    .filter((unit) => unit.startsWith(SUBGRAPH_PREFIX))
    .map((unit) => unit.slice(SUBGRAPH_PREFIX.length))
}

/** 部分モデル start から (中の部分モデルを通って) たどれる部分モデルの名前 (start を含む) */
export function reachableSubgraphs(subgraphs: Spec['subgraphs'], start: string[]): Set<string> {
  const found = new Set<string>()
  const stack = [...start]
  while (stack.length > 0) {
    const name = stack.pop()!
    if (found.has(name) || !subgraphs?.[name]) continue
    found.add(name)
    stack.push(...directSubgraphs(subgraphs[name]))
  }
  return found
}

/**
 * 部分モデル editing の中に置くと循環になる部品か (editing 自身と, 中で
 * editing を使う部分モデル)
 */
export function makesCycle(subgraphs: Spec['subgraphs'], editing: string, unit: string): boolean {
  if (!unit.startsWith(SUBGRAPH_PREFIX)) return false
  return reachableSubgraphs(subgraphs, [unit.slice(SUBGRAPH_PREFIX.length)]).has(editing)
}

/**
 * 部分モデルを 1 つの定義として確かめるための定義 (自分自身を除き, 中で
 * 使う部分モデルだけを subgraphs に持つ)
 */
export function subgraphCheckTarget(view: Spec, editing: string): Spec {
  const used = reachableSubgraphs(view.subgraphs, directSubgraphs(view))
  used.delete(editing)
  const subgraphs = Object.fromEntries(
    Object.entries(view.subgraphs ?? {}).filter(([key]) => used.has(key)),
  )
  const { subgraphs: _all, ...rest } = view
  void _all
  return Object.keys(subgraphs).length > 0 ? { ...rest, subgraphs } : rest
}

/** 名前の部分モデルを除いた subgraphs (空なら undefined) */
export function withoutSubgraph(subgraphs: Spec['subgraphs'], name: string): Spec['subgraphs'] {
  const rest = Object.fromEntries(Object.entries(subgraphs ?? {}).filter(([key]) => key !== name))
  return Object.keys(rest).length > 0 ? rest : undefined
}

/** 編集した部分モデルの定義を, 元の定義の subgraphs に書き戻した定義 */
export function writeSubgraph(document: Spec, name: string, view: Spec): Spec {
  const { format_version: _version, subgraphs: _subgraphs, ...body } = view
  void _version
  void _subgraphs
  return { ...document, subgraphs: { ...(document.subgraphs ?? {}), [name]: body } }
}

/** どのノードからも使われない部分モデルを消した定義を返す */
export function removeSubgraph(spec: Spec, name: string): Spec {
  const subgraphs = Object.fromEntries(
    Object.entries(spec.subgraphs ?? {}).filter(([key]) => key !== name),
  )
  const { subgraphs: _old, ...rest } = spec
  void _old
  return Object.keys(subgraphs).length === 0 ? rest : { ...rest, subgraphs }
}

/** 部分モデルを使っているもの (本体のノードと, それを使う他の部分モデルの名前) */
export function subgraphUsers(spec: Spec, name: string): { nodes: string[]; subgraphs: string[] } {
  const unit = `${SUBGRAPH_PREFIX}${name}`
  const nodes = Object.entries(spec.nodes)
    .filter(([, node]) => node.unit === unit)
    .map(([key]) => key)
  const subgraphs = Object.entries(spec.subgraphs ?? {})
    .filter(
      ([key, sub]) =>
        key !== name && Object.values(sub.nodes ?? {}).some((node) => node.unit === unit),
    )
    .map(([key]) => key)
  return { nodes, subgraphs }
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
    // 省略形は, 必須の口がちょうど 1 つの部品はその口, 必須の口が無く可変の
    // 数の口だけの部品 (op.Add など) はその口 (ML-ModelControl と同じ規則)
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
    // 出力の名前と位置は残す (つなぎ直せるように, 画面の情報に名前を追加)
    const uiOutputs = { ...(spec.ui?.outputs ?? {}) }
    uiOutputs[output] ??= {}
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
  // 長すぎる名前の部品は短い頭にする (Hugging Face の種類, 名前で作るモデル)
  const short = base.startsWith('hf.') ? 'hf' : base.endsWith('.Model') ? base.split('.')[0]! : base
  const stem =
    short
      .split('.')
      .pop()!
      .replace(/[^A-Za-z0-9_]/g, '_')
      .replace(/^([0-9])/, '_$1')
      .toLowerCase() || 'node'
  const used = new Set([...Object.keys(spec.inputs), ...Object.keys(spec.nodes)])
  // 数字で終わる名前 (myblock_2 など) は, 番号と区別できるように _ を挟む
  const head = /[0-9]$/.test(stem) ? `${stem}_` : stem
  for (let index = 1; ; index += 1) {
    const name = `${head}${index}`
    if (!used.has(name)) return name
  }
}

/** 部品のノードを加えた定義を返す (必須の引数は既定値の無い空欄のまま) */
export function addNode(spec: Spec, unit: UnitSpec, position: { x: number; y: number }): Spec {
  const name = freshName(spec, unit.name)
  const node: NodeDef = { unit: unit.name, params: {}, inputs: {}, ui: { ...position } }
  return { ...spec, nodes: { ...spec.nodes, [name]: node } }
}

/**
 * ノードの写しを置いた定義と, 写しの名前を返す
 *
 * 部品, 引数, 呼び出しの設定, 説明を写し, つなぎ先は写さない (別の所で
 * つなぐことが多いため). 名前は部品から新しく付ける
 */
export function pasteNode(
  spec: Spec,
  source: NodeDef,
  position: { x: number; y: number },
): { spec: Spec; name: string } {
  const name = freshName(spec, source.unit)
  const copy = structuredClone(source)
  const node: NodeDef = { ...copy, inputs: {}, ui: { ...copy.ui, ...position } }
  return { spec: { ...spec, nodes: { ...spec.nodes, [name]: node } }, name }
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
  // 消したものを参照するつなぎ先を, 全ての口と出力から 1 度に外す
  const refers = (ref: string) => parseRef(ref).name === name
  const cleaned = Object.fromEntries(
    Object.entries(nodes).map(([key, node]) => {
      const current = nodeInputs(node, units.get(node.unit))
      const variadic = (port: string) =>
        units.get(node.unit)?.inputs.find((item) => item.name === port)?.variadic
      const kept: Record<string, string | string[]> = {}
      for (const [port, refs] of Object.entries(current)) {
        const left = refs.filter((ref) => !refers(ref))
        if (left.length > 0) kept[port] = variadic(port) ? left : left[0]!
      }
      return [key, node.inputs === undefined ? node : { ...node, inputs: kept }]
    }),
  )
  const outputs = Object.fromEntries(Object.entries(spec.outputs).filter(([, ref]) => !refers(ref)))
  const uiOutputs = { ...(spec.ui?.outputs ?? {}) }
  for (const output of Object.keys(spec.outputs)) {
    if (!(output in outputs)) uiOutputs[output] ??= {}
  }
  return { ...spec, nodes: cleaned, inputs, outputs, ui: { ...spec.ui, outputs: uiOutputs } }
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

/** つなぎ先に選べるもの (入力と, 他のノードの出力の口) */
export function refOptions(
  spec: Spec,
  units: Map<string, UnitSpec>,
  exclude: string | null,
): { value: string; label: string }[] {
  // 新しく足したノードほどつなぐことが多いので, ノードは新しい順, 入力は最後
  // 選んだノード自身とその先のノード (つなぐと循環になる) は除く
  // 既につないであるものは (循環していても) 残す
  const below = exclude !== null ? downstream(spec, units, exclude) : new Set<string>()
  const node = exclude !== null ? spec.nodes[exclude] : undefined
  for (const refs of Object.values(node ? nodeInputs(node, units.get(node.unit)) : {})) {
    for (const ref of refs) below.delete(parseRef(ref).name)
  }
  const options: { value: string; label: string }[] = []
  for (const [name, node] of Object.entries(spec.nodes).reverse()) {
    if (name === exclude || below.has(name)) continue
    const outputs = units.get(node.unit)?.outputs ?? []
    if (outputs.length <= 1) {
      options.push({ value: name, label: `${name} (${node.unit})` })
    } else {
      for (const port of outputs) {
        options.push({ value: `${name}:${port.name}`, label: `${name}:${port.name}` })
      }
    }
  }
  for (const name of Object.keys(spec.inputs)) {
    options.push({ value: name, label: `${name} (入力)` })
  }
  return options
}

/** ノードの先にある (そのノードの出力を使う) ノードの名前 */
export function downstream(spec: Spec, units: Map<string, UnitSpec>, name: string): Set<string> {
  const next = new Map<string, string[]>()
  for (const wire of specWires(spec, units)) {
    const list = next.get(wire.source)
    if (list) list.push(wire.target)
    else next.set(wire.source, [wire.target])
  }
  const found = new Set<string>()
  const stack = [...(next.get(name) ?? [])]
  while (stack.length > 0) {
    const item = stack.pop()!
    if (found.has(item) || item === name) continue
    found.add(item)
    stack.push(...(next.get(item) ?? []))
  }
  return found
}

/** ノードの 1 つの口のつなぎ先を置き換えた定義を返す (空のリストは未接続) */
export function setPortRefs(
  spec: Spec,
  units: Map<string, UnitSpec>,
  name: string,
  port: string,
  refs: string[],
): Spec {
  const node = spec.nodes[name]
  if (!node) return spec
  const unit = units.get(node.unit)
  const current = nodeInputs(node, unit)
  const inputs: Record<string, string | string[]> = {}
  for (const [key, values] of Object.entries({ ...current, [port]: refs })) {
    if (values.length === 0) continue
    const variadic = unit?.inputs.find((item) => item.name === key)?.variadic
    inputs[key] = variadic ? values : values[0]!
  }
  return { ...spec, nodes: { ...spec.nodes, [name]: { ...node, inputs } } }
}
