// 部分モデル (定義の subgraphs) と, 定義の形の推定の結果を扱う関数

import {
  SUBGRAPH_PREFIX,
  nodeInputs,
  parseRef,
  type NodeDef,
  type Spec,
  type Subgraph,
  type SubgraphOrigin,
  type Ui,
  type UnitSpec,
} from './graph'

/** ui の写した元を除いた部分モデル (中身の比較に使う) */
function withoutSource(sub: Subgraph | undefined): Subgraph | undefined {
  if (!sub?.ui || !('source' in sub.ui)) return sub
  const { source: _source, ...ui } = sub.ui
  void _source
  return { ...sub, ui }
}

/** 部分モデルを写した元 (書いていない, 形の違うものは undefined) */
export function subgraphOrigin(sub: Subgraph | undefined): SubgraphOrigin | undefined {
  const value = sub?.ui?.source as Partial<SubgraphOrigin> | undefined
  return value &&
    (value.kind === 'model' || value.kind === 'example') &&
    typeof value.name === 'string'
    ? { kind: value.kind, name: value.name }
    : undefined
}

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
  options: {
    /** 写した元 (部分モデルの ui.source に書き, 写し直しに使う) */
    origin?: SubgraphOrigin
    /** この名前の部分モデルを置き換える (元の定義からの写し直し) */
    replace?: string
  } = {},
): { spec: Spec; name: string } {
  const subgraphs = { ...(spec.subgraphs ?? {}) }
  const { format_version: _version, subgraphs: nested = {}, ...body } = structuredClone(source)
  void _version
  // 置き換える部分モデルの前の中身は, 写す定義の中の部分モデルとまとめない
  if (options.replace) delete subgraphs[options.replace]

  // 使える名前 (同じ中身のものと同じ名前か, 空いていてクラスの名前も重ならない名前)
  const fits = (candidate: string, value: Subgraph) => {
    if (candidate === options.replace) return false
    if (candidate in subgraphs)
      return sameValue(withoutSource(subgraphs[candidate]), withoutSource(value))
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
  const renamed = renameUnits(body, renames)
  const own = options.origin
    ? { ...renamed, ui: { ...renamed.ui, source: options.origin } }
    : renamed
  const used = options.replace ?? pick(subgraphStem(name), own)
  subgraphs[used] = own
  // 写した部分モデルが (中の部分モデルを通って) 自分を使う場合は写さない
  if (reachableSubgraphs(subgraphs, directSubgraphs(own)).has(used)) {
    throw new Error(
      `${name} の中で ${SUBGRAPH_PREFIX}${used} を使っているので, 同じ名前で写せません (互いに使い合うため)`,
    )
  }
  return { spec: { ...spec, subgraphs }, name: used }
}

/** 位置などの ui (入力, ノード, 出力の位置) を除いた部分モデル (中身の比較に使う) */
function withoutLayout(sub: Subgraph | undefined): unknown {
  if (!sub) return sub
  const strip = <T extends { ui?: unknown }>(items: Record<string, T> | undefined) =>
    Object.fromEntries(
      Object.entries(items ?? {}).map(([key, item]) => {
        const { ui: _ui, ...rest } = item ?? ({} as T)
        void _ui
        return [key, rest]
      }),
    )
  const { ui: _ui, ...rest } = sub
  void _ui
  return { ...rest, inputs: strip(sub.inputs), nodes: strip(sub.nodes) }
}

/** 前の部分モデルの位置 (同じ名前の入力, ノード, 出力) を写し直した部分モデルに移す */
function carryLayout(before: Subgraph | undefined, after: Subgraph): Subgraph {
  if (!before) return after
  const keep = <T extends { ui?: Ui }>(
    items: Record<string, T> | undefined,
    old: Record<string, T> | undefined,
  ) =>
    Object.fromEntries(
      Object.entries(items ?? {}).map(([key, item]) => [
        key,
        old?.[key]?.ui && !item?.ui ? { ...item, ui: old[key]!.ui } : item,
      ]),
    ) as Record<string, T>
  return {
    ...after,
    inputs: keep(after.inputs, before.inputs),
    nodes: keep(after.nodes, before.nodes),
    ui: { ...after.ui, outputs: { ...before.ui?.outputs, ...after.ui?.outputs } },
  }
}

/**
 * 部分モデル name を, 写した元の定義の今の中身で置き換えた定義
 *
 * 名前はそのまま. 前の中身の位置は同じ名前のものに移す. 前の中身だけが
 * 使っていた部分モデルで, どこからも使われなくなったものは消す. 部分モデルが
 * 無いか写した元が違う場合 (読み込みの間に消した, 別の定義を開いた) は
 * null, 中身が同じなら changed が false
 */
export function refreshSubgraph(
  spec: Spec,
  name: string,
  source: Spec,
  origin: SubgraphOrigin,
): { spec: Spec; changed: boolean } | null {
  const before = spec.subgraphs?.[name]
  const current = subgraphOrigin(before)
  if (!before || current?.kind !== origin.kind || current.name !== origin.name) return null
  const embedded = embedSubgraph(spec, origin.name, source, { origin, replace: name })
  const subgraphs = { ...embedded.spec.subgraphs }
  subgraphs[name] = carryLayout(before, subgraphs[name]!)

  // 前の中身だけが使っていて, 使われなくなった部分モデルを消す
  const candidates = reachableSubgraphs(spec.subgraphs, directSubgraphs(before))
  candidates.delete(name)
  const roots = [
    ...directSubgraphs(spec),
    ...Object.entries(subgraphs)
      .filter(([key]) => !candidates.has(key))
      .flatMap(([, sub]) => directSubgraphs(sub)),
  ]
  const reached = reachableSubgraphs(subgraphs, roots)
  for (const key of candidates) if (!reached.has(key)) delete subgraphs[key]

  const next = { ...spec, subgraphs }
  const changed =
    !sameValue(withoutLayout(before), withoutLayout(subgraphs[name])) ||
    !sameValue(Object.keys(spec.subgraphs ?? {}).sort(), Object.keys(subgraphs).sort()) ||
    Object.keys(subgraphs).some(
      (key) => key !== name && !sameValue(spec.subgraphs?.[key], subgraphs[key]),
    )
  return { spec: changed ? next : spec, changed }
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

// 入力の例に使える型 (ML-ModelControl と同じ)
const DTYPES = new Set(['float32', 'float64', 'float16', 'bfloat16', 'int64', 'int32', 'bool'])

/** 形の推定の結果のうち, ここで使う部分 (ノードごとの出力の口の形) */
export interface InferredShapes {
  nodes: Record<string, { status: string; outputs?: Record<string, unknown> }>
}

/** 部分モデルの入力の口に入る値の例の作り方 (形, 型, 値の作り方) */
export interface UsageShape {
  shape: number[]
  dtype: string
  fill: string
}

/**
 * 本体で部分モデル name を使うノードに入ってくる値の形 (口ごと)
 *
 * 部分モデルの中を, 実際に使う所の形で推定するために使う. 使うノードを計算の
 * 順 (形の推定の結果の順) に見て, 全ての入力の口の形が分かる最初のノードを
 * 使う. そのようなノードが無ければ null. 入力から来る値は入力の型と値の
 * 作り方, ノードから来る値は出力の型と 0 (値は分からないので安全な値)
 */
export function usageShapes(
  whole: Spec,
  check: InferredShapes | undefined,
  name: string,
  units: Map<string, UnitSpec>,
): { node: string; shapes: Record<string, UsageShape> } | null {
  if (!check) return null
  const unitName = `${SUBGRAPH_PREFIX}${name}`
  const unit = units.get(unitName)
  const ports = unit?.inputs.map((port) => port.name) ?? []
  const users = Object.keys(check.nodes).filter((key) => whole.nodes[key]?.unit === unitName)
  for (const node of users) {
    const refs = nodeInputs(whole.nodes[node]!, unit)
    const shapes: Record<string, UsageShape> = {}
    for (const port of ports) {
      const ref = refs[port]?.[0]
      if (ref === undefined) break
      const { name: source, port: sourcePort } = parseRef(ref)
      const input = whole.inputs[source]
      if (input) {
        if (!input.shape) break
        shapes[port] = {
          shape: input.shape,
          dtype: input.dtype ?? 'float32',
          fill: input.fill ?? 'zeros',
        }
        continue
      }
      const result = check.nodes[source]
      const value = (
        result?.status === 'ok' && result.outputs
          ? sourcePort
            ? result.outputs[sourcePort]
            : Object.values(result.outputs)[0]
          : undefined
      ) as { kind?: string; shape?: number[]; dtype?: string } | undefined
      if (value?.kind !== 'tensor' || !Array.isArray(value.shape)) break
      shapes[port] = { shape: value.shape, dtype: value.dtype ?? 'float32', fill: 'zeros' }
    }
    if (Object.keys(shapes).length === ports.length) return { node, shapes }
  }
  return null
}

/** 部分モデルを確かめる定義の入力の形を, 使う所の形に置き換えたもの */
export function withInputShapes(target: Spec, shapes: Record<string, UsageShape>): Spec {
  if (Object.keys(shapes).length === 0) return target
  return {
    ...target,
    inputs: Object.fromEntries(
      Object.entries(target.inputs).map(([port, input]) => {
        const used = shapes[port]
        if (!used) return [port, input]
        // 入力の例に使える型だけを使い, それ以外は書いた型のまま
        const dtype = DTYPES.has(used.dtype) ? used.dtype : input?.dtype
        return [port, { ...input, shape: used.shape, fill: used.fill, ...(dtype ? { dtype } : {}) }]
      }),
    ),
  }
}

/**
 * ノードの入力の口ごとに入ってくる値の形の結果 (入力はその形, ノードは推定
 * した出力の口の値). 形の分からないつなぎ先は含めない
 */
export function incomingValues(
  spec: Spec,
  check: InferredShapes | undefined,
  name: string,
  units: Map<string, UnitSpec>,
): { port: string; ref: string; value: unknown }[] {
  const node = spec.nodes[name]
  if (!node) return []
  const found: { port: string; ref: string; value: unknown }[] = []
  for (const [port, refs] of Object.entries(nodeInputs(node, units.get(node.unit)))) {
    for (const ref of refs) {
      const { name: source, port: sourcePort } = parseRef(ref)
      const input = spec.inputs[source]
      if (input) {
        if (input.shape) found.push({ port, ref, value: { kind: 'tensor', shape: input.shape } })
        continue
      }
      const result = check?.nodes[source]
      if (result?.status !== 'ok' || !result.outputs) continue
      const value = sourcePort ? result.outputs[sourcePort] : Object.values(result.outputs)[0]
      if (value !== undefined) found.push({ port, ref, value })
    }
  }
  return found
}

/**
 * ノード node が使う部分モデルを写して新しい部分モデルにし, そのノードだけが
 * 使うようにした定義と, 新しい部分モデルの名前
 *
 * 同じ部分モデルを使う他のノードは変わらない. 写したものは元の定義からの
 * 写し直しの対象にしない (別のものとして直していくため). inside は, ノードが
 * 部分モデルの中にある場合のその部分モデルの名前
 */
export function forkSubgraph(
  whole: Spec,
  node: string,
  inside: string | null = null,
): { spec: Spec; name: string } | null {
  const graph = inside ? whole.subgraphs?.[inside] : whole
  const unit = graph?.nodes?.[node]?.unit
  if (!unit?.startsWith(SUBGRAPH_PREFIX)) return null
  const source = unit.slice(SUBGRAPH_PREFIX.length)
  const original = whole.subgraphs?.[source]
  if (!original) return null
  const taken = new Set(['Model', ...Object.keys(whole.subgraphs ?? {}).map(className)])
  let name = `${source}_${node}`
  for (let index = 2; name in (whole.subgraphs ?? {}) || taken.has(className(name)); index += 1) {
    name = `${source}_${node}_${index}`
  }
  const copy = structuredClone(original)
  if (copy.ui && 'source' in copy.ui) {
    const { source: _source, ...ui } = copy.ui
    void _source
    if (Object.keys(ui).length > 0) copy.ui = ui
    else delete copy.ui
  }
  const subgraphs = { ...whole.subgraphs, [name]: copy }
  const retarget = <T extends { nodes?: Record<string, NodeDef> }>(target: T): T => ({
    ...target,
    nodes: {
      ...target.nodes,
      [node]: { ...target.nodes![node]!, unit: `${SUBGRAPH_PREFIX}${name}` },
    },
  })
  if (inside) {
    subgraphs[inside] = retarget(subgraphs[inside]!)
    return { spec: { ...whole, subgraphs }, name }
  }
  return { spec: { ...retarget(whole), subgraphs }, name }
}

/**
 * 使えない部品 (ライブラリが無いもの) と登録されていない部品のノードの, 口を
 * 定義から推し量った仮の部品
 *
 * 口が分からないと線を描けないので, 入力の口はノードの inputs の名前 (省略形は
 * input), 出力の口は他から「ノード:口」で参照される名前 (無ければ out) にする
 */
export function placeholderUnits(spec: Spec, units: Map<string, UnitSpec>): UnitSpec[] {
  const found = new Map<string, { inputs: Set<string>; outputs: Set<string> }>()
  for (const node of Object.values(spec.nodes)) {
    const known = units.get(node.unit)
    if (known && (known.available || known.inputs.length + known.outputs.length > 0)) continue
    const entry = found.get(node.unit) ?? { inputs: new Set(), outputs: new Set() }
    if (typeof node.inputs === 'string') entry.inputs.add('input')
    else Object.keys(node.inputs ?? {}).forEach((port) => entry.inputs.add(port))
    found.set(node.unit, entry)
  }
  if (found.size === 0) return []
  // 他から参照される出力の口
  const refs: string[] = []
  const collect = (value: unknown) => {
    if (typeof value === 'string') refs.push(value)
    else if (Array.isArray(value)) value.forEach(collect)
    else if (value && typeof value === 'object') Object.values(value).forEach(collect)
  }
  Object.values(spec.nodes).forEach((node) => collect(node.inputs))
  Object.values(spec.outputs).forEach(collect)
  for (const ref of refs) {
    const { name, port } = parseRef(ref)
    const unit = spec.nodes[name]?.unit
    if (unit && port && found.has(unit)) found.get(unit)!.outputs.add(port)
  }
  return [...found.entries()].map(([name, { inputs, outputs }]) => {
    const known = units.get(name)
    return {
      ...(known ?? {
        name,
        category: '',
        source: '',
        description: '',
        params: [],
        forward_params: [],
        extra_params: true,
        available: false,
        reason: '登録されていない部品です',
      }),
      inputs: [...inputs].map((port) => ({
        name: port,
        description: '',
        optional: false,
        variadic: false,
      })),
      outputs: [...(outputs.size > 0 ? outputs : ['out'])].map((port) => ({
        name: port,
        description: '',
        optional: false,
        variadic: false,
      })),
    }
  })
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
