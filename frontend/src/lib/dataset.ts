// データセット (ML-EDA の model_inputs) の入力と出す値の数を, 定義に入れる

import { parseRef, renameItem, specWires, type InputDef, type Spec, type UnitSpec } from './graph'

/** データセットの入力 (形はバッチの次元を例の大きさにしたもの. dataset_shape は決まらない次元が null の形) */
export interface DatasetInput {
  name: string
  shape: number[]
  dtype: string
  dataset_shape?: (number | null)[]
}

/** バックエンドが ML-EDA の model_inputs から返す値 */
export interface DatasetInfo {
  path: string
  inputs: DatasetInput[]
  task?: string | null
  target?: string | null
  classes?: unknown[] | null
  /** ファイルの組の正解の種類 (mask, point_cloud など. 列の正解は null) */
  target_kind?: string | null
  outputs?: number | null
}

/** 入れた結果 (定義と, 知らせる文) */
export interface DatasetApplied {
  spec: Spec
  notes: string[]
}

/** 部品の無い定義に置く, 出す値の数を決める全結合の部品 */
const HEAD_UNIT = 'nn.Linear'

/** 形を変える部品 (最初の次元を出す値の数にする. 点群の点の数など) */
const RESHAPE_UNIT = 'op.Reshape'

/** 部品が引数を持つか (登録簿の引数の一覧で見る) */
function hasParam(units: Map<string, UnitSpec>, unit: string, name: string): boolean {
  return units.get(unit)?.params.some((param) => param.name === name) ?? false
}

/** 入力か, それを使うつなぎ先があるか */
function isUsed(spec: Spec, units: Map<string, UnitSpec>, name: string): boolean {
  return specWires(spec, units).some((wire) => wire.source === name)
}

/** 形を読み上げやすい文字にする */
function shapeText(shape: number[]): string {
  return `[${shape.join(', ')}]`
}

/**
 * 入力の形. データセットで決まらない次元 (バッチの次元を除く. 大きさの混ざる画像の高さと幅など) は,
 * 定義の入力が同じ次元の数の形を持てばその値のままにする
 */
function mergedShape(
  item: DatasetInput,
  before: InputDef | undefined,
): { shape: number[]; kept: boolean } {
  const dims = item.dataset_shape
  const current = before?.shape
  if (
    !dims ||
    !current ||
    dims.length !== item.shape.length ||
    current.length !== item.shape.length
  ) {
    return { shape: item.shape, kept: false }
  }
  let kept = false
  const shape = item.shape.map((size, index) => {
    if (index === 0 || dims[index] !== null) return size
    kept = true
    return current[index]!
  })
  return { shape, kept }
}

/** 出力の最後の形を変える部品 (op.Reshape) の最初の次元と, その前の out_features を出す値の数に合わせる */
function fitReshape(
  spec: Spec,
  units: Map<string, UnitSpec>,
  name: string,
  outputs: number,
): { spec: Spec; note: string } | null {
  const node = spec.nodes[name]
  const shape = node?.params?.shape
  if (!node || node.unit !== RESHAPE_UNIT || !Array.isArray(shape) || shape.length < 1) return null
  if (!shape.every((size) => typeof size === 'number' && Number.isInteger(size) && size > 0))
    return null
  const sources = specWires(spec, units).filter((wire) => wire.target === name)
  if (sources.length !== 1) return null
  const before = spec.nodes[sources[0]!.source]
  if (!before || !hasParam(units, before.unit, 'out_features')) return null
  const rest = (shape as number[]).slice(1)
  const features = outputs * rest.reduce((product, size) => product * size, 1)
  const reshaped = [outputs, ...rest]
  return {
    spec: {
      ...spec,
      nodes: {
        ...spec.nodes,
        [sources[0]!.source]: { ...before, params: { ...before.params, out_features: features } },
        [name]: { ...node, params: { ...node.params, shape: reshaped } },
      },
    },
    note: `${sources[0]!.source} の out_features を ${features}, ${name} の shape を ${shapeText(reshaped)} にしました`,
  }
}

/**
 * データセットの入力と出す値の数を定義に入れる
 *
 * 入力: 名前, 形, 型をデータセットに合わせる (1 つずつで名前が違えば名前を
 * 変えてつなぎ先も書き換える. 同じ名前は値の作り方と位置を残す. データセット
 * に無い入力は, つないでいなければ消し, つないでいれば残す. データセットで
 * 決まらない次元は定義の値のまま). 入力に直につながる in_features の引数は
 * 入力の最後の次元, in_channels の引数は 3 次元以上の入力の 2 番目の次元
 * (チャンネル) にする.
 * 出す値の数: 部品の無い定義で入力が 2 次元なら全結合の head を置いて出力に
 * つなぎ, 出力が 1 つでつなぎ元が out_features か out_channels を持てばそれ
 * にする. つなぎ元が op.Reshape なら最初の次元とその前の out_features にする
 */
export function applyDatasetInputs(
  spec: Spec,
  info: DatasetInfo,
  units: Map<string, UnitSpec>,
): DatasetApplied {
  const notes: string[] = []
  let next = spec
  // ノードと同じ名前の入力は作らない (つなぎ先が入力を指すように変わるため)
  const clashes = info.inputs.filter((item) => item.name in spec.nodes)
  if (clashes.length > 0) {
    const listed = clashes.map((item) => item.name).join(', ')
    notes.push(
      `ノード ${listed} がデータセットの入力と同じ名前なので, 何も変えていません. ノードの名前を変えてから入れ直してください`,
    )
    return { spec, notes }
  }
  const names = info.inputs.map((item) => item.name)

  // 1 つずつで名前が違えば, 名前を変える (つなぎ先も書き換える)
  const current = Object.keys(next.inputs)
  const only = info.inputs.length === 1 ? info.inputs[0]! : null
  if (only && current.length === 1 && current[0] !== only.name) {
    next = renameItem(next, current[0]!, only.name)
    notes.push(`入力 ${current[0]} の名前を ${only.name} にしました`)
  }

  // データセットに無い入力は, つないでいなければ消す
  const inputs: Record<string, InputDef> = {}
  for (const [name, value] of Object.entries(next.inputs)) {
    if (names.includes(name)) continue
    if (isUsed(next, units, name)) {
      inputs[name] = value
      notes.push(`入力 ${name} はデータセットに無いですが, つないでいるので残しました`)
    } else {
      notes.push(`データセットに無い入力 ${name} を消しました`)
    }
  }

  // データセットの入力の形と型 (値の作り方と位置は残す)
  const shapes = new Map<string, number[]>()
  info.inputs.forEach((item, index) => {
    const before = next.inputs[item.name]
    const { shape, kept } = mergedShape(item, before)
    shapes.set(item.name, shape)
    if (kept) {
      notes.push(
        `入力 ${item.name} のデータセットで決まらない次元は, 定義の値のままにしました (${shapeText(shape)})`,
      )
    }
    inputs[item.name] = {
      ...before,
      shape,
      dtype: item.dtype,
      ui: before?.ui ?? { x: 40, y: 160 + index * 120 },
    }
  })
  next = { ...next, inputs }

  // 入力に直につながる in_features を入力の最後の次元に, in_channels を
  // チャンネル (3 次元以上の入力の 2 番目の次元) に
  const nodes = { ...next.nodes }
  for (const wire of specWires(next, units)) {
    const shape = shapes.get(wire.source)
    const node = nodes[wire.target]
    if (!shape || !node || shape.length < 2) continue
    const fits: [string, number][] = [['in_features', shape.at(-1)!]]
    if (shape.length >= 3) fits.push(['in_channels', shape[1]!])
    for (const [param, size] of fits) {
      const current = nodes[wire.target]!
      if (!hasParam(units, current.unit, param) || current.params?.[param] === size) continue
      nodes[wire.target] = { ...current, params: { ...current.params, [param]: size } }
      notes.push(`${wire.target} の ${param} を ${size} にしました`)
    }
  }
  next = { ...next, nodes }

  // 出す値の数
  const outputs = info.outputs ?? null
  if (outputs === null) {
    notes.push('タスクを決めていないデータセットなので, 出力の大きさは変えていません')
    return { spec: next, notes }
  }
  const first = info.inputs[0]!
  const outputNames = Object.keys(next.outputs)
  if (Object.keys(next.nodes).length === 0 && outputNames.length === 0) {
    // 部品の無い定義は, 2 次元の入力に全結合の head を置いて出力 out につなぐ
    if (first.shape.length === 2 && units.get(HEAD_UNIT)?.available && !('head' in next.inputs)) {
      const at = next.inputs[first.name]?.ui
      const x = (at?.x ?? 40) + 280
      const y = at?.y ?? 160
      next = {
        ...next,
        nodes: {
          head: {
            unit: HEAD_UNIT,
            params: { in_features: first.shape[1]!, out_features: outputs },
            inputs: first.name,
            ui: { x, y },
          },
        },
        outputs: { out: 'head' },
        ui: { ...next.ui, outputs: { ...next.ui?.outputs, out: { x: x + 280, y } } },
      }
      notes.push(`全結合の head (出す値 ${outputs}) を置き, 出力 out につなぎました`)
      return { spec: next, notes }
    }
    notes.push(`出力の大きさ ${outputs} は, 最後の部品の引数に入れてください`)
    return { spec: next, notes }
  }
  const source = outputNames.length === 1 ? parseRef(next.outputs[outputNames[0]!]!).name : null
  const last = source ? next.nodes[source] : undefined
  // 最後の部品の出す数の引数 (全結合は out_features, 畳み込みは out_channels)
  const param = ['out_features', 'out_channels'].find(
    (name) => source && last && hasParam(units, last.unit, name),
  )
  const reshaped = source && !param ? fitReshape(next, units, source, outputs) : null
  if (source && last && param) {
    if (last.params?.[param] !== outputs) {
      next = {
        ...next,
        nodes: {
          ...next.nodes,
          [source]: { ...last, params: { ...last.params, [param]: outputs } },
        },
      }
    }
    notes.push(`${source} の ${param} を ${outputs} にしました`)
  } else if (reshaped) {
    next = reshaped.spec
    notes.push(reshaped.note)
  } else {
    notes.push(`出力の大きさ ${outputs} は, 最後の部品の引数に手で入れてください`)
  }
  return { spec: next, notes }
}

/** 知らせに出す入力の形 (データセットで決まらない次元は ?. バッチの次元は例の大きさ) */
function summaryShape(item: DatasetInput): string {
  const dims = item.dataset_shape
  if (!dims || dims.length !== item.shape.length) return shapeText(item.shape)
  const sizes = item.shape.map((size, index) => (index > 0 && dims[index] === null ? '?' : size))
  return `[${sizes.join(', ')}]`
}

/** 入れた結果の知らせの頭 (入力の形, タスク, 目的の列か正解の項目, クラスの数, 出す値の数) */
export function datasetSummary(info: DatasetInfo): string {
  const inputs = info.inputs.map((item) => `${item.name} ${summaryShape(item)}`).join(', ')
  const parts = [`入力 ${inputs}`]
  if (info.task) parts.push(`タスク ${info.task}`)
  if (info.target_kind) parts.push(`正解 ${info.target ?? ''} (${info.target_kind})`)
  else if (info.target) parts.push(`目的の列 ${info.target}`)
  if (info.classes) parts.push(`クラス ${info.classes.length}`)
  if (info.outputs != null) parts.push(`出す値 ${info.outputs}`)
  return `データセットから入れました: ${parts.join(', ')}`
}
