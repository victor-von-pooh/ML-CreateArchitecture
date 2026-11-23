// データセット (ML-EDA の model_inputs) の入力と出す値の数を, 定義に入れる

import { parseRef, renameItem, specWires, type InputDef, type Spec, type UnitSpec } from './graph'

/** データセットの入力 (形はバッチの次元を例の大きさにしたもの. dataset_shape は決まらない次元が null の形) */
export interface DatasetInput {
  name: string
  shape: number[]
  dtype: string
  dataset_shape?: (number | null)[]
  /** 文の語の番号の入力の語彙の大きさ (語の番号の上限 + 1. 文でなければ無い) */
  vocabulary?: number | null
}

/** 表の生成の符号化した行の幅と内訳 (ML-Quest の table_encoding) */
export interface TableWidth {
  width: number
  /** 数の列の数 (1 ずつ) */
  numbers: number
  /** 種類の列の数 */
  categories: number
  /** 種類の列の値の種類の数の和 */
  values: number
  /** 学習に使う列の数 */
  columns: number
  /** 問題の設定の名前 */
  problem?: string | null
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
  /** 表の生成の問題の設定で, 入力と出力の幅にした符号化した行の幅 (他は無い) */
  table?: TableWidth | null
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

/**
 * データセットに無いが残す入力の最初の次元をそろえる大きさ (バッチの次元)
 *
 * 定義の入力 (0 次元の値を除く) の最初の次元が全て同じで, そのうちの 1 つ
 * 以上がデータセットの入力と同じ名前の場合だけ, 最初の次元をバッチとみなし,
 * データセットの入力の最初の次元 (例の大きさ) を返す. 他 (最初の次元が違う
 * 入力がある, データセットの入力と同じ名前の入力が無い, データセットの入力の
 * 最初の次元がそろわない) は null (固定の形の値を変えないため)
 */
function batchSize(spec: Spec, info: DatasetInfo): number | null {
  const firsts = Object.values(spec.inputs).flatMap((value) =>
    value.shape && value.shape.length > 0 ? [value.shape[0]!] : [],
  )
  const sizes = new Set(info.inputs.map((item) => item.shape[0]))
  const [size] = sizes
  if (firsts.length === 0 || new Set(firsts).size !== 1 || sizes.size !== 1) return null
  const replaced = info.inputs.some((item) => (spec.inputs[item.name]?.shape?.length ?? 0) > 0)
  return replaced && typeof size === 'number' ? size : null
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

/** 表の生成で, 入力と出力の幅をどう決めたかの文 */
export function tableWidthNote(table: TableWidth): string {
  const from = table.problem ? `問題の設定 ${table.problem} が表の生成` : '表の生成の問題の設定'
  return (
    `${from}なので, 入力と出力の幅を, 学習で使う 1 行の長さ ${table.width} にしました ` +
    `(数の項目 ${table.numbers} は 1 つずつ, 分類の項目 ${table.categories} は選べる値の数の合計 ${table.values}. ` +
    `項目の数 ${table.columns} ではありません)`
  )
}

/**
 * データセットの入力と出す値の数を定義に入れる
 *
 * 入力: 名前, 形, 型をデータセットに合わせる (1 つずつで名前が違えば名前を
 * 変えてつなぎ先も書き換える. 同じ名前は値の作り方と位置を残す. データセット
 * に無い入力は, つないでいなければ消し, つないでいれば残す. 残す入力の最初
 * の次元は, 定義の入力の最初の次元が全て同じならバッチとみなし, データ
 * セットの入力の最初の次元 (例の大きさ) にそろえる. データセットで
 * 決まらない次元は定義の値のまま). 入力に直につながる in_features の引数は
 * 入力の最後の次元, in_channels の引数は 3 次元以上の入力の 2 番目の次元
 * (チャンネル), num_embeddings の引数は文の語の番号の入力の語彙の大きさにする.
 * 出す値の数: 部品の無い定義で入力が 2 次元なら全結合の head を置いて出力に
 * つなぎ, 出力が 1 つでつなぎ元が out_features か out_channels を持てばそれ
 * にする. つなぎ元が op.Reshape なら最初の次元とその前の out_features にする.
 * 表の生成の問題の設定 (info.table) では, 入力の列の次元と出す値の数が
 * 符号化した行の幅なので, その決め方を初めに知らせる
 */
export function applyDatasetInputs(
  spec: Spec,
  info: DatasetInfo,
  units: Map<string, UnitSpec>,
): DatasetApplied {
  const notes: string[] = info.table ? [tableWidthNote(info.table)] : []
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

  // データセットに無い入力は, つないでいなければ消す. 残す入力の最初の
  // 次元は, バッチとみなせればデータセットの入力にそろえる
  const batch = batchSize(next, info)
  const inputs: Record<string, InputDef> = {}
  for (const [name, value] of Object.entries(next.inputs)) {
    if (names.includes(name)) continue
    if (isUsed(next, units, name)) {
      const shape = value.shape
      if (batch !== null && shape && shape.length > 0 && shape[0] !== batch) {
        const aligned = [batch, ...shape.slice(1)]
        inputs[name] = { ...value, shape: aligned }
        notes.push(
          `入力 ${name} はデータセットに無いですが, つないでいるので残し, ` +
            `最初の次元 (バッチ) をデータセットの入力と同じ ${batch} にしました (${shapeText(aligned)})`,
        )
        continue
      }
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
  // チャンネル (3 次元以上の入力の 2 番目の次元) に, num_embeddings を
  // 文の語彙の大きさ (埋め込みの表の行の数) に
  const vocabularies = new Map(
    info.inputs.flatMap((item) =>
      typeof item.vocabulary === 'number' && item.vocabulary > 0
        ? [[item.name, item.vocabulary] as const]
        : [],
    ),
  )
  const nodes = { ...next.nodes }
  for (const wire of specWires(next, units)) {
    const shape = shapes.get(wire.source)
    const node = nodes[wire.target]
    if (!shape || !node || shape.length < 2) continue
    const fits: [string, number][] = [['in_features', shape.at(-1)!]]
    if (shape.length >= 3) fits.push(['in_channels', shape[1]!])
    const vocabulary = vocabularies.get(wire.source)
    if (vocabulary !== undefined) fits.push(['num_embeddings', vocabulary])
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
  // 最後の部品の出す数の引数 (全結合は out_features, 畳み込みは out_channels).
  // 点群の outputs は値の数でなく点の数なので, 形を変える部品 (Reshape) だけで合わせる
  const cloud = info.target_kind === 'point_cloud'
  const param = cloud
    ? undefined
    : ['out_features', 'out_channels'].find(
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
    notes.push(
      cloud
        ? `正解の点の数 ${outputs} は, 最後に点の数と 3 の形にする部品 (op.Reshape) を置いて手で合わせてください`
        : `出力の大きさ ${outputs} は, 最後の部品の引数に手で入れてください`,
    )
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
