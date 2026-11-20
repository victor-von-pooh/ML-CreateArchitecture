// データセット (ML-EDA の model_inputs) の入力と出す値の数を, 定義に入れる

import { parseRef, renameItem, specWires, type InputDef, type Spec, type UnitSpec } from './graph'

/** データセットの入力 (形はバッチの次元を例の大きさにしたもの) */
export interface DatasetInput {
  name: string
  shape: number[]
  dtype: string
}

/** バックエンドが ML-EDA の model_inputs から返す値 */
export interface DatasetInfo {
  path: string
  inputs: DatasetInput[]
  task?: string | null
  target?: string | null
  classes?: unknown[] | null
  outputs?: number | null
}

/** 入れた結果 (定義と, 知らせる文) */
export interface DatasetApplied {
  spec: Spec
  notes: string[]
}

/** 部品の無い定義に置く, 出す値の数を決める全結合の部品 */
const HEAD_UNIT = 'nn.Linear'

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
 * データセットの入力と出す値の数を定義に入れる
 *
 * 入力: 名前, 形, 型をデータセットに合わせる (1 つずつで名前が違えば名前を
 * 変えてつなぎ先も書き換える. 同じ名前は値の作り方と位置を残す. データセット
 * に無い入力は, つないでいなければ消し, つないでいれば残す). 入力に直に
 * つながる in_features の引数は入力の最後の次元にする.
 * 出す値の数: 部品の無い定義で入力が 2 次元なら全結合の head を置いて出力に
 * つなぎ, 出力が 1 つでつなぎ元が out_features を持てばそれにする
 */
export function applyDatasetInputs(
  spec: Spec,
  info: DatasetInfo,
  units: Map<string, UnitSpec>,
): DatasetApplied {
  const notes: string[] = []
  let next = spec
  const names = info.inputs.map((item) => item.name)

  // 1 つずつで名前が違えば, 名前を変える (つなぎ先も書き換える)
  const current = Object.keys(next.inputs)
  const only = info.inputs.length === 1 ? info.inputs[0]! : null
  if (only && current.length === 1 && current[0] !== only.name && !(only.name in next.nodes)) {
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
  info.inputs.forEach((item, index) => {
    const before = next.inputs[item.name]
    inputs[item.name] = {
      ...before,
      shape: item.shape,
      dtype: item.dtype,
      ui: before?.ui ?? { x: 40, y: 160 + index * 120 },
    }
  })
  next = { ...next, inputs }

  // 入力に直につながる in_features を, 入力の最後の次元に
  const nodes = { ...next.nodes }
  for (const wire of specWires(next, units)) {
    const item = info.inputs.find((input) => input.name === wire.source)
    const node = nodes[wire.target]
    if (!item || !node || item.shape.length < 2) continue
    if (!hasParam(units, node.unit, 'in_features')) continue
    const size = item.shape.at(-1)!
    if (node.params?.in_features === size) continue
    nodes[wire.target] = { ...node, params: { ...node.params, in_features: size } }
    notes.push(`${wire.target} の in_features を ${size} にしました`)
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
    if (first.shape.length === 2 && units.get(HEAD_UNIT)?.available) {
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
  if (source && last && hasParam(units, last.unit, 'out_features')) {
    if (last.params?.out_features !== outputs) {
      next = {
        ...next,
        nodes: {
          ...next.nodes,
          [source]: { ...last, params: { ...last.params, out_features: outputs } },
        },
      }
    }
    notes.push(`${source} の out_features を ${outputs} にしました`)
  } else {
    notes.push(`出力の大きさ ${outputs} は, 最後の部品の引数に手で入れてください`)
  }
  return { spec: next, notes }
}

/** 入れた結果の知らせの頭 (入力の形, タスク, 目的の列, クラスの数, 出す値の数) */
export function datasetSummary(info: DatasetInfo): string {
  const inputs = info.inputs.map((item) => `${item.name} ${shapeText(item.shape)}`).join(', ')
  const parts = [`入力 ${inputs}`]
  if (info.task) parts.push(`タスク ${info.task}`)
  if (info.target) parts.push(`目的の列 ${info.target}`)
  if (info.classes) parts.push(`クラス ${info.classes.length}`)
  if (info.outputs != null) parts.push(`出す値 ${info.outputs}`)
  return `データセットから入れました: ${parts.join(', ')}`
}
