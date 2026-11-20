import { describe, expect, it } from 'vitest'

import { applyDatasetInputs, datasetSummary, type DatasetInfo } from './dataset'
import { emptySpec, type Spec, type UnitSpec } from './graph'

/** テスト用の部品の定義 (引数の名前だけを持つ) */
function unit(name: string, params: string[]): UnitSpec {
  return {
    name,
    category: 'test',
    source: 'test',
    description: '',
    params: params.map((param) => ({
      name: param,
      kind: 'int',
      default: null,
      required: true,
      description: '',
      minimum: null,
      maximum: null,
      choices: null,
    })),
    inputs: [{ name: 'input', description: '', optional: false, variadic: false }],
    outputs: [{ name: 'out', description: '', optional: false, variadic: false }],
    forward_params: [],
    extra_params: false,
    available: true,
    reason: '',
  }
}

const units = new Map([
  ['nn.Linear', unit('nn.Linear', ['in_features', 'out_features'])],
  ['nn.ReLU', unit('nn.ReLU', [])],
])

/** 7 列の表の 3 クラスの分類 */
const table: DatasetInfo = {
  path: '/data/ds',
  inputs: [{ name: 'features', shape: [4, 7], dtype: 'float32' }],
  task: 'classification',
  target: 'label',
  classes: ['a', 'b', 'c'],
  outputs: 3,
}

describe('applyDatasetInputs', () => {
  it('部品の無い定義に入力と全結合の head を置く', () => {
    const { spec, notes } = applyDatasetInputs(emptySpec(), table, units)
    expect(spec.inputs).toEqual({
      features: { shape: [4, 7], dtype: 'float32', ui: { x: 40, y: 160 } },
    })
    expect(spec.nodes.head).toMatchObject({
      unit: 'nn.Linear',
      params: { in_features: 7, out_features: 3 },
      inputs: 'features',
    })
    expect(spec.outputs).toEqual({ out: 'head' })
    expect(notes.join('\n')).toContain('入力 x の名前を features にしました')
  })

  it('名前を変えてつなぎ先も書き換え, 最初と最後の大きさを合わせる', () => {
    const before: Spec = {
      ...emptySpec(),
      inputs: { x: { shape: [8, 16], fill: 'ones', ui: { x: 1, y: 2 } } },
      nodes: {
        fc: { unit: 'nn.Linear', params: { in_features: 16, out_features: 8 }, inputs: 'x' },
        act: { unit: 'nn.ReLU', inputs: 'fc' },
        out: { unit: 'nn.Linear', params: { in_features: 8, out_features: 2 }, inputs: 'act' },
      },
      outputs: { logits: 'out' },
    }
    const { spec } = applyDatasetInputs(before, table, units)
    expect(spec.inputs.features).toEqual({
      shape: [4, 7],
      dtype: 'float32',
      fill: 'ones',
      ui: { x: 1, y: 2 },
    })
    expect(spec.nodes.fc!.inputs).toBe('features')
    expect(spec.nodes.fc!.params).toEqual({ in_features: 7, out_features: 8 })
    expect(spec.nodes.out!.params).toEqual({ in_features: 8, out_features: 3 })
  })

  it('つないでいない余分な入力は消し, つないでいるものは残す', () => {
    const before: Spec = {
      ...emptySpec(),
      inputs: { a: { shape: [4, 2] }, b: { shape: [4, 2] }, features: { shape: [4, 1] } },
      nodes: { fc: { unit: 'nn.Linear', params: {}, inputs: 'a' } },
      outputs: { y: 'fc' },
    }
    const { spec, notes } = applyDatasetInputs(before, table, units)
    expect(Object.keys(spec.inputs).sort()).toEqual(['a', 'features'])
    expect(notes.join('\n')).toContain('入力 a はデータセットに無いですが')
  })

  it('出力が複数か, 最後が out_features を持たない場合は知らせるだけ', () => {
    const before: Spec = {
      ...emptySpec(),
      nodes: { act: { unit: 'nn.ReLU', inputs: 'x' } },
      outputs: { y: 'act' },
    }
    const { spec, notes } = applyDatasetInputs(before, table, units)
    expect(spec.nodes.act).toEqual({ unit: 'nn.ReLU', inputs: 'features' })
    expect(notes.at(-1)).toContain('出力の大きさ 3 は')
  })

  it('タスクを決めていないデータセットは出力を変えない', () => {
    const { spec, notes } = applyDatasetInputs(
      emptySpec(),
      { ...table, task: null, outputs: null },
      units,
    )
    expect(spec.nodes).toEqual({})
    expect(notes.at(-1)).toContain('タスクを決めていない')
  })
})

describe('datasetSummary', () => {
  it('入力の形, タスク, クラスの数, 出す値の数を並べる', () => {
    expect(datasetSummary(table)).toBe(
      'データセットから入れました: 入力 features [4, 7], タスク classification, 目的の列 label, クラス 3, 出す値 3',
    )
  })
})

describe('applyDatasetInputs (名前の重なり)', () => {
  it('ノードと同じ名前の入力は作らず, 何も変えない', () => {
    const before: Spec = {
      ...emptySpec(),
      nodes: { features: { unit: 'nn.ReLU', inputs: 'x' } },
      outputs: { y: 'features' },
    }
    const { spec, notes } = applyDatasetInputs(before, table, units)
    expect(spec).toBe(before)
    expect(notes[0]).toContain('ノード features')
  })
})
