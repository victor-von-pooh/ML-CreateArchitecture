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
  ['nn.Conv2d', unit('nn.Conv2d', ['in_channels', 'out_channels', 'kernel_size'])],
  ['op.Reshape', unit('op.Reshape', ['shape'])],
  ['nn.Embedding', unit('nn.Embedding', ['num_embeddings', 'embedding_dim'])],
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

describe('applyDatasetInputs (文の語彙)', () => {
  it('入力に直につながる埋め込みの num_embeddings を語彙の大きさにする', () => {
    const text: DatasetInfo = {
      path: '/data/text',
      inputs: [{ name: 'x', shape: [4, 16], dtype: 'int64', vocabulary: 21 }],
      task: 'classification',
      target: 'label',
      classes: ['neg', 'pos'],
      outputs: 2,
    }
    const before: Spec = {
      ...emptySpec(),
      inputs: { ids: { shape: [4, 12], dtype: 'int64' } },
      nodes: {
        emb: {
          unit: 'nn.Embedding',
          params: { num_embeddings: 100, embedding_dim: 8 },
          inputs: 'ids',
        },
        head: { unit: 'nn.Linear', params: { in_features: 8, out_features: 5 }, inputs: 'emb' },
      },
      outputs: { logits: 'head' },
    }
    const { spec, notes } = applyDatasetInputs(before, text, units)
    expect(spec.inputs.x?.shape).toEqual([4, 16])
    expect(spec.nodes.emb?.params).toEqual({ num_embeddings: 21, embedding_dim: 8 })
    expect(spec.nodes.head?.params?.out_features).toBe(2)
    expect(notes).toContain('emb の num_embeddings を 21 にしました')

    // 語彙の大きさが無ければ埋め込みは変えない
    const { spec: kept } = applyDatasetInputs(
      before,
      { ...text, inputs: [{ name: 'x', shape: [4, 16], dtype: 'int64' }] },
      units,
    )
    expect(kept.nodes.emb?.params?.num_embeddings).toBe(100)
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

/** 画像と領域のファイルの組 (大きさの混ざる灰色の画像. 背景を含む 3 クラス) */
const masks: DatasetInfo = {
  path: '/data/masks',
  inputs: [
    { name: 'image', shape: [4, 1, 4, 4], dtype: 'float32', dataset_shape: [null, 1, null, null] },
  ],
  task: null,
  target: 'mask',
  classes: ['background', 'circle', 'square'],
  target_kind: 'mask',
  outputs: 3,
}

/** 撮影と点群のファイルの組 (入力 512 点, 出す点 1024) */
const clouds: DatasetInfo = {
  path: '/data/clouds',
  inputs: [
    { name: 'capture', shape: [4, 512, 3], dtype: 'float32', dataset_shape: [null, 512, 3] },
  ],
  task: null,
  target: 'cloud',
  classes: null,
  target_kind: 'point_cloud',
  outputs: 1024,
}

describe('applyDatasetInputs (ファイルの組)', () => {
  it('畳み込みの in_channels と head の out_channels を合わせ, 決まらない大きさは定義のまま', () => {
    const before: Spec = {
      ...emptySpec(),
      inputs: { image: { shape: [4, 3, 32, 32] } },
      nodes: {
        enc: {
          unit: 'nn.Conv2d',
          params: { in_channels: 3, out_channels: 16, kernel_size: 3 },
          inputs: 'image',
        },
        head: {
          unit: 'nn.Conv2d',
          params: { in_channels: 16, out_channels: 2, kernel_size: 1 },
          inputs: 'enc',
        },
      },
      outputs: { logits: 'head' },
    }
    const { spec, notes } = applyDatasetInputs(before, masks, units)
    expect(spec.inputs.image!.shape).toEqual([4, 1, 32, 32])
    expect(spec.nodes.enc!.params).toEqual({ in_channels: 1, out_channels: 16, kernel_size: 3 })
    expect(spec.nodes.head!.params).toEqual({ in_channels: 16, out_channels: 3, kernel_size: 1 })
    expect(notes.join('\n')).toContain('定義の値のまま')
    expect(notes.at(-1)).toBe('head の out_channels を 3 にしました')
  })

  it('最後の op.Reshape の点の数と, その前の out_features を出す点の数にする', () => {
    const before: Spec = {
      ...emptySpec(),
      inputs: { observed: { shape: [4, 256, 3] } },
      nodes: {
        encode: {
          unit: 'nn.Linear',
          params: { in_features: 3, out_features: 8 },
          inputs: 'observed',
        },
        decode: {
          unit: 'nn.Linear',
          params: { in_features: 8, out_features: 768 },
          inputs: 'encode',
        },
        cloud: { unit: 'op.Reshape', params: { shape: [256, 3] }, inputs: 'decode' },
      },
      outputs: { cloud: 'cloud' },
    }
    const { spec, notes } = applyDatasetInputs(before, clouds, units)
    expect(Object.keys(spec.inputs)).toEqual(['capture'])
    expect(spec.inputs.capture!.shape).toEqual([4, 512, 3])
    expect(spec.nodes.encode!.inputs).toBe('capture')
    expect(spec.nodes.encode!.params).toEqual({ in_features: 3, out_features: 8 })
    expect(spec.nodes.decode!.params).toEqual({ in_features: 8, out_features: 3072 })
    expect(spec.nodes.cloud!.params).toEqual({ shape: [1024, 3] })
    expect(notes.at(-1)).toBe(
      'decode の out_features を 3072, cloud の shape を [1024, 3] にしました',
    )
  })

  it('点群で最後が点ごとの全結合なら, 点の数を値の数に入れず知らせる', () => {
    const before: Spec = {
      ...emptySpec(),
      inputs: { observed: { shape: [4, 512, 3] } },
      nodes: {
        dec: {
          unit: 'nn.Linear',
          params: { in_features: 3, out_features: 3 },
          inputs: 'observed',
        },
      },
      outputs: { cloud: 'dec' },
    }
    const { spec, notes } = applyDatasetInputs(before, clouds, units)
    expect(spec.nodes.dec!.params).toEqual({ in_features: 3, out_features: 3 })
    expect(notes.at(-1)).toContain('正解の点の数 1024')
  })

  it('正解の種類と項目の名前を知らせる', () => {
    expect(datasetSummary(masks)).toBe(
      'データセットから入れました: 入力 image [4, 1, ?, ?], 正解 mask (mask), クラス 3, 出す値 3',
    )
  })
})

describe('applyDatasetInputs (表の生成)', () => {
  /** 同梱の例 diffusion_denoiser の形 (雑音を加えた行 x と時刻 t から雑音を当てる) */
  const denoiser: Spec = {
    ...emptySpec(),
    inputs: {
      x: { shape: [8, 16], ui: { x: 0, y: 60 } },
      t: { shape: [8], dtype: 'int64', fill: 'arange', ui: { x: 0, y: 260 } },
    },
    nodes: {
      t_fc: { unit: 'nn.Linear', params: { in_features: 32, out_features: 32 }, inputs: 't' },
      x_fc: { unit: 'nn.Linear', params: { in_features: 16, out_features: 32 }, inputs: 'x' },
      head: { unit: 'nn.Linear', params: { in_features: 32, out_features: 16 }, inputs: 'x_fc' },
    },
    outputs: { noise: 'head' },
  }
  /** 列の数 5 (数の列 3, 値の種類 3 と 2 の種類の列) の符号化した行の幅 8 */
  const generated: DatasetInfo = {
    path: '/data/ds',
    inputs: [{ name: 'x', shape: [4, 8], dtype: 'float32', dataset_shape: [null, 8] }],
    task: null,
    outputs: 8,
    table: { width: 8, numbers: 3, categories: 2, values: 5, columns: 5, problem: '顧客の表' },
  }

  it('入力と出力の幅を符号化した行の幅にし, 決め方を初めに知らせる', () => {
    const { spec, notes } = applyDatasetInputs(denoiser, generated, units)
    expect(spec.inputs.x).toMatchObject({ shape: [4, 8], dtype: 'float32' })
    expect(spec.inputs.t).toMatchObject({ shape: [8], dtype: 'int64', fill: 'arange' })
    expect(spec.nodes.x_fc!.params).toMatchObject({ in_features: 8, out_features: 32 })
    expect(spec.nodes.head!.params).toMatchObject({ in_features: 32, out_features: 8 })
    expect(spec.nodes.t_fc!.params).toMatchObject({ in_features: 32 })
    expect(notes[0]).toBe(
      '問題の設定 顧客の表 が表の生成なので, 入力と出力の幅を学習のスクリプトと同じ符号化した行の幅 8 にしました ' +
        '(数の列 3 は 1 ずつ, 種類の列 2 は値の種類の数の和 5. 列の数 5 ではありません)',
    )
    expect(notes).toContain('入力 t はデータセットに無いですが, つないでいるので残しました')
    expect(notes).toContain('head の out_features を 8 にしました')
  })

  it('表の生成でなければ決め方を知らせない', () => {
    const { notes } = applyDatasetInputs(denoiser, { ...generated, table: null }, units)
    expect(notes.join('\n')).not.toContain('符号化した行の幅')
  })
})
