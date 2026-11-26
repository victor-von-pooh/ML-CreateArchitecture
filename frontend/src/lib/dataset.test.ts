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
    expect(spec.inputs.t).toMatchObject({ shape: [4], dtype: 'int64', fill: 'arange' })
    expect(spec.nodes.x_fc!.params).toMatchObject({ in_features: 8, out_features: 32 })
    expect(spec.nodes.head!.params).toMatchObject({ in_features: 32, out_features: 8 })
    expect(spec.nodes.t_fc!.params).toMatchObject({ in_features: 32 })
    expect(notes[0]).toBe(
      '問題の設定 顧客の表 が表の生成なので, 入力と出力の幅を, 学習で使う 1 行の長さ 8 にしました ' +
        '(数の項目 3 は 1 つずつ, 分類の項目 2 は選べる値の数の合計 5. 項目の数 5 ではありません)',
    )
    expect(notes).toContain(
      '入力 t はデータセットに無いですが, つないでいるので残し, ' +
        '最初の次元 (バッチ) を例の大きさ 4 にしました ([4])',
    )
    expect(notes).toContain('head の out_features を 8 にしました')
  })

  it('最初の次元が入力ごとに違えば, 残す入力の形を変えない', () => {
    const fixed: Spec = {
      ...denoiser,
      inputs: { ...denoiser.inputs, t: { ...denoiser.inputs.t!, shape: [10] } },
    }
    const { spec, notes } = applyDatasetInputs(fixed, generated, units)
    expect(spec.inputs.t!.shape).toEqual([10])
    expect(notes).toContain('入力 t はデータセットに無いですが, つないでいるので残しました')
  })

  it('データセットの入力と同じ名前の入力が無ければ, 残す入力の形を変えない', () => {
    const other: Spec = {
      ...denoiser,
      inputs: {
        a: { shape: [8, 16] },
        t: denoiser.inputs.t!,
      },
      nodes: {
        ...denoiser.nodes,
        a_fc: { unit: 'nn.Linear', params: { in_features: 16, out_features: 32 }, inputs: 'a' },
      },
    }
    const { spec } = applyDatasetInputs(other, generated, units)
    expect(spec.inputs.a!.shape).toEqual([8, 16])
    expect(spec.inputs.t!.shape).toEqual([8])
  })

  it('0 次元の値は数えず, 形も変えない', () => {
    const scalar: Spec = {
      ...denoiser,
      inputs: { ...denoiser.inputs, s: { shape: [] } },
      nodes: {
        ...denoiser.nodes,
        s_fc: { unit: 'nn.Linear', params: { in_features: 1, out_features: 1 }, inputs: 's' },
      },
    }
    const { spec } = applyDatasetInputs(scalar, generated, units)
    expect(spec.inputs.s!.shape).toEqual([])
    expect(spec.inputs.t!.shape).toEqual([4])

    // 0 次元の値だけの定義はそろえない
    const only: Spec = {
      ...scalar,
      inputs: { x: { shape: [] }, s: { shape: [] } },
    }
    const { spec: kept } = applyDatasetInputs(only, generated, units)
    expect(kept.inputs.s!.shape).toEqual([])
  })

  it('データセットの入力どうしで最初の次元が違えば, 残す入力の形を変えない', () => {
    const pair: Spec = {
      ...denoiser,
      inputs: { ...denoiser.inputs, y: { shape: [8, 16] } },
    }
    const mixed: DatasetInfo = {
      ...generated,
      inputs: [
        { name: 'x', shape: [4, 8], dtype: 'float32' },
        { name: 'y', shape: [2, 8], dtype: 'float32' },
      ],
    }
    const { spec } = applyDatasetInputs(pair, mixed, units)
    expect(spec.inputs.t!.shape).toEqual([8])
  })

  it('名前を変えた後の入力で照らす (1 つずつの入力は名前を変えて形を入れ, 残す入力は無い)', () => {
    const single: Spec = {
      ...emptySpec(),
      inputs: { a: { shape: [8, 16] } },
      nodes: {
        fc: { unit: 'nn.Linear', params: { in_features: 16, out_features: 8 }, inputs: 'a' },
      },
      outputs: { out: 'fc' },
    }
    const { spec, notes } = applyDatasetInputs(single, generated, units)
    expect(Object.keys(spec.inputs)).toEqual(['x'])
    expect(spec.inputs.x!.shape).toEqual([4, 8])
    expect(notes).toContain('入力 a の名前を x にしました')
    expect(notes.join('\n')).not.toContain('残し')
  })

  it('表の生成でなければ決め方を知らせない', () => {
    const { notes } = applyDatasetInputs(denoiser, { ...generated, table: null }, units)
    expect(notes.join('\n')).not.toContain('1 行の長さ')
  })
})

describe('applyDatasetInputs (並べる)', () => {
  /** 同梱の例 two_tower_retrieval の入力と 1 段目の全結合 (利用者 12 人, 品物 20) */
  const tower: Spec = {
    ...emptySpec(),
    inputs: {
      user: { shape: [8, 12], ui: { x: 0, y: 60 } },
      item: { shape: [8, 20], ui: { x: 0, y: 300 } },
    },
    nodes: {
      user_fc1: {
        unit: 'nn.Linear',
        params: { in_features: 12, out_features: 32 },
        inputs: 'user',
      },
      item_fc1: {
        unit: 'nn.Linear',
        params: { in_features: 20, out_features: 32 },
        inputs: 'item',
      },
    },
    outputs: { user_embedding: 'user_fc1', item_embedding: 'item_fc1' },
  }
  /** 学習用の利用者 6 人と品物 4 つ (プロジェクトの学習の設定の割合で分けた) */
  const reactions: DatasetInfo = {
    path: '/proj/data/reactions.csv',
    inputs: [
      { name: 'user', shape: [4, 6], dtype: 'float32', dataset_shape: [null, 6] },
      { name: 'item', shape: [4, 4], dtype: 'float32', dataset_shape: [null, 4] },
    ],
    outputs: null,
    rank: {
      users: 6,
      items: 4,
      rows: { train: 31, valid: 4, test: 4 },
      valid: 0.1,
      test: 0.1,
      notes: ['反応の強さ (qty) が 0 以下か数でない 1 行を除きました'],
      problem: '会員の品',
    },
  }

  it('入力 user と item の大きさを学習用の利用者と品物の数にし, 決め方を初めに知らせる', () => {
    const { spec, notes } = applyDatasetInputs(tower, reactions, units)
    expect(spec.inputs.user).toMatchObject({ shape: [4, 6], dtype: 'float32' })
    expect(spec.inputs.item).toMatchObject({ shape: [4, 4], dtype: 'float32' })
    expect(spec.nodes.user_fc1!.params).toMatchObject({ in_features: 6, out_features: 32 })
    expect(spec.nodes.item_fc1!.params).toMatchObject({ in_features: 4, out_features: 32 })
    expect(notes[0]).toBe(
      '問題の設定 会員の品 が並べるなので, 入力 user の大きさを学習用の利用者の数 6, ' +
        'item の大きさを学習用の品物の数 4 にしました (学習用の反応の行の数 31 (時刻の順に分けたもの). ' +
        '分ける割合はプロジェクトの学習の設定の検証用 0.1 とテスト用 0.1. ' +
        '検証用とテスト用にだけ出る利用者と品物は数えません)',
    )
    expect(notes[1]).toBe('反応の強さ (qty) が 0 以下か数でない 1 行を除きました')
    expect(notes).toContain(
      '出力 score は利用者と品物の全ての組の点数の表なので, 出力の大きさは変えていません',
    )
    expect(datasetSummary(reactions)).toBe('反応の表から入れました: 入力 user [4, 6], item [4, 4]')
  })

  it('config に割合が無ければ既定の割合と知らせる', () => {
    const rank = { ...reactions.rank!, valid: null, test: null, problem: null }
    const { notes } = applyDatasetInputs(tower, { ...reactions, rank }, units)
    expect(notes[0]).toContain('並べるの問題の設定なので')
    expect(notes[0]).toContain('分ける割合は学習のスクリプトの既定.')
  })
})

describe('applyDatasetInputs (異常の検知)', () => {
  /** 同梱の例 tabular_autoencoder の形 (16 列を縮めた幅 4 を通して戻す) */
  const autoencoder: Spec = {
    ...emptySpec(),
    inputs: { x: { shape: [8, 16] } },
    nodes: {
      enc1: { unit: 'nn.Linear', params: { in_features: 16, out_features: 8 }, inputs: 'x' },
      enc2: { unit: 'nn.Linear', params: { in_features: 8, out_features: 4 }, inputs: 'enc1' },
      dec1: { unit: 'nn.Linear', params: { in_features: 4, out_features: 8 }, inputs: 'enc2' },
      dec2: { unit: 'nn.Linear', params: { in_features: 8, out_features: 16 }, inputs: 'dec1' },
    },
    outputs: { recon: 'dec2' },
  }
  /** 学習に使う 3 列と, 異常の行にだけ印のある正解の列の表 */
  const readings: DatasetInfo = {
    path: '/data/readings',
    inputs: [{ name: 'x', shape: [4, 3], dtype: 'float32', dataset_shape: [null, 3] }],
    task: 'anomaly',
    target: 'is_anomaly',
    classes: null,
    outputs: 3,
    anomaly: { columns: 3, target: 'is_anomaly', score: 'autoencoder', problem: '機械の異常' },
  }

  it('入力と出力の幅を学習に使う列の数にし, 縮めた幅は変えず, 決め方を初めに知らせる', () => {
    const { spec, notes } = applyDatasetInputs(autoencoder, readings, units)
    expect(spec.inputs.x!.shape).toEqual([4, 3])
    expect(spec.nodes.enc1!.params).toEqual({ in_features: 3, out_features: 8 })
    expect(spec.nodes.dec2!.params).toEqual({ in_features: 8, out_features: 3 })
    expect(spec.nodes.enc2!.params).toEqual({ in_features: 8, out_features: 4 })
    expect(spec.nodes.dec1!.params).toEqual({ in_features: 4, out_features: 8 })
    expect(notes[0]).toBe(
      '問題の設定 機械の異常 が異常の検知 (自己符号化器)なので, 入力と出力の幅を, 学習に使う列の数 3 にしました ' +
        '(自己符号化器は入力を作り直すので, 出力を入力と同じ幅にします. 正解の列 is_anomaly は数えません)',
    )
    expect(notes).toContain('dec2 の out_features を 3 にしました')
    expect(datasetSummary(readings)).toBe(
      'データセットから入れました: 入力 x [4, 3], タスク anomaly, 目的の列 is_anomaly, 出す値 3',
    )
  })

  it('正解の無い表で問題の設定を渡さなければ, データセットから決めたと知らせる', () => {
    const plain: DatasetInfo = {
      ...readings,
      target: null,
      anomaly: { columns: 3, target: null, score: null, problem: null },
    }
    const { spec, notes } = applyDatasetInputs(autoencoder, plain, units)
    expect(spec.nodes.dec2!.params).toMatchObject({ out_features: 3 })
    expect(notes[0]).toBe(
      '異常の検知のデータセットなので, 入力と出力の幅を, 学習に使う列の数 3 にしました ' +
        '(出す値の数はデータセットの決まり (異常の検知は学習に使う列の数) のままです)',
    )
    expect(notes[0]).not.toContain('自己符号化器')
  })

  it('度合いの決め方に答えていない問題の設定では, 自己符号化器と言わずに未回答を知らせる', () => {
    const unanswered: DatasetInfo = {
      ...readings,
      anomaly: { ...readings.anomaly!, score: null },
    }
    const { spec, notes } = applyDatasetInputs(autoencoder, unanswered, units)
    expect(spec.nodes.dec2!.params).toMatchObject({ out_features: 3 })
    expect(notes[0]).toBe(
      '問題の設定 機械の異常 が異常の検知なので, 入力と出力の幅を, 学習に使う列の数 3 にしました ' +
        '(度合いの決め方にまだ答えていないので, 出す値の数はデータセットの決まり (異常の検知は学習に使う列の数) のままです. ' +
        '正解の列 is_anomaly は数えません)',
    )
  })

  it('距離に基づく度合いの問題の設定では, モデルの定義が要らないと知らせて何も変えない', () => {
    const distance: DatasetInfo = {
      ...readings,
      anomaly: { ...readings.anomaly!, score: 'distance' },
    }
    const { spec, notes } = applyDatasetInputs(autoencoder, distance, units)
    expect(spec).toBe(autoencoder)
    expect(notes).toEqual([
      '問題の設定 機械の異常 の度合いの決め方は距離に基づく度合い (学習しない) なので, ' +
        'モデルの定義は要りません. 何も変えていません',
    ])
    expect(datasetSummary(distance)).toMatch(/^データセットを読みました: /)
  })
})

describe('applyDatasetInputs (効果を見積もる)', () => {
  /** 同梱の例 tabular_uplift の形 (16 列を重みを共有しない 2 つの枝に通す) */
  const twoHeads: Spec = {
    ...emptySpec(),
    inputs: { x: { shape: [8, 16] } },
    nodes: {
      treated1: { unit: 'nn.Linear', params: { in_features: 16, out_features: 32 }, inputs: 'x' },
      treated2: {
        unit: 'nn.Linear',
        params: { in_features: 32, out_features: 1 },
        inputs: 'treated1',
      },
      control1: { unit: 'nn.Linear', params: { in_features: 16, out_features: 32 }, inputs: 'x' },
      control2: {
        unit: 'nn.Linear',
        params: { in_features: 32, out_features: 1 },
        inputs: 'control1',
      },
    },
    outputs: { treated: 'treated2', control: 'control2' },
  }
  /** 処置の列を除いた学習に使う 5 列の介入の記録の表 */
  const log: DatasetInfo = {
    path: '/data/log',
    inputs: [{ name: 'x', shape: [4, 5], dtype: 'float32', dataset_shape: [null, 5] }],
    task: 'uplift',
    target: 'purchased',
    classes: null,
    outputs: 1,
    uplift: {
      columns: 5,
      outcome: 'purchased',
      treatment: 'coupon',
      id: 'customer_id',
      problem: 'クーポンの効果',
    },
  }

  it('入力の幅を学習に使う列の数にし, 2 つの出力を 1 にし, 隠れ層の幅は変えず, 決め方を初めに知らせる', () => {
    const { spec, notes } = applyDatasetInputs(twoHeads, log, units)
    expect(spec.inputs.x!.shape).toEqual([4, 5])
    expect(spec.nodes.treated1!.params).toEqual({ in_features: 5, out_features: 32 })
    expect(spec.nodes.control1!.params).toEqual({ in_features: 5, out_features: 32 })
    expect(spec.nodes.treated2!.params).toEqual({ in_features: 32, out_features: 1 })
    expect(spec.nodes.control2!.params).toEqual({ in_features: 32, out_features: 1 })
    expect(notes[0]).toBe(
      '問題の設定 クーポンの効果 が効果の見積もりなので, 入力の幅を, 学習に使う列の数 5 にしました. ' +
        '入力に入れない列: 処置 (行動したかどうか. クーポンを配ったかなど) の列 coupon, ' +
        '結果 (行動の後にどうなったか. 買ったかなど) の列 purchased, 行を見分ける id の列 customer_id ' +
        '(効果は処置したときとしないときの結果の差として学ぶので, 処置と結果は入力に入れません). ' +
        '効果の見積もりの 2 つの出力 (処置したときの結果の予測と, しないときの結果の予測) は, ' +
        'どちらも 1 行に 1 つの値を出します',
    )
    expect(notes).toContain('treated1 の in_features を 5 にしました')
    expect(notes).toContain('treated2 の out_features を 1 にしました')
    expect(notes).toContain('control2 の out_features を 1 にしました')
    expect(datasetSummary(log)).toBe(
      'データセットから入れました: 入力 x [4, 5], タスク uplift, 目的の列 purchased, 出す値 1',
    )
  })

  it('出す値の数が違う出力も 1 にし, 名前の分からない列は役だけを出す', () => {
    const wide: Spec = {
      ...twoHeads,
      nodes: {
        ...twoHeads.nodes,
        control2: { ...twoHeads.nodes.control2!, params: { in_features: 32, out_features: 3 } },
      },
    }
    const plain: DatasetInfo = {
      ...log,
      uplift: { columns: 5, outcome: 'purchased', treatment: null, id: null, problem: null },
    }
    const { spec, notes } = applyDatasetInputs(wide, plain, units)
    expect(spec.nodes.control2!.params).toEqual({ in_features: 32, out_features: 1 })
    expect(notes[0]).toBe(
      '効果を見積もるのデータセットなので, 入力の幅を, 学習に使う列の数 5 にしました. ' +
        '入力に入れない列: 処置 (行動したかどうか. クーポンを配ったかなど) の列, ' +
        '結果 (行動の後にどうなったか. 買ったかなど) の列 purchased, 行を見分ける id の列 ' +
        '(効果は処置したときとしないときの結果の差として学ぶので, 処置と結果は入力に入れません). ' +
        '効果の見積もりの 2 つの出力 (処置したときの結果の予測と, しないときの結果の予測) は, ' +
        'どちらも 1 行に 1 つの値を出します',
    )
  })

  it('out_features を持たないつなぎ元の出力は手で入れるよう知らせる', () => {
    const relu: Spec = {
      ...twoHeads,
      nodes: { ...twoHeads.nodes, act: { unit: 'nn.ReLU', inputs: 'control2' } },
      outputs: { treated: 'treated2', control: 'act' },
    }
    const { notes } = applyDatasetInputs(relu, log, units)
    expect(notes).toContain('出力 control の大きさ 1 は, 最後の部品の引数に手で入れてください')
  })
})
