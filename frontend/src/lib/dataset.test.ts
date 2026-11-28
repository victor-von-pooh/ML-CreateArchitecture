import { describe, expect, it } from 'vitest'

import {
  applyDatasetInputs,
  datasetSummary,
  sameDefinition,
  templateValues,
  type DatasetInfo,
  type TemplateMatch,
  type TemplatePlace,
} from './dataset'
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
  ['nn.Softplus', unit('nn.Softplus', ['beta', 'threshold'])],
  ['nn.Conv2d', unit('nn.Conv2d', ['in_channels', 'out_channels', 'kernel_size'])],
  ['op.Reshape', unit('op.Reshape', ['shape'])],
  ['nn.Embedding', unit('nn.Embedding', ['num_embeddings', 'embedding_dim'])],
  ['op.CumSum', unit('op.CumSum', ['dim'])],
])

/** 雛形の引数ごとの書き換える場所 (ML-ModelControl の TEMPLATE_ARGS の形) */
type Places = Record<string, TemplatePlace[]>

/** 入力の形の軸の場所 */
function inputAt(owner: string, key: number): TemplatePlace {
  return { kind: 'input', owner, key, scale: 1 }
}

/** ノードの引数の場所 */
function paramAt(owner: string, key: string, scale = 1): TemplatePlace {
  return { kind: 'param', owner, key, scale }
}

/** 場所の今の値 (倍率で割る) */
function readPlace(spec: Spec, place: TemplatePlace): number {
  const key = place.key
  const raw =
    place.kind === 'input'
      ? spec.inputs[place.owner]!.shape![key as number]!
      : place.kind === 'shape'
        ? (spec.nodes[place.owner]!.params!.shape as number[])[key as number]!
        : (spec.nodes[place.owner]!.params![key] as number)
  return raw / place.scale
}

/** ML-ModelControl の adapt_example の代わりに, 引数の全ての場所に値を書いた定義 */
function writePlaces(spec: Spec, places: Places, values: Record<string, number>): Spec {
  const next = structuredClone(spec)
  for (const [arg, value] of Object.entries(values)) {
    for (const place of places[arg]!) {
      const written = value * place.scale
      if (place.kind === 'input') next.inputs[place.owner]!.shape![place.key as number] = written
      else if (place.kind === 'shape') {
        ;(next.nodes[place.owner]!.params!.shape as number[])[place.key as number] = written
      } else next.nodes[place.owner]!.params![place.key as string] = written
    }
  }
  return next
}

/**
 * 雛形のままの定義として入れた結果 (バックエンドの照らし合わせと書き換えの代わりに,
 * 場所から今の値を読み, データセットから決めた値を書いた定義を渡す)
 */
function viaTemplate(name: string, spec: Spec, places: Places, info: DatasetInfo) {
  const match: TemplateMatch = {
    name,
    args: Object.entries(places).map(([arg, found]) => ({
      name: arg,
      value: readPlace(spec, found[0]!),
      places: found,
    })),
  }
  const { values, notes } = templateValues(match, spec, info)
  const args = match.args
    .filter((arg) => arg.name in values)
    .map((arg) => ({
      name: arg.name,
      before: arg.value,
      after: values[arg.name]!,
      places: arg.places,
    }))
  const applied = applyDatasetInputs(spec, info, units, {
    name,
    spec: writePlaces(spec, places, values),
    args,
    notes,
  })
  return { ...applied, values }
}

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
      '問題 顧客の表 が表の生成なので, 入力と出力の幅を, 学習で使う 1 行の長さ 8 にしました ' +
        '(数の項目 3 は 1 つずつ, 分類の項目 2 は選べる値の数の合計 5. 項目の数 5 ではありません)',
    )
    expect(notes).toContain(
      '入力 t はデータセットに無いですが, つないでいるので残し, ' +
        '最初の次元 (バッチ) を例の大きさ 4 にしました ([4])',
    )
    expect(notes).toContain('head の out_features を 8 にしました')
  })

  it('雛形 diffusion_denoiser のままなら, 書き換えた雛形に入れて今と同じ定義にする', () => {
    const places: Places = {
      features: [inputAt('x', 1), paramAt('x_fc', 'in_features'), paramAt('head', 'out_features')],
    }
    const { spec, notes, values } = viaTemplate('diffusion_denoiser', denoiser, places, generated)
    expect(values).toEqual({ features: 8 })
    expect(spec).toEqual(applyDatasetInputs(denoiser, generated, units).spec)
    expect(notes[0]).toContain('問題 顧客の表 が表の生成なので')
    expect(notes).toContain(
      '定義は雛形 diffusion_denoiser のままなので, データに合わせる引数を雛形の書き換え ' +
        '(adapt_example) で, 関わるノードの引数と入力の形をそろえて書き換えました',
    )
    expect(notes.at(-1)).toBe(
      '雛形の引数 features (特徴の数) を 16 から 8 にしました ' +
        '(入力 x の形の 2 番目, x_fc の in_features, head の out_features)',
    )
    expect(notes.join('\n')).not.toContain('head の out_features を 8 にしました')
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
  /** 学習用の利用者 6 人と品物 4 つ (学習のプロジェクトの学習の設定の割合で分けた) */
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
      '問題 会員の品 が並べるなので, 入力 user の大きさを学習用の利用者の数 6, ' +
        'item の大きさを学習用の品物の数 4 にしました (学習用の反応の行の数 31 (時刻の順に分けたもの). ' +
        '分ける割合は学習のプロジェクトの学習の設定の検証用 0.1 とテスト用 0.1. ' +
        '検証用とテスト用にだけ出る利用者と品物は数えません)',
    )
    expect(notes[1]).toBe('反応の強さ (qty) が 0 以下か数でない 1 行を除きました')
    expect(notes).toContain(
      '出力 score は利用者と品物の全ての組の点数の表なので, 出力の大きさは変えていません',
    )
    expect(datasetSummary(reactions)).toBe('反応の表から入れました: 入力 user [4, 6], item [4, 4]')
  })

  it('雛形 two_tower_retrieval のままなら, 書き換えた雛形に入れて今と同じ定義にする', () => {
    const places: Places = {
      users: [inputAt('user', 1), paramAt('user_fc1', 'in_features')],
      items: [inputAt('item', 1), paramAt('item_fc1', 'in_features')],
    }
    const { spec, notes, values } = viaTemplate('two_tower_retrieval', tower, places, reactions)
    expect(values).toEqual({ users: 6, items: 4 })
    expect(spec).toEqual(applyDatasetInputs(tower, reactions, units).spec)
    expect(notes[1]).toBe('反応の強さ (qty) が 0 以下か数でない 1 行を除きました')
    expect(notes).toContain(
      '雛形の引数 users (利用者の数) を 12 から 6 にしました (入力 user の形の 2 番目, user_fc1 の in_features)',
    )
    expect(notes.at(-1)).toBe(
      '出力 score は利用者と品物の全ての組の点数の表なので, 出力の大きさは変えていません',
    )
  })

  it('config に割合が無ければ既定の割合と知らせる', () => {
    const rank = { ...reactions.rank!, valid: null, test: null, problem: null }
    const { notes } = applyDatasetInputs(tower, { ...reactions, rank }, units)
    expect(notes[0]).toContain('並べるの問題なので')
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
      '問題 機械の異常 が異常の検知 (自己符号化器)なので, 入力と出力の幅を, 学習に使う列の数 3 にしました ' +
        '(自己符号化器は入力を作り直すので, 出力を入力と同じ幅にします. 正解の列 is_anomaly は数えません)',
    )
    expect(notes).toContain('dec2 の out_features を 3 にしました')
    expect(datasetSummary(readings)).toBe(
      'データセットから入れました: 入力 x [4, 3], タスク anomaly, 目的の列 is_anomaly, 出す値 3',
    )
  })

  it('雛形 tabular_autoencoder のままなら, 縮めた幅は渡さずに今と同じ定義にする', () => {
    const places: Places = {
      features: [inputAt('x', 1), paramAt('enc1', 'in_features'), paramAt('dec2', 'out_features')],
      bottleneck: [paramAt('enc2', 'out_features'), paramAt('dec1', 'in_features')],
    }
    const { spec, notes, values } = viaTemplate(
      'tabular_autoencoder',
      autoencoder,
      places,
      readings,
    )
    expect(values).toEqual({ features: 3 })
    expect(spec).toEqual(applyDatasetInputs(autoencoder, readings, units).spec)
    expect(notes.at(-1)).toBe(
      '雛形の引数 features (特徴の数) を 16 から 3 にしました ' +
        '(入力 x の形の 2 番目, enc1 の in_features, dec2 の out_features)',
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
      '問題 機械の異常 が異常の検知なので, 入力と出力の幅を, 学習に使う列の数 3 にしました ' +
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
      '問題 機械の異常 の度合いの決め方は距離に基づく度合い (学習しない) なので, ' +
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
      '問題 クーポンの効果 が効果の見積もりなので, 入力の幅を, 学習に使う列の数 5 にしました. ' +
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

  it('雛形 tabular_uplift のままなら, 書き換えた雛形に入れて今と同じ定義にし, 出力は 1 のままと知らせる', () => {
    const places: Places = {
      features: [
        inputAt('x', 1),
        paramAt('treated1', 'in_features'),
        paramAt('control1', 'in_features'),
      ],
    }
    const { spec, notes, values } = viaTemplate('tabular_uplift', twoHeads, places, log)
    expect(values).toEqual({ features: 5 })
    expect(spec).toEqual(applyDatasetInputs(twoHeads, log, units).spec)
    expect(notes).toContain(
      '雛形の引数 features (特徴の数) を 16 から 5 にしました ' +
        '(入力 x の形の 2 番目, treated1 の in_features, control1 の in_features)',
    )
    expect(notes.at(-1)).toBe(
      '出力 treated と control の値の数は, 雛形の 1 のままです ' +
        '(効果の見積もりの 2 つの出力は, どちらも 1 行に 1 つの値を出します)',
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

describe('applyDatasetInputs (値段を変える判断)', () => {
  /** 同梱の例 tabular_price の形 (特徴の 16 列と値段の 1 列を通し, 最後に softplus) */
  const sales: Spec = {
    ...emptySpec(),
    inputs: { x: { shape: [8, 17] } },
    nodes: {
      units1: { unit: 'nn.Linear', params: { in_features: 17, out_features: 32 }, inputs: 'x' },
      units_act: { unit: 'nn.ReLU', inputs: 'units1' },
      units2: {
        unit: 'nn.Linear',
        params: { in_features: 32, out_features: 1 },
        inputs: 'units_act',
      },
      units_softplus: { unit: 'nn.Softplus', inputs: 'units2' },
    },
    outputs: { units: 'units_softplus' },
  }
  /** 特徴の 4 列と値段の 1 列の値段を変えた記録の表 */
  const record: DatasetInfo = {
    path: '/data/sales',
    inputs: [{ name: 'x', shape: [4, 5], dtype: 'float32', dataset_shape: [null, 5] }],
    task: 'price',
    target: 'units',
    classes: null,
    outputs: 1,
    price: {
      columns: 5,
      price: 'price',
      units: 'units',
      cost: 'unit_cost',
      product: 'product_id',
      id: 'record_id',
      problem: '値段の見直し',
    },
  }

  /** 出力の値の数を変えないことの知らせ */
  const kept =
    '出力の値の数は変えていません (売れた数の予測は 1 行に 1 つの値なので, 雛形の 1 のまま使います)'

  it('入力の幅を特徴の列と値段の列の数にし, 出力の値の数は変えず, 決め方を初めに知らせる', () => {
    const { spec, notes } = applyDatasetInputs(sales, record, units)
    expect(spec.inputs.x!.shape).toEqual([4, 5])
    expect(spec.nodes.units1!.params).toEqual({ in_features: 5, out_features: 32 })
    expect(spec.nodes.units2!.params).toEqual({ in_features: 32, out_features: 1 })
    expect(notes[0]).toBe(
      '問題 値段の見直し が値段を変える判断なので, ' +
        '入力の幅を, 商品の特徴の列と値段の列を合わせた 5 列にしました. ' +
        '値段 (商品の売り値) の列 price は, 入力の最後の列です. ' +
        '次の列は入力に入れません: 売れた数 (予測する値) の列 units, ' +
        '原価 (仕入れか作るのにかかる値段. 利益の計算に使う) の列 unit_cost, ' +
        '商品の id (値段を選ぶ単位) の列 product_id, 行を見分ける id の列 record_id',
    )
    expect(notes).toContain('units1 の in_features を 5 にしました')
    expect(notes.at(-1)).toBe(kept)
    expect(datasetSummary(record)).toBe(
      'データセットから入れました: 入力 x [4, 5], タスク price, 目的の列 units, 出す値 1',
    )
  })

  it('雛形 tabular_price のままなら, 書き換えた雛形に入れて今と同じ定義にし, 出力は変えないと知らせる', () => {
    const places: Places = { features: [inputAt('x', 1), paramAt('units1', 'in_features')] }
    const { spec, notes, values } = viaTemplate('tabular_price', sales, places, record)
    expect(values).toEqual({ features: 5 })
    expect(spec).toEqual(applyDatasetInputs(sales, record, units).spec)
    expect(notes).toContain(
      '雛形の引数 features (特徴の数) を 17 から 5 にしました (入力 x の形の 2 番目, units1 の in_features)',
    )
    expect(notes.at(-1)).toBe(kept)
  })

  it('出力の前の全結合の値の数は書き換えず, 無い原価と商品の id の列と分からない幅は出さない', () => {
    const wide: Spec = {
      ...sales,
      nodes: {
        ...sales.nodes,
        units2: { ...sales.nodes.units2!, params: { in_features: 32, out_features: 3 } },
      },
    }
    const bare: DatasetInfo = {
      ...record,
      price: { columns: null, price: null, units: 'units', cost: null, product: null, id: null },
    }
    const { spec, notes } = applyDatasetInputs(wide, bare, units)
    expect(spec.nodes.units2!.params).toEqual({ in_features: 32, out_features: 3 })
    expect(notes[0]).toBe(
      '値段を変える判断のデータセットなので, ' +
        '入力の幅は, データセットの入力の幅のままにしました (商品の特徴の列と値段の列を合わせた数). ' +
        '値段 (商品の売り値) の列は, 入力の最後の列です. ' +
        '次の列は入力に入れません: 売れた数 (予測する値) の列 units, 行を見分ける id の列',
    )
    expect(notes.at(-1)).toBe(kept)
  })

  it('値段を変える判断でないデータセットでは, softplus の前を書き換えない', () => {
    const regression: DatasetInfo = { ...record, task: 'regression', price: null }
    const { spec, notes } = applyDatasetInputs(sales, regression, units)
    expect(spec.nodes.units2!.params).toEqual({ in_features: 32, out_features: 1 })
    expect(notes).toContain('出力の大きさ 1 は, 最後の部品の引数に手で入れてください')
  })
})

describe('applyDatasetInputs (時間をまたぐ量の判断)', () => {
  /** 同梱の例 tabular_demand の形 (16 列を通し, 9 つの水準の値に softplus をかけて前から足す) */
  const quantiles: Spec = {
    ...emptySpec(),
    inputs: { x: { shape: [8, 16] } },
    nodes: {
      demand1: { unit: 'nn.Linear', params: { in_features: 16, out_features: 32 }, inputs: 'x' },
      demand_act: { unit: 'nn.ReLU', inputs: 'demand1' },
      demand2: {
        unit: 'nn.Linear',
        params: { in_features: 32, out_features: 9 },
        inputs: 'demand_act',
      },
      demand_softplus: { unit: 'nn.Softplus', inputs: 'demand2' },
      demand_cumsum: { unit: 'op.CumSum', params: { dim: 1 }, inputs: 'demand_softplus' },
    },
    outputs: { quantiles: 'demand_cumsum' },
  }
  /** 学習に使う 5 列の需要の記録の表と, 問題の設定の 4 つの水準 */
  const record: DatasetInfo = {
    path: '/data/demand',
    inputs: [{ name: 'x', shape: [4, 5], dtype: 'float32', dataset_shape: [null, 5] }],
    task: 'demand',
    target: 'demand',
    classes: null,
    outputs: 4,
    demand: {
      columns: 5,
      units: 'units',
      product: 'product_id',
      date: 'date',
      stock: 'stock',
      id: 'record_id',
      levels: [0.1, 0.5, 0.75, 0.9],
      levels_from: 'problem',
      notes: [],
      problem: '発注の見直し',
    },
  }

  it('入力の幅を学習に使う列の数にし, 出力の値の数を水準の数にして, 決め方を知らせる', () => {
    const { spec, notes } = applyDatasetInputs(quantiles, record, units)
    expect(spec.inputs.x!.shape).toEqual([4, 5])
    expect(spec.nodes.demand1!.params).toEqual({ in_features: 5, out_features: 32 })
    expect(spec.nodes.demand2!.params).toEqual({ in_features: 32, out_features: 4 })
    expect(spec.nodes.demand_cumsum!.params).toEqual({ dim: 1 })
    expect(notes[0]).toBe(
      '問題 発注の見直し が時間をまたぐ量の判断なので, ' +
        '入力の幅を, 学習に使う 5 列にしました ' +
        '(商品の特徴, 過去の売れた数のずらしと移動平均, 曜日などの列). ' +
        '次の列は入力に入れません: ' +
        '日ごとの売れた数 (先の期間の合計が予測する値) の列 units, ' +
        '商品の id (発注の数を決める単位) の列 product_id, 日付 (発注を決める日) の列 date, ' +
        '在庫の数 (発注の数から引く) の列 stock, 行を見分ける id の列 record_id',
    )
    expect(notes).toContain('demand1 の in_features を 5 にしました')
    expect(notes.at(-1)).toBe(
      '出力 quantiles の値の数を, 学ぶ分位点の水準 0.1, 0.5, 0.75, 0.9 の数 4 ' +
        '(問題 発注の見直し の水準. 水準ごとに 1 つの値) にしました ' +
        '(demand2 の out_features を 9 から 4 に)',
    )
    expect(datasetSummary(record)).toBe(
      'データセットから入れました: 入力 x [4, 5], タスク demand, 目的の列 demand, 出す値 4',
    )
  })

  it('雛形 tabular_demand のままなら, 水準の数を渡して今と同じ定義にし, 水準の決め方を知らせる', () => {
    const places: Places = {
      features: [inputAt('x', 1), paramAt('demand1', 'in_features')],
      levels: [paramAt('demand2', 'out_features')],
    }
    const { spec, notes, values } = viaTemplate('tabular_demand', quantiles, places, record)
    expect(values).toEqual({ features: 5, levels: 4 })
    expect(spec).toEqual(applyDatasetInputs(quantiles, record, units).spec)
    expect(notes.at(-1)).toBe(
      '出力 quantiles の値の数を, 学ぶ分位点の水準 0.1, 0.5, 0.75, 0.9 の数 4 ' +
        '(問題 発注の見直し の水準. 水準ごとに 1 つの値) にしました ' +
        '(雛形の引数 levels を 9 から 4 に: demand2 の out_features)',
    )
    // 水準が分からなければ水準の数は渡さず, 今の値のまま
    const unknown = { ...record, outputs: null, demand: { ...record.demand!, levels: null } }
    const without = viaTemplate('tabular_demand', quantiles, places, unknown)
    expect(without.values).toEqual({ features: 5 })
    expect(without.spec).toEqual(applyDatasetInputs(quantiles, unknown, units).spec)
    expect(without.notes.at(-1)).toContain('学ぶ分位点の水準の数は問題が決めるので')
  })

  it('学習の設定の水準と食い違いの知らせを入れ, 同じ数なら値の数を変えずに知らせる', () => {
    const configured: DatasetInfo = {
      ...record,
      outputs: 3,
      demand: {
        ...record.demand!,
        levels: [0.25, 0.5, 0.75],
        levels_from: 'config',
        notes: ['学習の設定の水準の数 3 は, 問題の水準の数 4 と違います'],
      },
    }
    const three: Spec = {
      ...quantiles,
      nodes: {
        ...quantiles.nodes,
        demand2: { ...quantiles.nodes.demand2!, params: { in_features: 32, out_features: 3 } },
      },
    }
    const { spec, notes } = applyDatasetInputs(three, configured, units)
    expect(spec.nodes.demand2!.params).toEqual({ in_features: 32, out_features: 3 })
    expect(notes[1]).toBe('学習の設定の水準の数 3 は, 問題の水準の数 4 と違います')
    expect(notes.at(-1)).toBe(
      '出力 quantiles の値の数は, 学ぶ分位点の水準 0.25, 0.5, 0.75 の数 3 ' +
        '(学習のプロジェクトの学習の設定の水準. 水準ごとに 1 つの値) のままです (demand2 の out_features)',
    )
  })

  it('水準が分からなければ出力の値の数を変えず, 無い在庫の数の列と分からない名前は出さない', () => {
    const bare: DatasetInfo = {
      ...record,
      outputs: null,
      demand: { columns: null, units: 'units', stock: null, levels: null, notes: [] },
    }
    const { spec, notes } = applyDatasetInputs(quantiles, bare, units)
    expect(spec.nodes.demand2!.params).toEqual({ in_features: 32, out_features: 9 })
    expect(notes[0]).toBe(
      '需要の記録のデータセットなので, 入力の幅は, データセットの入力の幅のままにしました ' +
        '(商品の特徴, 過去の売れた数のずらしと移動平均, 曜日などの列). ' +
        '次の列は入力に入れません: ' +
        '日ごとの売れた数 (先の期間の合計が予測する値) の列 units, ' +
        '商品の id (発注の数を決める単位) の列, 日付 (発注を決める日) の列, 行を見分ける id の列',
    )
    expect(notes.at(-1)).toBe(
      '学ぶ分位点の水準の数は問題が決めるので, 出力の値の数は変えていません. ' +
        '時間をまたぐ量の判断の問題から保存先を確かめてから入れ直すと, 水準の数にします',
    )
  })

  it('形を変える部品の後ろの出力は, 手で入れるよう知らせる', () => {
    const reshaped: Spec = {
      ...quantiles,
      nodes: {
        ...quantiles.nodes,
        demand_cumsum: { unit: 'op.Reshape', params: { shape: [9] }, inputs: 'demand_softplus' },
      },
    }
    const { spec, notes } = applyDatasetInputs(reshaped, record, units)
    expect(spec.nodes.demand2!.params).toEqual({ in_features: 32, out_features: 9 })
    expect(notes.at(-1)).toBe(
      '出力の値の数 4 (学ぶ分位点の水準 0.1, 0.5, 0.75, 0.9 の数. 問題 発注の見直し の水準) は, ' +
        '最後の部品の引数に手で入れてください',
    )
  })
})

describe('applyDatasetInputs (雛形のままのファイルの組)', () => {
  /** 同梱の例 unet_segmentation の入力と最初と最後の畳み込み (3 チャンネル, 2 クラス) */
  const unet: Spec = {
    ...emptySpec(),
    inputs: { image: { shape: [4, 3, 32, 32] } },
    nodes: {
      enc1a: {
        unit: 'nn.Conv2d',
        params: { in_channels: 3, out_channels: 16, kernel_size: 3 },
        inputs: 'image',
      },
      head: {
        unit: 'nn.Conv2d',
        params: { in_channels: 16, out_channels: 2, kernel_size: 1 },
        inputs: 'enc1a',
      },
    },
    outputs: { logits: 'head' },
  }
  /** 同梱の例 point_cloud_predictor の入力と復号と並べ直し (入力 512 点, 出す点 256) */
  const predictor: Spec = {
    ...emptySpec(),
    inputs: { observed: { shape: [4, 512, 3] } },
    nodes: {
      encode1: {
        unit: 'nn.Linear',
        params: { in_features: 3, out_features: 64 },
        inputs: 'observed',
      },
      decode2: {
        unit: 'nn.Linear',
        params: { in_features: 64, out_features: 768 },
        inputs: 'encode1',
      },
      cloud: { unit: 'op.Reshape', params: { shape: [256, 3] }, inputs: 'decode2' },
    },
    outputs: { cloud: 'cloud' },
  }

  it('雛形 unet_segmentation のままなら, チャンネル数とクラス数を渡して今と同じ定義にする', () => {
    const places: Places = {
      channels: [inputAt('image', 1), paramAt('enc1a', 'in_channels')],
      classes: [paramAt('head', 'out_channels')],
    }
    const { spec, notes, values } = viaTemplate('unet_segmentation', unet, places, masks)
    expect(values).toEqual({ channels: 1, classes: 3 })
    expect(spec).toEqual(applyDatasetInputs(unet, masks, units).spec)
    expect(spec.inputs.image!.shape).toEqual([4, 1, 32, 32])
    expect(notes.at(-1)).toBe(
      '雛形の引数 classes (クラス数 (背景を含む)) を 2 から 3 にしました (head の out_channels)',
    )
    // タスクの無い (出す値の数の分からない) データセットはクラス数を渡さない
    const unknown = { ...masks, outputs: null }
    const without = viaTemplate('unet_segmentation', unet, places, unknown)
    expect(without.values).toEqual({ channels: 1 })
    expect(without.notes.at(-1)).toBe(
      'タスクを決めていないデータセットなので, 出力の大きさは変えていません',
    )
  })

  /** 点群の雛形の引数の場所 */
  const cloudPlaces: Places = {
    points_in: [inputAt('observed', 1)],
    points_out: [
      paramAt('decode2', 'out_features', 3),
      { kind: 'shape', owner: 'cloud', key: 0, scale: 1 },
    ],
  }

  it('雛形 point_cloud_predictor のままなら, 入力の点の数と出す点の数を渡して今と同じ定義にする', () => {
    const observed: DatasetInfo = {
      ...clouds,
      inputs: [{ ...clouds.inputs[0]!, name: 'observed', shape: [4, 128, 3] }],
    }
    const { spec, notes, values } = viaTemplate(
      'point_cloud_predictor',
      predictor,
      cloudPlaces,
      observed,
    )
    expect(values).toEqual({ points_in: 128, points_out: 1024 })
    expect(spec).toEqual(applyDatasetInputs(predictor, observed, units).spec)
    expect(spec.inputs.observed!.shape).toEqual([4, 128, 3])
    expect(spec.nodes.decode2!.params).toEqual({ in_features: 64, out_features: 3072 })
    expect(spec.nodes.cloud!.params).toEqual({ shape: [1024, 3] })
    expect(notes).toContain(
      '雛形の引数 points_in (入力の点の数) を 512 から 128 にしました (入力 observed の形の 2 番目)',
    )
    expect(notes.at(-1)).toBe(
      '雛形の引数 points_out (出す点の数) を 256 から 1024 にしました ' +
        '(decode2 の out_features (× 3), cloud の shape の 1 番目)',
    )
  })

  it('雛形の入力と名前の違うデータセットの入力は, 雛形の入力の名前を変えずに入れない', () => {
    const { spec, notes, values } = viaTemplate(
      'point_cloud_predictor',
      predictor,
      cloudPlaces,
      clouds,
    )
    expect(values).toEqual({ points_in: 512, points_out: 1024 })
    expect(Object.keys(spec.inputs)).toEqual(['observed'])
    expect(spec.nodes.encode1!.inputs).toBe('observed')
    expect(spec.nodes.cloud!.params).toEqual({ shape: [1024, 3] })
    expect(notes).toContain(
      'データセットの入力 capture は雛形 point_cloud_predictor の入力 (observed) と名前が違うので, ' +
        '定義に入れていません (雛形の入力の名前は学習のスクリプトとの約束なので変えません)',
    )
    expect(notes.join('\n')).not.toContain('の名前を capture にしました')
  })

  it('次元の数が雛形の入力と違うデータセットの入力からは, 引数を決めずに知らせる', () => {
    const flat: DatasetInfo = {
      ...clouds,
      inputs: [{ name: 'observed', shape: [4, 7], dtype: 'float32' }],
    }
    const { values, notes } = templateValues(
      {
        name: 'point_cloud_predictor',
        args: [{ name: 'points_in', value: 512, places: [inputAt('observed', 1)] }],
      },
      predictor,
      flat,
    )
    expect(values).toEqual({})
    expect(notes).toEqual([
      'データセットの入力 observed [4, 7] は雛形の入力 observed と次元の数が違うので, ' +
        '雛形の引数 points_in (入力の点の数) は今の 512 のままです',
    ])
  })

  it('GAN の生成器は雑音の幅を決めず, 出す 1 行の幅をデータセットの幅にし, 出す値の数を使わないと知らせる', () => {
    const generator: Spec = {
      ...emptySpec(),
      inputs: { z: { shape: [8, 16] } },
      nodes: {
        fc1: { unit: 'nn.Linear', params: { in_features: 16, out_features: 32 }, inputs: 'z' },
        fc2: { unit: 'nn.Linear', params: { in_features: 32, out_features: 10 }, inputs: 'fc1' },
      },
      outputs: { fake: 'fc2' },
    }
    const places: Places = {
      noise: [inputAt('z', 1), paramAt('fc1', 'in_features')],
      features: [paramAt('fc2', 'out_features')],
    }
    const { spec, notes, values } = viaTemplate('gan_generator', generator, places, table)
    expect(values).toEqual({ features: 7 })
    expect(Object.keys(spec.inputs)).toEqual(['z'])
    expect(spec.inputs.z!.shape).toEqual([8, 16])
    expect(spec.nodes.fc1!.params).toEqual({ in_features: 16, out_features: 32 })
    expect(spec.nodes.fc2!.params).toEqual({ in_features: 32, out_features: 7 })
    expect(notes).toContain(
      '雛形の引数 noise (雑音の幅) はデータの幅でないので, データセットから決めずに今の 16 のままです',
    )
    expect(notes.at(-1)).toBe(
      'データセットの出す値の数 3 は使っていません (雛形 gan_generator の出力の大きさは雛形の決まりで決まります. ' +
        '出力 fake は作る行なので 1 行の幅)',
    )
  })

  it('データセットで決まらない軸の引数は渡さない', () => {
    const match: TemplateMatch = {
      name: 'point_cloud_predictor',
      args: [{ name: 'points_in', value: 512, places: [inputAt('observed', 1)] }],
    }
    const varied: DatasetInfo = {
      ...clouds,
      inputs: [{ ...clouds.inputs[0]!, dataset_shape: [null, null, 3] }],
    }
    expect(templateValues(match, predictor, varied)).toEqual({ values: {}, notes: [] })
  })
})

describe('sameDefinition', () => {
  it('名前だけが違う定義は同じとみなし, 中身が違えば違う', () => {
    const spec: Spec = { ...emptySpec(), inputs: { x: { shape: [4, 3] } } }
    expect(sameDefinition({ ...spec, name: 'a' }, { ...spec, name: 'b' })).toBe(true)
    expect(sameDefinition(spec, { ...spec, inputs: { x: { shape: [4, 5] } } })).toBe(false)
  })
})
