import { describe, expect, it } from 'vitest'

import {
  embedSubgraph,
  emptySpec,
  removeSubgraph,
  specWires,
  subgraphUnits,
  subgraphUsers,
  subgraphView,
  makesCycle,
  subgraphOrigin,
  usageShapes,
  withInputShapes,
  subgraphCheckTarget,
  writeSubgraph,
  type Spec,
} from './graph'

/** 部分モデルとして写す, 入力 2 つと出力 2 つの定義 */
function block(): Spec {
  return {
    format_version: 1,
    name: 'block',
    description: '残差',
    inputs: { x: { shape: [4, 8] }, mask: {} },
    nodes: { fc: { unit: 'nn.Linear', params: { in_features: 8, out_features: 8 }, inputs: 'x' } },
    outputs: { y: 'fc', skip: 'x' },
  }
}

describe('subgraphs', () => {
  it('写した定義は部分モデルの部品になり, 入力と出力が口になる', () => {
    const { spec, name } = embedSubgraph(emptySpec(), 'my-block', block())
    expect(name).toBe('my_block')
    expect(spec.subgraphs?.my_block).not.toHaveProperty('format_version')
    const [unit] = subgraphUnits(spec)
    expect(unit?.name).toBe('sub.my_block')
    expect(unit?.description).toBe('残差')
    expect(unit?.inputs.map((port) => port.name)).toEqual(['x', 'mask'])
    expect(unit?.outputs.map((port) => port.name)).toEqual(['y', 'skip'])

    // 部分モデルのノードの線は口の名前で引ける
    const used: Spec = {
      ...spec,
      nodes: { b: { unit: 'sub.my_block', inputs: { x: 'x', mask: 'x' } } },
      outputs: { o: 'b:y' },
    }
    const units = new Map(subgraphUnits(used).map((item) => [item.name, item]))
    expect(specWires(used, units).map((wire) => `${wire.source}:${wire.sourcePort}`)).toContain(
      'b:y',
    )
  })

  it('同じ名前の違う中身は番号を付け, 同じ中身は 1 つにまとめる', () => {
    const first = embedSubgraph(emptySpec(), 'block', block()).spec
    expect(embedSubgraph(first, 'block', block()).name).toBe('block')
    const changed = block()
    changed.description = '別'
    const second = embedSubgraph(first, 'block', changed)
    expect(second.name).toBe('block_2')
    expect(Object.keys(second.spec.subgraphs ?? {})).toEqual(['block', 'block_2'])
  })

  it('写す定義の部分モデルも写し, 中身の違う同じ名前は番号を付けて中の参照も直す', () => {
    const inner = embedSubgraph(emptySpec('outer'), 'block', block()).spec
    const outer: Spec = { ...inner, nodes: { b: { unit: 'sub.block', inputs: { x: 'x' } } } }
    const result = embedSubgraph(emptySpec(), 'outer', outer)
    expect(Object.keys(result.spec.subgraphs ?? {}).sort()).toEqual(['block', 'outer'])
    expect(result.spec.subgraphs?.outer).not.toHaveProperty('subgraphs')

    const conflict: Spec = {
      ...emptySpec(),
      subgraphs: { block: { ...block(), description: '別' } },
    }
    const renamed = embedSubgraph(conflict, 'outer', outer).spec
    expect(renamed.subgraphs?.block?.description).toBe('別')
    expect(renamed.subgraphs?.block_2?.description).toBe('残差')
    expect(renamed.subgraphs?.outer?.nodes.b?.unit).toBe('sub.block_2')
  })

  it('予約語と, 書き出すクラスの名前 (Model) と重なる名前は付け直す', () => {
    expect(embedSubgraph(emptySpec(), 'class', block()).name).toBe('class_')
    expect(embedSubgraph(emptySpec(), 'None', block()).name).toBe('None_')
    expect(embedSubgraph(emptySpec(), 'model', block()).name).toBe('model_2')
    const first = embedSubgraph(emptySpec(), 'res_block', block()).spec
    const changed = { ...block(), description: '別' }
    // resBlock と res_block は同じクラスの名前 ResBlock になる
    expect(embedSubgraph(first, 'resBlock', changed).name).toBe('resBlock_2')
  })

  it('使っているノードを探し, 消すと subgraphs も空なら無くなる', () => {
    const spec = embedSubgraph(emptySpec(), 'block', block()).spec
    const used: Spec = { ...spec, nodes: { b: { unit: 'sub.block', inputs: { x: 'x' } } } }
    expect(subgraphUsers(used, 'block')).toEqual({ nodes: ['b'], subgraphs: [] })
    expect(subgraphUsers(spec, 'block')).toEqual({ nodes: [], subgraphs: [] })
    const nested = embedSubgraph(emptySpec(), 'outer', {
      ...used,
      outputs: { y: 'b:y' },
    }).spec
    expect(subgraphUsers(nested, 'block')).toEqual({ nodes: [], subgraphs: ['outer'] })
    expect(removeSubgraph(spec, 'block')).not.toHaveProperty('subgraphs')
  })

  it('部分モデルを 1 つの定義として編集し, 書き戻すと他は変わらない', () => {
    const spec = embedSubgraph(emptySpec(), 'block', block()).spec
    const view = subgraphView(spec, 'block')
    expect(view.format_version).toBe(1)
    expect(Object.keys(view.nodes)).toEqual(['fc'])
    const edited: Spec = {
      ...view,
      nodes: { ...view.nodes, act: { unit: 'nn.ReLU', inputs: 'fc' } },
    }
    const written = writeSubgraph(spec, 'block', edited)
    expect(Object.keys(written.subgraphs?.block?.nodes ?? {})).toEqual(['fc', 'act'])
    expect(written.subgraphs?.block).not.toHaveProperty('format_version')
    expect(written.subgraphs?.block).not.toHaveProperty('subgraphs')
    expect(written.inputs).toEqual(spec.inputs)
    // 変えずに書き戻すと同じ中身
    expect(JSON.stringify(writeSubgraph(spec, 'block', view))).toBe(JSON.stringify(spec))
  })

  it('置くと循環になる部分モデルと, 確かめるときに持たせる部分モデル', () => {
    const inner = embedSubgraph(emptySpec(), 'block', block()).spec
    const outer: Spec = {
      ...inner,
      nodes: { b: { unit: 'sub.block', inputs: { x: 'x' } } },
      outputs: { y: 'b:y' },
    }
    const spec = embedSubgraph(
      embedSubgraph(emptySpec(), 'outer', outer).spec,
      'other',
      block(),
    ).spec
    // block の中に, block 自身と block を使う outer は置けない. other は置ける
    expect(makesCycle(spec.subgraphs, 'block', 'sub.block')).toBe(true)
    expect(makesCycle(spec.subgraphs, 'block', 'sub.outer')).toBe(true)
    expect(makesCycle(spec.subgraphs, 'block', 'sub.other')).toBe(false)
    expect(makesCycle(spec.subgraphs, 'block', 'nn.Linear')).toBe(false)

    // outer を確かめるときは, 中で使う block だけを持たせる
    const target = subgraphCheckTarget(subgraphView(spec, 'outer'), 'outer')
    expect(Object.keys(target.subgraphs ?? {})).toEqual(['block'])
    expect(subgraphCheckTarget(subgraphView(spec, 'block'), 'block')).not.toHaveProperty(
      'subgraphs',
    )
  })

  it('本体で使う所に入る形を, 部分モデルの入力の形にする', () => {
    const spec = embedSubgraph(emptySpec(), 'block', block()).spec
    const used: Spec = {
      ...spec,
      inputs: { x: { shape: [2, 8] } },
      nodes: {
        fc: { unit: 'nn.Linear', inputs: 'x' },
        b: { unit: 'sub.block', inputs: { x: 'fc', mask: 'x' } },
      },
    }
    const units = new Map(subgraphUnits(used).map((item) => [item.name, item]))
    // 形の推定の結果は計算の順に全てのノードを持つ
    const check = {
      nodes: {
        fc: {
          status: 'ok',
          outputs: { out: { kind: 'tensor', shape: [2, 6], dtype: 'float32' } },
        },
        b: { status: 'ok' },
      },
    }
    const usage = usageShapes(used, check, 'block', units)
    expect(usage).toEqual({
      node: 'b',
      shapes: {
        x: { shape: [2, 6], dtype: 'float32', fill: 'zeros' },
        mask: { shape: [2, 8], dtype: 'float32', fill: 'zeros' },
      },
    })
    const target = withInputShapes(subgraphView(used, 'block'), usage!.shapes)
    expect(target.inputs.x?.shape).toEqual([2, 6])
    expect(target.inputs.mask?.shape).toEqual([2, 8])
    expect(target.inputs.x?.fill).toBe('zeros')
    // 口の形が分からないノードは使わず, 次に使うノードを見る
    const two: Spec = {
      ...used,
      nodes: {
        ...used.nodes,
        a: { unit: 'sub.block', inputs: { x: 'broken', mask: 'x' } },
        b: used.nodes.b!,
      },
    }
    const order = { nodes: { a: { status: 'error' }, ...check.nodes, b: { status: 'ok' } } }
    expect(usageShapes(two, order, 'block', units)?.node).toBe('b')
    // 使うノードが無ければ null
    expect(usageShapes(spec, check, 'block', units)).toBeNull()
  })

  it('写した元を記録し, 同じ名前のまま写し直せる', () => {
    const origin = { kind: 'model' as const, name: 'block' }
    const first = embedSubgraph(emptySpec(), 'block', block(), { origin })
    expect(subgraphOrigin(first.spec.subgraphs?.block)).toEqual(origin)
    const changed = { ...block(), description: '直した' }
    const again = embedSubgraph(first.spec, 'block', changed, { origin, replace: 'block' })
    expect(again.name).toBe('block')
    expect(Object.keys(again.spec.subgraphs ?? {})).toEqual(['block'])
    expect(again.spec.subgraphs?.block?.description).toBe('直した')
    expect(subgraphOrigin(again.spec.subgraphs?.block)).toEqual(origin)
    expect(
      subgraphOrigin(embedSubgraph(emptySpec(), 'b', block()).spec.subgraphs?.b),
    ).toBeUndefined()
  })
})
