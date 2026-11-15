import { describe, expect, it } from 'vitest'

import {
  embedSubgraph,
  emptySpec,
  removeSubgraph,
  specWires,
  subgraphUnits,
  subgraphUsers,
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
})
