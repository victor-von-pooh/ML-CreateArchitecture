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

  it('写す定義の部分モデルも写し, 中身の違う同じ名前は誤り', () => {
    const inner = embedSubgraph(emptySpec('outer'), 'block', block()).spec
    const outer: Spec = { ...inner, nodes: { b: { unit: 'sub.block', inputs: { x: 'x' } } } }
    const result = embedSubgraph(emptySpec(), 'outer', outer)
    expect(Object.keys(result.spec.subgraphs ?? {}).sort()).toEqual(['block', 'outer'])
    expect(result.spec.subgraphs?.outer).not.toHaveProperty('subgraphs')

    const conflict = { ...result.spec, subgraphs: { block: { ...block(), description: '別' } } }
    expect(() => embedSubgraph(conflict, 'outer', outer)).toThrow('block')
  })

  it('使っているノードを探し, 消すと subgraphs も空なら無くなる', () => {
    const spec = embedSubgraph(emptySpec(), 'block', block()).spec
    const used: Spec = { ...spec, nodes: { b: { unit: 'sub.block', inputs: { x: 'x' } } } }
    expect(subgraphUsers(used, 'block')).toEqual(['b'])
    expect(subgraphUsers(spec, 'block')).toEqual([])
    expect(removeSubgraph(spec, 'block')).not.toHaveProperty('subgraphs')
  })
})
