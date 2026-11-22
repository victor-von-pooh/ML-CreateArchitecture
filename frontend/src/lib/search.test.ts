import { describe, expect, it } from 'vitest'

import type { UnitSpec } from './graph'
import { firstMatch, matchesQuery } from './search'

/** テスト用の部品 (名前と説明と使えるか) */
function unit(name: string, description = '', available = true): UnitSpec {
  return {
    name,
    category: 'test',
    source: 'test',
    description,
    params: [],
    inputs: [],
    outputs: [],
    forward_params: [],
    extra_params: false,
    available,
    reason: '',
  }
}

describe('firstMatch', () => {
  it('名前の一致, 名前の終わり, 名前に含む, 説明に含むの順で使える部品を選ぶ', () => {
    const units = [
      unit('op.Add', 'Linear の後に足す'),
      unit('nn.LazyLinear', '', false),
      unit('nn.Bilinear'),
      unit('nn.Linear'),
    ]
    expect(firstMatch(units, 'nn.linear')?.name).toBe('nn.Linear')
    expect(firstMatch(units, 'linear')?.name).toBe('nn.Linear')
    expect(firstMatch(units, 'bilin')?.name).toBe('nn.Bilinear')
    expect(firstMatch(units, '足す')?.name).toBe('op.Add')
    expect(firstMatch(units, '  ')).toBeUndefined()
  })
})

describe('よく使う呼び方', () => {
  it('他のライブラリの関数の名前でも部品が見つかる', () => {
    const units = [
      unit('pyg.GCNConv'),
      unit('monai.DenseNet'),
      unit('pyg.MeanAggregation'),
      unit('nn.Linear'),
    ]
    expect(firstMatch(units, 'global_mean_pool')?.name).toBe('pyg.MeanAggregation')
    expect(firstMatch(units, 'dense')?.name).toBe('nn.Linear')
    expect(matchesQuery(unit('pyg.MeanAggregation'), 'readout')).toBe(true)
    expect(matchesQuery(unit('pyg.GCNConv'), 'readout')).toBe(false)
  })

  it('点群のまとめの最大値は max pool の呼び方で見つかる', () => {
    const units = [unit('op.Mean'), unit('op.Max')]
    expect(firstMatch(units, 'max pool')?.name).toBe('op.Max')
    expect(firstMatch(units, 'global max pool')?.name).toBe('op.Max')
    expect(matchesQuery(unit('op.Max'), 'maximum')).toBe(true)
    // グラフのまとめの名前そのものは, グラフの部品を先に置く
    const graph = [unit('op.Max'), unit('pyg.MaxAggregation')]
    expect(firstMatch(graph, 'global max pool')?.name).toBe('pyg.MaxAggregation')
    expect(matchesQuery(unit('op.Max'), 'global max pool')).toBe(true)
  })

  it('max pool の呼び方は, グラフのまとめと並べても op.Max を先に置く', () => {
    for (const names of [
      ['pyg.MaxAggregation', 'op.Max'],
      ['op.Max', 'pyg.MaxAggregation'],
    ]) {
      const units = names.map((name) => unit(name))
      expect(firstMatch(units, 'max pool')?.name).toBe('op.Max')
    }
  })

  it('max の語では, 呼び方より名前の決まりの順で置く', () => {
    for (const names of [
      ['nn.MaxPool2d', 'op.Max'],
      ['op.Max', 'nn.MaxPool2d'],
    ]) {
      const units = names.map((name) => unit(name))
      // 名前の終わりが一致する op.Max が先 (足した呼び方に依らない)
      expect(firstMatch(units, 'max')?.name).toBe('op.Max')
      // MaxPool の名前の語では nn.MaxPool2d のまま
      expect(firstMatch(units, 'maxpool')?.name).toBe('nn.MaxPool2d')
      expect(firstMatch(units, 'maxpool2d')?.name).toBe('nn.MaxPool2d')
    }
  })
})
