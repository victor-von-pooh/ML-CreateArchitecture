import { describe, expect, it } from 'vitest'

import type { UnitSpec } from './graph'
import { groupKey, groupLabel, groupUnits, isModuleGroup } from './groups'

/** 一覧の並びを確かめるだけの部品 */
function unit(name: string, source: string, module = ''): UnitSpec {
  return {
    name,
    category: '',
    source,
    description: '',
    params: [],
    inputs: [],
    outputs: [],
    forward_params: [],
    extra_params: false,
    available: true,
    reason: '',
    module,
  }
}

describe('部品のまとまり', () => {
  it('起動の設定のモジュールの自作の部品は, モジュールごとに組み込みの部品の後', () => {
    const groups = groupUnits([
      unit('zeta.Block', 'custom', 'zeta_lab'),
      unit('nn.Linear', 'torch.nn'),
      unit('my.Direct', 'custom'),
      unit('alpha.Block', 'custom', 'alpha_lab.units'),
      unit('op.Add', 'ml_modelcontrol'),
    ])
    expect(groups.map(([key]) => groupLabel(key))).toEqual([
      '演算',
      'PyTorch',
      '自作',
      '自作 (alpha_lab.units)',
      '自作 (zeta_lab)',
    ])
  })

  it('鍵と表示名', () => {
    expect(groupKey(unit('a.B', 'custom', 'lab'))).toBe('custom:lab')
    expect(groupKey(unit('a.B', 'custom'))).toBe('custom')
    expect(isModuleGroup('custom:lab')).toBe(true)
    expect(isModuleGroup('custom')).toBe(false)
    expect(groupLabel('unknown_source')).toBe('unknown_source')
  })
})
