import { describe, expect, it } from 'vitest'

import type { UnitSpec } from './graph'
import { firstMatch } from './search'

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
