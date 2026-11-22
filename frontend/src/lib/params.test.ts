import { describe, expect, it } from 'vitest'

import type { ParamSpec, UnitSpec } from './graph'
import { checkValue, isUnitValue, pickUnit, setUnitParam, unitOptions } from './params'

// 確かめる引数 (種類と範囲だけを変える)
const base: ParamSpec = {
  name: 'p',
  kind: 'int',
  default: null,
  required: false,
  description: '',
  minimum: null,
  maximum: null,
  choices: null,
}

describe('checkValue', () => {
  it('種類に合わない値の理由を返す', () => {
    expect(checkValue(base, 'abc')).toBe('整数を入れてください')
    expect(checkValue(base, '1.5')).toBe('整数を入れてください')
    expect(checkValue({ ...base, kind: 'float' }, 'x')).toBe('有限の数を入れてください')
    expect(checkValue({ ...base, kind: 'float' }, 'Infinity')).toBe('有限の数を入れてください')
    expect(checkValue(base, 'null')).toBeNull()
    expect(checkValue({ ...base, required: true }, 'null')).toBe('必須の引数です')
    expect(checkValue(base, '3')).toBeNull()
    expect(checkValue({ ...base, kind: 'float' }, '1e-5')).toBeNull()
  })

  it('範囲と必須を確かめる', () => {
    const ranged = { ...base, minimum: 1, maximum: 4 }
    expect(checkValue(ranged, '0')).toBe('1 以上にしてください')
    expect(checkValue(ranged, '5')).toBe('4 以下にしてください')
    expect(checkValue({ ...base, required: true }, ' ')).toBe('必須の引数です')
    expect(checkValue(base, '')).toBeNull()
    expect(checkValue({ ...base, kind: 'any' }, '[1, 2]')).toBeNull()
  })
})

describe('引数に書く部品', () => {
  // 登録簿の部品 (使えるか, 重みを読むかだけを変える)
  const unit = (name: string, extra: Partial<UnitSpec> = {}): UnitSpec => ({
    name,
    category: '',
    source: '',
    description: '',
    params: [],
    inputs: [],
    outputs: [],
    forward_params: [],
    extra_params: false,
    available: true,
    reason: '',
    ...extra,
  })

  it('候補を先に, 書けない部品を除いて選択肢を作る', () => {
    const units = [
      unit('nn.Linear'),
      unit('nn.TransformerEncoderLayer'),
      unit('nn.LayerNorm'),
      unit('sub.block'),
      unit('nn.Broken', { available: false }),
      unit('hf.Pretrained', { weights_params: ['name'] }),
    ]
    expect(unitOptions(units, ['nn.TransformerEncoderLayer', 'nn.Missing'])).toEqual({
      suggested: ['nn.TransformerEncoderLayer'],
      others: ['nn.LayerNorm', 'nn.Linear'],
    })
    expect(unitOptions(units, null).suggested).toEqual([])
  })

  it('部品を選び直すと, 別の部品なら引数を空にし, 空なら未設定', () => {
    const value = { unit: 'nn.LayerNorm', params: { normalized_shape: 16 } }
    expect(pickUnit(value, 'nn.LayerNorm')).toBe(value)
    expect(pickUnit(value, 'nn.RMSNorm')).toEqual({ unit: 'nn.RMSNorm' })
    expect(pickUnit(value, '')).toBeUndefined()
    expect(pickUnit(3, 'nn.LayerNorm')).toEqual({ unit: 'nn.LayerNorm' })
  })

  it('引数を書き換え, 最後の引数を消すと params を除く', () => {
    const value = setUnitParam({ unit: 'nn.LayerNorm' }, 'normalized_shape', 16)
    expect(value).toEqual({ unit: 'nn.LayerNorm', params: { normalized_shape: 16 } })
    expect(setUnitParam(value, 'normalized_shape', undefined)).toEqual({ unit: 'nn.LayerNorm' })
  })

  it('unit を持つ辞書だけを部品の値とみなす', () => {
    expect(isUnitValue({ unit: 'nn.Linear' })).toBe(true)
    expect(isUnitValue({ params: {} })).toBe(false)
    expect(isUnitValue(['nn.Linear'])).toBe(false)
    expect(isUnitValue(null)).toBe(false)
  })
})
