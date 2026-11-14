import { describe, expect, it } from 'vitest'

import type { ParamSpec } from './graph'
import { checkValue } from './params'

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
    expect(checkValue({ ...base, kind: 'float' }, 'x')).toBe('数を入れてください')
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
