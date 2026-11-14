import { describe, expect, it } from 'vitest'

import type { Spec, UnitSpec } from './graph'
import { arrangeSpec, layoutPositions, placeMissing } from './layout'

/** テスト用の部品 (入力の口と, 1 つの出力の口) */
function unit(name: string, inputs: string[], variadic = false): UnitSpec {
  return {
    name,
    category: 'test',
    source: 'test',
    description: '',
    params: [],
    inputs: inputs.map((port) => ({ name: port, description: '', optional: false, variadic })),
    outputs: [{ name: 'out', description: '', optional: false, variadic: false }],
    forward_params: [],
    extra_params: false,
    available: true,
    reason: '',
  }
}

const units = new Map([
  ['nn.Linear', unit('nn.Linear', ['input'])],
  ['op.Add', unit('op.Add', ['xs'], true)],
])

// 残差のつなぎ方 (x → fc1 → fc2, x と fc2 を足す)
const residual: Spec = {
  format_version: 1,
  name: 'residual',
  inputs: { x: { shape: [4, 8] } },
  nodes: {
    add: { unit: 'op.Add', inputs: { xs: ['x', 'fc2'] } },
    fc1: { unit: 'nn.Linear', inputs: { input: 'x' } },
    fc2: { unit: 'nn.Linear', inputs: { input: 'fc1' } },
  },
  outputs: { y: 'add' },
}

describe('layout', () => {
  it('つながりの深さで段に分け, 出力は最後の段', () => {
    const positions = layoutPositions(residual, units)
    const x = (id: string) => positions.get(id)!.x
    expect(x('x')).toBeLessThan(x('fc1'))
    expect(x('fc1')).toBeLessThan(x('fc2'))
    expect(x('fc2')).toBeLessThan(x('add'))
    expect(x('add')).toBeLessThan(x('output:y'))
  })

  it('並べ直した位置を ui に書き, 循環があっても止まる', () => {
    const arranged = arrangeSpec(residual, units)
    expect(arranged.nodes.fc1!.ui).toEqual(layoutPositions(residual, units).get('fc1'))
    expect(arranged.ui?.outputs?.y).toBeDefined()
    const cycle: Spec = {
      ...residual,
      nodes: {
        a: { unit: 'nn.Linear', inputs: { input: 'b' } },
        b: { unit: 'nn.Linear', inputs: { input: 'a' } },
      },
      outputs: {},
    }
    expect(layoutPositions(cycle, units).size).toBe(3)
  })

  it('位置の無いものにだけ位置を書く', () => {
    const partly: Spec = {
      ...residual,
      nodes: { ...residual.nodes, fc1: { ...residual.nodes.fc1!, ui: { x: 999, y: 5 } } },
    }
    const placed = placeMissing(partly, units)
    expect(placed.nodes.fc1!.ui).toEqual({ x: 999, y: 5 })
    expect(placed.nodes.fc2!.ui).toEqual(layoutPositions(partly, units).get('fc2'))
    expect(placed.inputs.x!.ui).toBeDefined()
    expect(placed.ui?.outputs?.y).toBeDefined()
  })
})
