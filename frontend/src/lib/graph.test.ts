import { describe, expect, it } from 'vitest'

import { parseShape } from './params'
import {
  addNode,
  connect,
  disconnect,
  emptySpec,
  downstream,
  pasteNode,
  removeItem,
  refOptions,
  renameItem,
  setPortRefs,
  specWires,
  type Spec,
  type UnitSpec,
} from './graph'

/** テスト用の部品の定義 */
function unit(name: string, inputs: string[], outputs = ['out'], variadic = false): UnitSpec {
  return {
    name,
    category: 'test',
    source: 'test',
    description: '',
    params: [],
    inputs: inputs.map((port) => ({ name: port, description: '', optional: false, variadic })),
    outputs: outputs.map((port) => ({
      name: port,
      description: '',
      optional: false,
      variadic: false,
    })),
    forward_params: [],
    extra_params: false,
    available: true,
    reason: '',
  }
}

const units = new Map([
  ['nn.Linear', unit('nn.Linear', ['input'])],
  ['op.Add', unit('op.Add', ['xs'], ['out'], true)],
  ['nn.LSTM', unit('nn.LSTM', ['input'], ['output', 'h_n', 'c_n'])],
])

describe('graph', () => {
  it('部品を置いて線でつなぎ, 線とノードを消す', () => {
    let spec: Spec = addNode(emptySpec(), units.get('nn.Linear')!, { x: 0, y: 0 })
    expect(Object.keys(spec.nodes)).toEqual(['linear1'])
    spec = connect(spec, units, {
      source: 'x',
      sourcePort: 'out',
      target: 'linear1',
      targetPort: 'input',
    })
    expect(spec.nodes.linear1!.inputs).toEqual({ input: 'x' })
    spec = connect(spec, units, {
      source: 'linear1',
      sourcePort: 'out',
      target: 'output:y',
      targetPort: 'in',
    })
    expect(spec.outputs).toEqual({ y: 'linear1' })
    const wires = specWires(spec, units)
    expect(wires.map((wire) => wire.id)).toEqual([
      'x:out->linear1:input:0',
      'linear1:out->output:y',
    ])
    spec = disconnect(spec, units, 'x:out->linear1:input:0')
    expect(spec.nodes.linear1!.inputs).toEqual({})
    spec = removeItem(spec, units, 'linear1')
    expect(spec.nodes).toEqual({})
    expect(spec.outputs).toEqual({})
  })

  it('可変の数の口は線を加え, 出力が複数の部品は口の名前で参照する', () => {
    let spec: Spec = addNode(emptySpec(), units.get('op.Add')!, { x: 0, y: 0 })
    spec = addNode(spec, units.get('nn.LSTM')!, { x: 0, y: 0 })
    spec = connect(spec, units, {
      source: 'x',
      sourcePort: 'out',
      target: 'add1',
      targetPort: 'xs',
    })
    spec = connect(spec, units, {
      source: 'lstm1',
      sourcePort: 'h_n',
      target: 'add1',
      targetPort: 'xs',
    })
    expect(spec.nodes.add1!.inputs).toEqual({ xs: ['x', 'lstm1:h_n'] })
    expect(specWires(spec, units)[1]!.sourcePort).toBe('h_n')
  })

  it('名前を変えると, それを参照するつなぎ先も変わる', () => {
    let spec: Spec = addNode(emptySpec(), units.get('nn.Linear')!, { x: 0, y: 0 })
    spec = connect(spec, units, {
      source: 'x',
      sourcePort: 'out',
      target: 'linear1',
      targetPort: 'input',
    })
    spec = { ...spec, outputs: { y: 'linear1' } }
    spec = renameItem(spec, 'x', 'features')
    spec = renameItem(spec, 'linear1', 'head')
    expect(Object.keys(spec.inputs)).toEqual(['features'])
    expect(spec.nodes.head!.inputs).toEqual({ input: 'features' })
    expect(spec.outputs).toEqual({ y: 'head' })
  })

  it('同じノードを 2 回つないだ可変の口と出力から, 消したノードを外す', () => {
    let spec: Spec = addNode(emptySpec(), units.get('op.Add')!, { x: 0, y: 0 })
    spec = addNode(spec, units.get('nn.Linear')!, { x: 0, y: 0 })
    spec = {
      ...spec,
      nodes: {
        ...spec.nodes,
        add1: { ...spec.nodes.add1!, inputs: { xs: ['linear1', 'x', 'linear1'] } },
      },
      outputs: { y: 'linear1' },
    }
    spec = removeItem(spec, units, 'linear1')
    expect(spec.nodes.add1!.inputs).toEqual({ xs: ['x'] })
    expect(spec.outputs).toEqual({})
    // 出力の名前は残る (つなぎ直せるように)
    expect(Object.keys(spec.ui?.outputs ?? {})).toEqual(['y'])
  })
})

describe('inputs editor', () => {
  it('つなぎ先の候補と, 口のつなぎ先の置き換え', () => {
    let spec: Spec = addNode(emptySpec(), units.get('nn.LSTM')!, { x: 0, y: 0 })
    spec = addNode(spec, units.get('op.Add')!, { x: 0, y: 0 })
    expect(refOptions(spec, units, 'add1').map((option) => option.value)).toEqual([
      'lstm1:output',
      'lstm1:h_n',
      'lstm1:c_n',
      'x',
    ])
    spec = setPortRefs(spec, units, 'add1', 'xs', ['x', 'lstm1:h_n'])
    expect(spec.nodes.add1!.inputs).toEqual({ xs: ['x', 'lstm1:h_n'] })
    spec = setPortRefs(spec, units, 'add1', 'xs', [])
    expect(spec.nodes.add1!.inputs).toEqual({})
  })
})

describe('params', () => {
  it('形は [4, 16] と 4, 16 のどちらでも読み, 読めない値は null', () => {
    expect(parseShape('[4, 16]')).toEqual([4, 16])
    expect(parseShape('4, 16')).toEqual([4, 16])
    expect(parseShape([2, 3])).toEqual([2, 3])
    expect(parseShape('4, a')).toBeNull()
    expect(parseShape('')).toBeNull()
  })

  it('ノードの写しは, つなぎ先を除いて独立に写す', () => {
    const source = {
      unit: 'nn.Linear',
      params: { in_features: 4, sizes: [1, 2] },
      inputs: 'x',
      ui: { x: 10, y: 20, note: 'keep' },
    }
    let spec: Spec = { ...emptySpec(), nodes: { linear1: source } }
    const pasted = pasteNode(spec, source, { x: 50, y: 60 })
    spec = pasted.spec
    expect(pasted.name).toBe('linear2')
    const copy = spec.nodes.linear2!
    expect(copy.inputs).toEqual({})
    expect(copy.ui).toEqual({ x: 50, y: 60, note: 'keep' })
    ;(copy.params!.sizes as number[]).push(3)
    expect(source.params.sizes).toEqual([1, 2])
  })

  it('先のノード (つなぐと循環になる) を候補から除く', () => {
    const spec: Spec = {
      ...emptySpec(),
      nodes: {
        a: { unit: 'nn.Linear', inputs: { input: 'x' } },
        b: { unit: 'nn.Linear', inputs: { input: 'a' } },
        c: { unit: 'nn.Linear', inputs: { input: 'b' } },
      },
    }
    expect([...downstream(spec, units, 'a')].sort()).toEqual(['b', 'c'])
    expect(refOptions(spec, units, 'a').map((option) => option.value)).toEqual(['x'])
    expect(refOptions(spec, units, 'c').map((option) => option.value)).toEqual(['b', 'a', 'x'])
    // 循環していても, 既につないであるものは候補に残す
    const cycle: Spec = {
      ...spec,
      nodes: { ...spec.nodes, a: { unit: 'nn.Linear', inputs: { input: 'c' } } },
    }
    expect(refOptions(cycle, units, 'a').map((option) => option.value)).toContain('c')
  })
})
