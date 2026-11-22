import { describe, expect, it } from 'vitest'

import { emptySpec, type Spec } from './graph'
import { startsFresh, templateAction } from './template'

describe('startsFresh', () => {
  it('新規の定義は, 保存していない変更が無ければ新しく始める定義', () => {
    expect(startsFresh(emptySpec(), false)).toBe(true)
    // 入力の形だけを変えた (保存していない変更がある) 定義は含めない
    expect(startsFresh(emptySpec(), true)).toBe(false)
  })

  it('部品, 出力, 部分モデルのどれかがあれば, 既に定義がある', () => {
    const base = emptySpec()
    const withNode: Spec = {
      ...base,
      nodes: { fc: { unit: 'nn.Linear', params: {}, inputs: 'x' } },
    }
    const withOutput: Spec = { ...base, outputs: { y: 'x' } }
    const withSubgraph: Spec = {
      ...base,
      subgraphs: { block: { inputs: {}, nodes: {}, outputs: {} } },
    }
    for (const spec of [withNode, withOutput, withSubgraph]) {
      expect(startsFresh(spec, false)).toBe(false)
    }
  })
})

describe('templateAction', () => {
  it('一覧から選び, 新しく始める定義で, 保存先に定義が無ければそのまま開く', () => {
    expect(
      templateAction({ example: 'unet_segmentation', chosen: true, fresh: true, exists: false }),
    ).toBe('open')
  })

  it('既に定義があるか, 場所を打って確かめた場合は, 開かずに選べるようにする', () => {
    for (const state of [
      { chosen: true, fresh: false, exists: false },
      { chosen: true, fresh: true, exists: true },
      { chosen: false, fresh: true, exists: false },
    ]) {
      expect(templateAction({ example: 'unet_segmentation', ...state })).toBe('offer')
    }
  })

  it('雛形が無ければ今までどおり', () => {
    for (const example of [null, undefined, '']) {
      expect(templateAction({ example, chosen: true, fresh: true, exists: false })).toBeNull()
    }
  })
})
