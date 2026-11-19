import { describe, expect, it } from 'vitest'

import { emptySpec, renameItem, type Spec } from './graph'
import {
  formatShareLines,
  parseNames,
  parseShareLines,
  setNodeState,
  switchRuleKind,
  switchWeightsSource,
  usesState,
  weightsSource,
  withStateVersion,
} from './state'

function linearSpec(): Spec {
  return {
    ...emptySpec(),
    nodes: {
      embed: { unit: 'nn.Embedding', params: { num_embeddings: 10, embedding_dim: 4 } },
      head: { unit: 'nn.Linear', params: { in_features: 4, out_features: 10 } },
    },
  }
}

describe('パラメータの状態', () => {
  it('版 2 の項目を書くと版を 2 にし, 消すと項目を除く', () => {
    const spec = linearSpec()
    expect(spec.format_version).toBe(1)
    expect(usesState(spec)).toBe(false)
    const frozen = setNodeState(spec, 'embed', 'train', false)
    expect(frozen.format_version).toBe(2)
    expect(frozen.nodes.embed!.train).toBe(false)
    const back = setNodeState(frozen, 'embed', 'train', undefined)
    expect('train' in back.nodes.embed!).toBe(false)
  })

  it('部分モデルの中の項目でも版を 2 にする', () => {
    const spec: Spec = {
      ...linearSpec(),
      subgraphs: {
        block: { inputs: {}, nodes: { fc: { unit: 'nn.Linear', zero_init: true } }, outputs: {} },
      },
    }
    expect(withStateVersion(spec).format_version).toBe(2)
  })

  it('名前と共有の行を読み書きする', () => {
    expect(parseNames('weight, bias  nodes.fc.weight')).toEqual([
      'weight',
      'bias',
      'nodes.fc.weight',
    ])
    expect(parseShareLines('weight = embed.weight\n')).toEqual({ weight: 'embed.weight' })
    expect(typeof parseShareLines('weight embed')).toBe('string')
    expect(typeof parseShareLines('')).toBe('string')
    expect(formatShareLines({ weight: 'embed.weight' })).toBe('weight = embed.weight')
  })

  it('名前を変えると共有の元の名前も書き換える', () => {
    let spec = setNodeState(linearSpec(), 'head', 'share', { weight: 'embed.weight' })
    spec = renameItem(spec, 'embed', 'tok')
    expect(spec.nodes.head!.share).toEqual({ weight: 'tok.weight' })
    spec = setNodeState(spec, 'head', 'share', 'tok')
    expect(renameItem(spec, 'tok', 'emb').nodes.head!.share).toBe('emb')
  })

  it('外部の重みの出どころを切り替えても付け替えと strict を残す', () => {
    const weights = {
      path: 'a.pt',
      remap: [{ from: 'model.', to: 'nodes.' }],
      strict: false,
    }
    expect(weightsSource(weights)).toBe('path')
    const hub = switchWeightsSource(weights, 'hub')
    expect(hub).toEqual({ hub: '', file: '', remap: weights.remap, strict: false })
    expect(weightsSource(hub)).toBe('hub')
    expect(weightsSource(switchWeightsSource(hub, 'unit'))).toBe('unit')
    expect(switchWeightsSource(hub, 'none')).toBeUndefined()
    expect(switchRuleKind({ from: 'a.', to: 'b.' }, 'pattern')).toEqual({ pattern: 'a.', to: 'b.' })
  })
})
