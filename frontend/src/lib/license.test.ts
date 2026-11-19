import { describe, expect, it } from 'vitest'

import { licenseParams } from './license'

describe('licenseParams', () => {
  it('ライセンスに関わる引数だけを決まった順に並べる', () => {
    const text = licenseParams(
      { num_labels: 3, revision: 'main', model_name: 'org/model', pretrained: true },
      ['pretrained'],
    )
    expect(text).toBe('[["model_name","org/model"],["pretrained",true],["revision","main"]]')
    // 関わらない引数を変えても同じ文字
    expect(
      licenseParams(
        { num_labels: 5, revision: 'main', model_name: 'org/model', pretrained: true },
        ['pretrained'],
      ),
    ).toBe(text)
  })

  it('重みを変える引数 (timm の pretrained_cfg など) も送る', () => {
    const text = licenseParams(
      { model_name: 'resnet50', pretrained: true, pretrained_cfg: 'fb_swsl_ig1b_ft_in1k' },
      ['pretrained'],
    )
    expect(text).toContain('"pretrained_cfg","fb_swsl_ig1b_ft_in1k"')
    expect(licenseParams({ model_name: 'resnet50', pretrained: true }, ['pretrained'])).not.toBe(
      text,
    )
  })
})
