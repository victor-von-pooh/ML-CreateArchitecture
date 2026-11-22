import { describe, expect, it } from 'vitest'

import { formatBytes, licenseParams } from './license'

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

describe('formatBytes', () => {
  it('1024 ごとに単位を上げ, 10 未満は小数 1 桁にする', () => {
    expect(formatBytes(0)).toBe('0 B')
    expect(formatBytes(1023)).toBe('1023 B')
    expect(formatBytes(1536)).toBe('1.5 KB')
    expect(formatBytes(440 * 1024 * 1024)).toBe('440 MB')
    expect(formatBytes(1.5 * 1024 ** 3)).toBe('1.5 GB')
    expect(formatBytes(2048 * 1024 ** 4)).toBe('2048 TB')
  })

  it('単位を上げるかは丸めた後の値で決める', () => {
    expect(formatBytes(1048575)).toBe('1.0 MB')
    expect(formatBytes(1023.6 * 1024)).toBe('1.0 MB')
    expect(formatBytes(9.97 * 1024 ** 2)).toBe('10 MB')
    expect(formatBytes(9.94 * 1024 ** 2)).toBe('9.9 MB')
    expect(formatBytes(1024 ** 3 - 1)).toBe('1.0 GB')
  })

  it('負の値, 無限, NaN は null', () => {
    expect(formatBytes(-1)).toBeNull()
    expect(formatBytes(Number.NaN)).toBeNull()
    expect(formatBytes(Number.POSITIVE_INFINITY)).toBeNull()
  })
})
