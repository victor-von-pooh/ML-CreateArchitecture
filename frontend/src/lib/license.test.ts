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
})
