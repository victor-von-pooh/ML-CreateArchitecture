import { afterEach, describe, expect, it } from 'vitest'

import { outsideStage } from './stage'

afterEach(() => {
  document.body.innerHTML = ''
})

describe('outsideStage', () => {
  it('部品の範囲の外 (殻の帯) の要素だけを外とし, 中と body は外としない', () => {
    document.body.innerHTML =
      '<header><select id="shell"></select></header>' +
      '<div data-stage-root><button id="inner"></button></div>'
    expect(outsideStage(document.getElementById('shell'))).toBe(true)
    expect(outsideStage(document.getElementById('inner'))).toBe(false)
    expect(outsideStage(document.body)).toBe(false)
    expect(outsideStage(null)).toBe(false)
  })
})
