import { describe, expect, it } from 'vitest'

import { pushHistory, redoHistory, undoHistory } from './history'

describe('history', () => {
  it('戻してやり直し, 新しい変更でやり直しを捨てる', () => {
    let history = { past: [] as number[], present: 0, future: [] as number[] }
    history = pushHistory(history, 1, false)
    history = pushHistory(history, 2, false)
    history = undoHistory(history)
    expect(history.present).toBe(1)
    history = redoHistory(history)
    expect(history.present).toBe(2)
    history = undoHistory(undoHistory(history))
    expect(history.present).toBe(0)
    expect(undoHistory(history)).toBe(history)
    history = pushHistory(history, 5, false)
    expect(history.future).toEqual([])
    expect(redoHistory(history)).toBe(history)
  })

  it('まとめる変更は 1 回分にし, 同じ値は残さない', () => {
    let history = { past: [] as number[], present: 0, future: [] as number[] }
    history = pushHistory(history, 1, false)
    history = pushHistory(history, 2, true)
    history = pushHistory(history, 3, true)
    expect(history.past).toEqual([0])
    expect(pushHistory(history, 3, false)).toBe(history)
    const objects = { past: [], present: { a: [1] }, future: [] }
    expect(pushHistory(objects, { a: [1] }, false)).toBe(objects)
    expect(undoHistory(history).present).toBe(0)
  })
})
