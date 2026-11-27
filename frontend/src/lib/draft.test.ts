import { act, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'

import { readDraft, tabId, useDraftWriter, writeDraft, type Draft } from './draft'
import { emptySpec, type Spec } from './graph'

/** 部品を 1 つ置いた定義 */
function changed(): Spec {
  return { ...emptySpec(), nodes: { linear1: { unit: 'nn.Linear', args: {} } } } as Spec
}

/** 書く下書き */
function nextOf(spec: Spec) {
  return { spec, fileName: 'model', ownFile: null }
}

describe('useDraftWriter', () => {
  beforeEach(() => {
    window.localStorage.clear()
    window.sessionStorage.clear()
    vi.useFakeTimers()
  })
  afterEach(() => {
    vi.useRealTimers()
  })

  it('変更は少し待ってから書く', () => {
    renderHook(() => useDraftWriter(true, nextOf(changed()), null))
    expect(readDraft()).toBeNull()
    act(() => {
      vi.advanceTimersByTime(1000)
    })
    expect(Object.keys(readDraft()?.spec.nodes ?? {})).toEqual(['linear1'])
  })

  it('変更してすぐ外すと, 待たずに書く', () => {
    const { unmount } = renderHook(() => useDraftWriter(true, nextOf(changed()), null))
    unmount()
    const draft = readDraft()
    expect(draft?.tab).toBe(tabId())
    expect(Object.keys(draft?.spec.nodes ?? {})).toEqual(['linear1'])
  })

  it('変更が無ければ, 外しても書かない', () => {
    const { unmount } = renderHook(() => useDraftWriter(false, nextOf(emptySpec()), null))
    unmount()
    expect(readDraft()).toBeNull()
  })

  it('聞いている間は待って書かず, 外すときに前の下書きを残して書く', () => {
    // この画面の前の下書き (戻すかを聞いている)
    writeDraft({
      spec: emptySpec('old'),
      fileName: 'old',
      ownFile: null,
      time: new Date().toISOString(),
    })
    const asking = readDraft() as Draft
    const { unmount } = renderHook(() => useDraftWriter(true, nextOf(changed()), asking))
    act(() => {
      vi.advanceTimersByTime(5000)
    })
    expect(readDraft()?.fileName).toBe('old')
    unmount()
    // 前の下書きはこの画面の下書きのまま残し, 聞いている間の変更は別の画面の下書きとして書く
    const drafts = JSON.parse(
      window.localStorage.getItem('ml-createarchitecture:drafts') ?? '{}',
    ) as Record<string, Draft>
    expect(drafts[tabId()]?.fileName).toBe('old')
    const aside = Object.values(drafts).filter((item) => item.tab !== tabId())
    expect(aside.map((item) => Object.keys(item.spec.nodes))).toEqual([['linear1']])
  })

  it('聞き終えてすぐ外すと, この画面の印で書き, 別に書いた下書きを消す', () => {
    writeDraft({
      spec: emptySpec('old'),
      fileName: 'old',
      ownFile: null,
      time: new Date().toISOString(),
    })
    const asking = readDraft() as Draft
    const { rerender, unmount } = renderHook(
      ({ shown, key }: { shown: Draft | null; key: string }) =>
        useDraftWriter(true, nextOf(changed()), shown, key),
      { initialProps: { shown: asking as Draft | null, key: 'a' } },
    )
    // 聞いている間に殻で替えると, 別の画面の下書きとして書く
    rerender({ shown: asking, key: 'b' })
    // 「下書きを捨てる」で聞き終え, 待つ間 (1 秒以内) に外す
    rerender({ shown: null, key: 'b' })
    unmount()
    const drafts = JSON.parse(
      window.localStorage.getItem('ml-createarchitecture:drafts') ?? '{}',
    ) as Record<string, Draft>
    expect(Object.keys(drafts)).toEqual([tabId()])
    expect(Object.keys(drafts[tabId()]?.spec.nodes ?? {})).toEqual(['linear1'])
  })

  it('殻で選んだ問題の設定が替わると, 待たずに書く', () => {
    const { rerender } = renderHook(
      ({ project }: { project: string | null }) =>
        useDraftWriter(true, nextOf(changed()), null, project),
      { initialProps: { project: 'aaaaaaaaaaaa' } },
    )
    expect(readDraft()).toBeNull()
    rerender({ project: 'bbbbbbbbbbbb' })
    expect(Object.keys(readDraft()?.spec.nodes ?? {})).toEqual(['linear1'])
  })
})
