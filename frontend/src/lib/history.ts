import { useCallback, useRef, useState } from 'react'

// 同じ印の続けて起きた変更 (ノードのドラッグなど) を 1 回分にまとめる間隔
// (ミリ秒)
const MERGE_MS = 2000

// 戻せる回数の上限
const LIMIT = 100

/** 値の変更の履歴 (戻すための過去と, やり直すための未来) */
export interface History<T> {
  past: T[]
  present: T
  future: T[]
}

/** 変更の指定 (新しい値か, 今の値から新しい値を作る関数) */
type Update<T> = T | ((current: T) => T)

/** 値と, 元に戻す・やり直すの操作 */
export interface HistoryState<T> {
  value: T
  /**
   * 変更 (履歴に残す). merge に同じ印を付けた続けての変更 (ドラッグなど)
   * は 1 回分にまとめる. 中身の同じ変更は残さない
   */
  set: (next: Update<T>, merge?: string) => void
  /** 履歴に残さない変更 (保存した名前の反映など) */
  replace: (next: Update<T>) => void
  /** 履歴を消して値を置き換える (定義を開いた場合と新規) */
  reset: (next: T) => void
  undo: () => void
  redo: () => void
  /** 元に戻した場合の値 (戻せない場合は undefined) */
  previous: T | undefined
  /** やり直した場合の値 (やり直せない場合は undefined) */
  next: T | undefined
}

/** 2 つの値の中身が同じか (JSON にした文字で比べる) */
function same<T>(a: T, b: T): boolean {
  return Object.is(a, b) || JSON.stringify(a) === JSON.stringify(b)
}

/** 履歴の 1 歩を進める (純粋な関数. 中身が同じなら変えない) */
export function pushHistory<T>(history: History<T>, next: T, merge: boolean): History<T> {
  if (same(next, history.present)) return history
  const past = merge ? history.past : [...history.past, history.present].slice(-LIMIT)
  return { past, present: next, future: [] }
}

/** 1 回分戻す (戻せない場合はそのまま) */
export function undoHistory<T>(history: History<T>): History<T> {
  if (history.past.length === 0) return history
  const previous = history.past[history.past.length - 1] as T
  return {
    past: history.past.slice(0, -1),
    present: previous,
    future: [history.present, ...history.future],
  }
}

/** 1 回分やり直す (やり直せない場合はそのまま) */
export function redoHistory<T>(history: History<T>): History<T> {
  if (history.future.length === 0) return history
  const next = history.future[0] as T
  return {
    past: [...history.past, history.present].slice(-LIMIT),
    present: next,
    future: history.future.slice(1),
  }
}

/** 変更の指定から新しい値を作る */
function apply<T>(next: Update<T>, current: T): T {
  return typeof next === 'function' ? (next as (value: T) => T)(current) : next
}

/** 元に戻す・やり直すができる値 */
export function useHistory<T>(initial: () => T): HistoryState<T> {
  const [history, setHistory] = useState<History<T>>(() => ({
    past: [],
    present: initial(),
    future: [],
  }))
  // 最後に履歴に残した変更の印と時刻 (同じ印の続けての変更をまとめるため)
  const last = useRef<{ merge: string | undefined; time: number }>({
    merge: undefined,
    time: 0,
  })

  const set = useCallback((next: Update<T>, merge?: string) => {
    const now = Date.now()
    const together =
      merge !== undefined && last.current.merge === merge && now - last.current.time < MERGE_MS
    last.current = { merge, time: now }
    setHistory((current) =>
      pushHistory(current, apply(next, current.present), together && current.past.length > 0),
    )
  }, [])

  const replace = useCallback((next: Update<T>) => {
    setHistory((current) => ({ ...current, present: apply(next, current.present) }))
  }, [])

  const reset = useCallback((next: T) => {
    last.current = { merge: undefined, time: 0 }
    setHistory({ past: [], present: next, future: [] })
  }, [])

  // 戻した直後の変更は, 前の変更とまとめない
  const undo = useCallback(() => {
    last.current = { merge: undefined, time: 0 }
    setHistory(undoHistory)
  }, [])
  const redo = useCallback(() => {
    last.current = { merge: undefined, time: 0 }
    setHistory(redoHistory)
  }, [])

  return {
    value: history.present,
    set,
    replace,
    reset,
    undo,
    redo,
    previous: history.past[history.past.length - 1],
    next: history.future[0],
  }
}
