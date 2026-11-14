import { useCallback, useRef, useState } from 'react'

// 続けて起きた変更 (ノードのドラッグなど) を 1 回分にまとめる間隔 (ミリ秒)
const MERGE_MS = 500

// 戻せる回数の上限
const LIMIT = 100

/** 値の変更の履歴 (戻すための過去と, やり直すための未来) */
interface History<T> {
  past: T[]
  present: T
  future: T[]
}

/** 値と, 元に戻す・やり直すの操作 */
export interface HistoryState<T> {
  value: T
  /** 変更 (履歴に残す. 続けて起きた変更は 1 回分にまとめる) */
  set: (next: T | ((current: T) => T)) => void
  /** 履歴に残さない変更 (保存した名前の反映など) */
  replace: (next: T | ((current: T) => T)) => void
  /** 履歴を消して値を置き換える (定義を開いた場合と新規) */
  reset: (next: T) => void
  undo: () => void
  redo: () => void
  canUndo: boolean
  canRedo: boolean
}

/** 履歴の 1 歩を進める (純粋な関数. テストで使う) */
export function pushHistory<T>(history: History<T>, next: T, merge: boolean): History<T> {
  if (Object.is(next, history.present)) return history
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

/** 元に戻す・やり直すができる値 */
export function useHistory<T>(initial: () => T): HistoryState<T> {
  const [history, setHistory] = useState<History<T>>(() => ({
    past: [],
    present: initial(),
    future: [],
  }))
  // 最後に履歴に残した時刻 (続けて起きた変更をまとめるため)
  const last = useRef(0)

  const set = useCallback((next: T | ((current: T) => T)) => {
    const now = Date.now()
    const merge = now - last.current < MERGE_MS
    last.current = now
    setHistory((current) => {
      const value = typeof next === 'function' ? (next as (value: T) => T)(current.present) : next
      return pushHistory(current, value, merge && current.past.length > 0)
    })
  }, [])

  const replace = useCallback((next: T | ((current: T) => T)) => {
    setHistory((current) => ({
      ...current,
      present: typeof next === 'function' ? (next as (value: T) => T)(current.present) : next,
    }))
  }, [])

  const reset = useCallback((next: T) => {
    last.current = 0
    setHistory({ past: [], present: next, future: [] })
  }, [])

  // 戻した直後の変更は, 前の変更とまとめない
  const undo = useCallback(() => {
    last.current = 0
    setHistory(undoHistory)
  }, [])
  const redo = useCallback(() => {
    last.current = 0
    setHistory(redoHistory)
  }, [])

  return {
    value: history.present,
    set,
    replace,
    reset,
    undo,
    redo,
    canUndo: history.past.length > 0,
    canRedo: history.future.length > 0,
  }
}
